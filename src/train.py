import os
import time
import yaml
import argparse
from typing import Dict, Any

import wandb
import torch 
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from src.data import get_dataloaders
from src.model import Nano, NanoConfig


BACKEND = "nccl"


class Trainer:

    def __init__(self, 
                 model_config: NanoConfig, 
                 adamw_config: Dict[str, Any],
                 train_config: Dict[str, Any],
                 load_from: str = None):
        '''
        Initialize the TransformerTrainer.

        Args:
            model_config (NanoConfig): transformer parameters
            adamw_params (Dict[str, Any]): optimizer parameters
            train_config (Dict[str, Any]): training setup parameters
            load_from (str, optional): path to load checkpoint from.
        '''
        # save configurations
        self.model_config = model_config
        self.adamw_config = adamw_config
        self.train_config = train_config
        self.load_from = load_from 

        self.epochs = self.train_config["num_epochs"]
        self.global_step = 0

        # ddp setup
        self.ddp = int(os.environ.get("RANK", -1)) != -1
        if self.ddp:
            init_process_group(backend=BACKEND)
            self.rank = int(os.environ["RANK"])
            self.local_rank = int(os.environ["LOCAL_RANK"])
            self.world_size = int(os.environ["WORLD_SIZE"])
            self.device = torch.device(f"cuda:{self.local_rank}")
            torch.cuda.set_device(self.device)
            # master process will do logging, checkpointing
            self.master_process = self.local_rank == 0 
        else:
            self.master_process = True
            self.world_size = 1
            self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        # ensure checkpoint dir exists
        self.checkpoint_dir = train_config.get("checkpoint_dir", "./checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # load training and validation dataloaders
        self.train_loader, self.val_loader = get_dataloaders(
            self.train_config["dataset"],
            self.train_config["data_dir"], 
            model_config.max_seq_len, 
            model_config.max_batch_size,
            self.train_config.get("num_examples", 0),
            self.train_config.get("val_ratio", 0.0),
            self.train_config["n_workers"],
            pin_memory=self.device.type == "cuda",
            distributed=self.world_size > 1
        )

        # initialize model and compile
        self.model = Nano(model_config).to(self.device)
        if self.train_config.get("compile_model", True):
            self.model = torch.compile(self.model)

        if self.ddp:
            self.model = DDP(self.model, device_ids=[self.local_rank], 
                             output_device=self.local_rank, find_unused_parameters=False)

        torch.set_float32_matmul_precision(self.train_config.get("precision", "high"))
        
        # configure optimizer
        if self.world_size > 1:
            self.optimizer = self.model.module.configure_optimizer(self.adamw_config["weight_decay"],
                                                                    self.adamw_config["lr"],
                                                                    self.adamw_config["betas"])
        else:
            self.optimizer = self.model.configure_optimizer(self.adamw_config["weight_decay"],
                                                            self.adamw_config["lr"],
                                                            self.adamw_config["betas"])
        
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.train_config["num_epochs"]
        )
        
        # load checkpoint
        if load_from:
            self.global_step = self.load_checkpoint(load_from)

        # initialize logging for wandb library for master process
        if self.master_process:
            self.run_id = time.strftime("%m%d_%H%M%S")
            self.setup_wandb()

    def setup_wandb(self) -> None:
        '''Set up wandb logging.'''
        wandb.init(
            project="nano-llm", 
            name=f"run_{self.run_id}_{self.train_config['run_name']}", 
            config={
                "model_config": vars(self.model_config),
                "adamw_params": self.adamw_config,
                "train_config": self.train_config,
                "dataset": self.train_config['dataset'],
            }
        ) 

    def train(self):
        '''Train loop for the model.'''
        val_every = self.train_config.get("val_every", 1000)

        for epoch in range(self.epochs):
            # set model to train mode
            self.model.train()
            # iterate over batch
            for x, y in self.train_loader:
                x, y = x.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()

                # forward with mixed precision
                dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
                with torch.autocast(device_type=self.device.type, dtype=dtype):
                    logits = self.model(x)
                    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-100)
                
                # backprop + grad clip + step
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.train_config.get("grad_clip", 1.0))
                self.optimizer.step()
                self.scheduler.step()

                # wait for all GPUs to finish
                if self.device.type == "cuda":
                    torch.cuda.synchronize() 

                self.global_step += 1

                # wandb logging
                if self.master_process and self.global_step % self.train_config.get("log_every", 100) == 0:
                    wandb.log({"train_loss": loss.item(), "step": self.global_step})

                # validation
                if self.master_process and self.global_step % val_every == 0:
                    val_loss = self.validate()
                    wandb.log({"val_loss": val_loss, "step": self.global_step})

                    # checkpointing
                    ckpt_path = os.path.join(self.checkpoint_dir, f"checkpoint_step_{self.global_step}.pt")
                    self.save_checkpoint(self.global_step, ckpt_path)
        
        if self.ddp:
            destroy_process_group()

    def validate(self):
        '''Run validation on the val_loader and return average loss.'''
        self.model.eval()
        total_loss = 0.0
        count = 0
        with torch.no_grad():
            for x, y in self.val_loader:
                x, y = x.to(self.device), y.to(self.device)
                logits = self.model(x)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-100)
                total_loss += loss.item() * x.size(0)
                count += x.size(0)
        avg_loss = total_loss / count
        self.model.train()
        return avg_loss

    def save_checkpoint(self, iteration: int, path: str):
        '''Saves state of the model, optimizer and lr scheduler on a given iteration.'''
        to_save = {"model": self.model.module.state_dict() if self.world_size > 1 
                                                           else self.model.state_dict(),
                   "optimizer": self.optimizer.state_dict(),
                   "lr_scheduler": self.scheduler.state_dict(),
                   "iteration": iteration}
        torch.save(to_save, path)

    def load_checkpoint(self, src: str):
        '''Load state of the model, optimizer and lr scheduler.'''
        loaded = torch.load(src, map_location=self.device)
        if self.world_size > 1:
            self.model.module.load_state_dict(loaded["model"])
        else:
            self.model.load_state_dict(loaded["model"])
        self.optimizer.load_state_dict(loaded["optimizer"])
        self.scheduler.load_state_dict(loaded["lr_scheduler"])
        return loaded["iteration"]
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_config", type=str, default="config/model/nano_327m.yaml")
    parser.add_argument("--adamw_config", type=str, default="config/optimizer/adamw_327m.yaml")
    parser.add_argument("--train_config", type=str, default="config/train/train_327m.yaml")
    args = parser.parse_args()

    # Load separate YAMLs
    with open(args.model_config, "r") as f:
        model_config = NanoConfig(**yaml.safe_load(f))
    with open(args.adamw_config, "r") as f:
        adamw_config = yaml.safe_load(f)
    with open(args.train_config, "r") as f:
        train_config = yaml.safe_load(f)

    trainer = Trainer(model_config, adamw_config, train_config)
    trainer.train()


if __name__ == "__main__":
    main()
