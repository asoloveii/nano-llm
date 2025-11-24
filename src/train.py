import os
import time
import yaml
import argparse
from typing import Dict

import wandb
import torch 
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group, all_reduce, ReduceOp

from .models.nano import NanoConfig, Nano
from .data_utils import get_owt_dataloaders, get_sni_dataloaders
from .utils import load_checkpoint, save_checkpoint


class Trainer:

    def __init__(self, 
                 config: Dict, 
                 epochs: int, 
                 dataset: str, 
                 load_ckpt: str,
                 run_name: str):
        '''
        Initialize the Trainer for NanoLM pretraining, instruction-tuning.
        '''
        
        self.epochs = epochs
        self.dataset = dataset
        self.load_ckpt = load_ckpt
        self.run_name = run_name
        self.gloabal_step = 0

        if self.dataset not in ["owt", "sni"]:
            raise ValueError("Unsupported dataset")

        self.parse_arguments(config)
        self.initialize_trainer()

        if self.load_ckpt:
            self.load_from_checkpoint(self.load_ckpt)

    def parse_arguments(self, config: Dict):
        '''Loads all arguments from config'''
        # model's config
        self.model_config = NanoConfig(**config["model"])

        # training args
        self.grad_clip = config["data"]["grad_clip"]
        self.checkpoint_dir = config["data"]["checkpoint_dir"]
        self.compile_model = config["data"]["compile_model"]
        self.precision = config["data"]["precision"]
        self.n_workers = config["data"]["n_workers"]
        self.pin_memory = config["data"]["pin_memory"]
        self.backend = config["data"]["backend"]

        # dataloader args
        if self.dataset == "owt":
            self.data_dir = config["data"]["owt"]["data_dir"]
            self.random_sample = config["data"]["owt"]["random_sample"]
            self.log_every = config["data"]["owt"]["log_every"]
            self.val_every = config["data"]["owt"]["val_every"]
            self.save_every = config["data"]["owt"]["save_every"]
        elif self.dataset == "sni":
            self.data_dir = config["data"]["sni"]["data_dir"]
            self.num_examples = config["data"]["sni"]["num_examples"]
            self.val_ratio = config["data"]["sni"]["val_ratio"]
            self.log_every = config["data"]["sni"]["log_every"]
            self.val_every = config["data"]["sni"]["val_every"]
            self.save_every = config["data"]["owt"]["save_every"]

        # adamw parameters
        self.lr = config["optimization"]["lr"]
        self.weight_decay = config["optimization"]["weight_decay"]
        self.betas = config["optimization"]["betas"]

    def initialize_trainer(self):
        '''Initialize trainer'''

        # DDP setup
        self.ddp = int(os.environ.get("RANK", -1)) != -1
        if self.ddp:
            init_process_group(backend=self.backend)
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

        # dataloaders setup
        if self.dataset == "owt":
            self.train_loader, self.val_loader = get_owt_dataloaders(
                data_dir=self.data_dir,
                max_seq_len=self.model_config.max_seq_len,
                batch_size=self.model_config.max_batch_size,
                num_workers=self.n_workers,
                pin_memory=self.pin_memory,
                distributed=self.ddp,
                random_sample=self.random_sample
            )
        elif self.dataset == "sni":
            self.train_loader, self.val_loader = get_sni_dataloaders(
                data_dir=self.data_dir,
                max_seq_len=self.model_config.max_seq_len,
                batch_size=self.model_config.max_batch_size,
                num_examples=self.num_examples,
                val_ratio=self.val_ratio,
                num_workers=self.n_workers,
                pin_memory=self.pin_memory,
                distributed=self.ddp,
            )

        # NanoLM setup
        self.model = Nano(self.model_config).to(self.device)

        if self.compile_model:
            self.model = torch.compile(self.model)

        if self.ddp:
            self.model = DDP(
                self.model, 
                device_ids=[self.local_rank], 
                output_device=self.local_rank, 
                find_unused_parameters=False
            )

        # AdamW setup
        if self.world_size > 1:
            self.optimizer = self.model.module.configure_optimizer(
                self.weight_decay, self.lr, self.betas
            )
        else:
            self.optimizer = self.model.configure_optimizer(
                self.weight_decay, self.lr, self.betas
            )
        
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.epochs
        )

        torch.set_float32_matmul_precision(self.precision)

        if self.master_process:
            self.run_id = time.strftime("%m%d_%H%M%S")
            self.setup_wandb()

    def setup_wandb(self) -> None:
        '''Set up wandb logging.'''
        wandb.init(
            project="nano-lm", 
            name=f"run_{self.run_id}_{self.run_name}", 
            config={
                "model_config": vars(self.model_config),
                "dataset": self.dataset,
            }
        ) 

    def train(self):
        '''Train loop for the model.'''

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
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()
                self.scheduler.step()

                # wait for all GPUs to finish
                if self.device.type == "cuda":
                    torch.cuda.synchronize() 

                self.global_step += 1

                # wandb logging
                if self.master_process and self.global_step % self.log_every == 0:
                    current_lr = self.optimizer.param_groups[0]['lr']
                    wandb.log({"train_loss": loss.item(), "lr": current_lr, "step": self.global_step})

                # validation
                if self.master_process and self.global_step % self.val_every == 0:
                    val_loss = self.validate()
                    wandb.log({"val_loss": val_loss, "step": self.global_step})

                # checkpointing
                if self.master_process and self.global_step % self.save_every == 0:
                    self.save_to_checkpoint(self.global_step)
        
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
    
    def save_to_checkpoint(self, step: int):
        target_model = self.model.module if self.ddp else self.model
        model_data = target_model.state_dict()
        optimizer_data = self.optimizer.state_dict()
        meta_data = {"global_step": self.global_step}
        save_checkpoint(self.checkpoint_dir, step, model_data, optimizer_data, meta_data, rank=self.rank if self.ddp else 0)

    def load_from_checkpoint(self, ckpt_path: str):
        model_data, optimizer_data, meta_data = load_checkpoint(ckpt_path, step=None, device=self.device, load_optimizer=True, rank=self.rank if self.ddp else 0)
        target_model = self.model.module if self.ddp else self.model
        target_model.load_state_dict(model_data)
        self.optimizer.load_state_dict(optimizer_data)
        self.global_step = meta_data.get("global_step", 0)
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="configs/nano_268m.yaml")
    parser.add_argument("--run_name", type=str, default="default")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--load_ckpt", type=str)

    args = parser.parse_args()

    with open(args.config_path, "r") as f:
        config = yaml.safe_load(f)

    trainer = Trainer(
        config, 
        args.epochs, 
        args.dataset, 
        args.load_ckpt, 
        args.run_name
    )
    trainer.train()


if __name__ == "__main__":
    main()
