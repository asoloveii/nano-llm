import os
import time
import json
import argparse
from typing import Dict, Any

import wandb
import torch 
import torch.optim as optim
import torch.nn.functional as F

from src.data import get_dataloader
from src.model import Nano, NanoConfig


DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
DATA_DIR = "./data"
MODEL_CONFIG_PATH = "config/nano_400m.json"


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

        self.epochs = self.train_config["n_epochs"]
        self.start_step = 0

        # ensure checkpoint dir exists
        self.checkpoint_dir = train_config.get("checkpoint_dir", "./checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # load training and validation dataloaders
        self.train_loader, self.val_loader = get_dataloader(self.train_config["dataset_type"],
                                                            self.train_config["dataset_name"],
                                                            DATA_DIR, 
                                                            model_config.max_seq_len, 
                                                            model_config.max_batch_size,
                                                            self.train_config["n_workers"])

        # initialize model and compile
        self.model = Nano(model_config)
        self.model.to(DEVICE)
        self.model = torch.compile(self.model)
        torch.set_float32_matmul_precision(self.train_config["precision"])
        
        # configure optimizer
        self.optimizer = self.model.configure_optimizer(self.adamw_config["weight_decay"],
                                                        self.adamw_config["lr"],
                                                        self.adamw_config["betas"])
        
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, 
                                                              T_max=self.train_config["num_epochs"])
        
        # load checkpoint
        if load_from:
            self.load_checkpoint(load_from)

        # initialize logging for wandb library
        self.run_id = time.strftime("%m%d_%H%M%S")
        self.setup_wandb()

    def setup_wandb(self) -> None:
        '''Set up wandb logging.'''
        wandb.init(
            project="nano-llm", 
            name=f"run_{self.run_id}_{self.train_config['run_name']}", 
            config={
                "model_config": self.model_config,
                "adamw_params": self.adamw_config,
                "train_config": self.train_config,
                "dataset": self.train_config['dataset_name'],
            }
        ) 

    def train(self):
        '''Train loop for the model.'''
        global_step = self.start_step
        val_every = self.train_config.get("val_every", 1000)

        for i in range(self.epochs):
            # set model to train mode
            self.model.train()
            # iterate over batch
            for x, y in self.train_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                self.optimizer.zero_grad()

                # forward with mixed precision
                dtype = torch.bfloat16 if DEVICE.type == 'cuda' else torch.float32
                with torch.autocast(device_type=DEVICE.type, dtype=dtype):
                    logits = self.model(x)
                    loss = F.cross_entropy(logits, y)
                
                # backprop + grad clip + step
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.train_config["grad_clip"])
                self.optimizer.step()
                self.scheduler.step()

                # wait for all GPUs to finish
                if torch.cuda.is_available():
                    torch.cuda.synchronize() 

                global_step += 1

                # wandb logging
                if global_step % self.train_config.get("log_every", 100) == 0:
                    wandb.log({"train_loss": loss.item(), "step": global_step})

                # validation
                if global_step % val_every == 0:
                    val_loss = self.validate()
                    wandb.log({"val_loss": val_loss, "step": global_step})

                    # checkpointing
                    ckpt_path = os.path.join(self.checkpoint_dir, f"checkpoint_step_{global_step}.pt")
                    self.save_checkpoint(global_step, ckpt_path)

    def validate(self):
        '''Run validation on the val_loader and return average loss.'''
        self.model.eval()
        total_loss = 0.0
        count = 0
        with torch.no_grad():
            for x, y in self.val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                logits = self.model(x)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-100)
                total_loss += loss.item() * x.size(0)
                count += x.size(0)
        avg_loss = total_loss / count
        print(f"[Validation] Loss: {avg_loss:.4f}")
        self.model.train()
        return avg_loss

    def save_checkpoint(self, iteration: int):
        '''Saves state of the model, optimizer and lr scheduler on a given iteration.'''
        to_save = {"model": self.model.state_dict(),
                   "optimizer": self.optimizer.state_dict(),
                   "lr_scheduler": self.scheduler.state_dict(),
                   "iteration": iteration}
        torch.save(to_save, self.checkpoint_dir)

    def load_checkpoint(self, src: str):
        '''Load state of the model, optimizer and lr scheduler.'''
        loaded = torch.load(src)
        self.model.load_state_dict(loaded["model"])
        self.optimizer.load_state_dict(loaded["optimizer"])
        self.scheduler.load_state_dict(loaded["lr_scheduler"])
        return loaded["iteration"]
    

def parse_args():
    parser = argparse.ArgumentParser(description="Nano LLM Training Script")

    # Dataset & training
    parser.add_argument("--dataset_name", type=str, default="openwebtext",
                        help="Name of dataset to use")
    parser.add_argument("--dataset_type", type=str, default="pretraining",
                        choices=["pretraining", "instruction"],
                        help="Type of dataset (pretraining or instruction)")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping norm")
    parser.add_argument("--log_every", type=int, default=100, help="Log training metrics every N steps")
    parser.add_argument("--val_every", type=int, default=1000, help="Run validation every N steps")
    parser.add_argument("--n_workers", type=int, default=2, help="Number of dataloader workers")
    parser.add_argument("--precision", type=str, default="high", help="Float32 matmul precision (high, medium, low)")
    parser.add_argument("--run_name", type=str, default="default_run", help="WandB run name")

    # Optimizer
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--betas", type=float, nargs=2, default=(0.9, 0.95), help="AdamW betas")

    # Checkpointing
    parser.add_argument("--load_from", type=str, default=None, help="Path to load checkpoint from")

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # load model configuration from JSON
    with open(MODEL_CONFIG_PATH) as f:
        model_config_dict = json.load(f)
    model_config = NanoConfig(**model_config_dict)

    # override batch_size and max_seq_len from command-line
    model_config.max_batch_size = args.batch_size

    # build optimizer config
    adamw_config = {
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "betas": tuple(args.betas)
    }

    # build training config
    train_config = {
        "dataset_name": args.dataset_name,
        "dataset_type": args.dataset_type,
        "num_epochs": args.num_epochs,
        "grad_clip": args.grad_clip,
        "log_every": args.log_every,
        "val_every": args.val_every,
        "n_workers": args.n_workers,
        "precision": args.precision,
        "run_name": args.run_name
    }

    # Initialize trainer
    trainer = Trainer(
        model_config=model_config,
        adamw_config=adamw_config,
        train_config=train_config,
        load_from=args.load_from
    )

    # Start training
    trainer.train()


if __name__ == "__main__":
    main()
