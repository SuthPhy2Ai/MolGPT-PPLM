"""
Simple training loop; Boilerplate that could apply to any arbitrary neural network,
so nothing in this file really has anything to do with GPT specifically.
"""

import math
import logging

from tqdm import tqdm
import numpy as np

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data.dataloader import DataLoader
from accelerate import Accelerator

from utils import check_novelty, sample, sample_with_uncertainty, canonic_smiles, get_mol
import re
import pandas as pd
from rdkit import Chem


logger = logging.getLogger(__name__)

class TrainerConfig:
    # optimization parameters
    max_epochs = 10
    batch_size = 64
    learning_rate = 3e-4
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    weight_decay = 0.1 # only applied on matmul weights
    # learning rate decay params: linear warmup followed by cosine decay to 10% of original
    lr_decay = False
    warmup_tokens = 375e6 # these two numbers come from the GPT-3 paper, but may not be good defaults elsewhere
    final_tokens = 260e9 # (at what point we reach 10% of original LR)
    # checkpoint settings
    ckpt_path = None
    num_workers = 0 # for DataLoader

    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)

class Trainer:

    def __init__(self, model, train_dataset, test_dataset, config, stoi, itos):
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.config = config
        self.stoi = stoi
        self.itos = itos

        # Initialize Accelerator for multi-GPU support
        self.accelerator = Accelerator(
            mixed_precision='fp16' if torch.cuda.is_available() else 'no',
            gradient_accumulation_steps=1
        )

        # Check if model uses EDL
        raw_model = model.module if hasattr(model, "module") else model
        self.use_edl = getattr(raw_model, 'use_edl', False)

        # Device is handled by accelerator
        self.device = self.accelerator.device

    def save_checkpoint(self):
        # Unwrap model from accelerator wrapper
        raw_model = self.accelerator.unwrap_model(self.model)
        logger.info("saving %s", self.config.ckpt_path)
        # Use accelerator.save to handle distributed saving
        self.accelerator.save(raw_model.state_dict(), self.config.ckpt_path)

    def train(self, wandb):
        model, config = self.model, self.config
        raw_model = self.accelerator.unwrap_model(model)
        optimizer = raw_model.configure_optimizers(config)

        # Prepare model and optimizer with accelerator
        model, optimizer = self.accelerator.prepare(model, optimizer)
        self.model = model  # Update reference

        # [' ', '#', '(', ')', '-', '1', '2', '3', '4', '5', '6', '<', '=', 'B', 'C', 'F', 'H', 'N', 'O', 'S', '[', ']', 'c', 'l', 'n', 'o', 'r', 's']
        # ['#', '(', ')', '-', '1', '2', '3', '4', '5', '6', '<', '=', 'Br', 'C', 'Cl', 'F', 'N', 'O', 'S', '[H]', '[nH]', 'c', 'n', 'o', 's']


        def run_epoch(split, epoch_num=0):
            is_train = split == 'train'
            model.train(is_train)
            data = self.train_dataset if is_train else self.test_dataset
            loader = DataLoader(data, shuffle=True, pin_memory=True,
                                batch_size=config.batch_size,
                                num_workers=config.num_workers)

            # Prepare dataloader with accelerator
            loader = self.accelerator.prepare(loader)

            # Set current epoch for EDL annealing
            if self.use_edl:
                raw_model.current_epoch = epoch_num

            losses = []
            main_losses = []
            kl_losses = []
            pbar = tqdm(enumerate(loader), total=len(loader)) if is_train else enumerate(loader)
            for it, (x, y, p, scaffold) in pbar:
                # Data is automatically placed on correct device by accelerator
                # No need for manual .to(device) calls 

                # forward the model (accelerator handles mixed precision)
                with torch.set_grad_enabled(is_train):
                    logits, loss, _ = model(x, y, p, scaffold)

                    # Handle EDL loss dict or scalar loss
                    if isinstance(loss, dict):
                        loss_dict = loss
                        loss = loss_dict['total']
                        # mse for mse_loss, ce for log_loss
                        main_loss = loss_dict.get('mse', loss_dict.get('ce')).item()
                        main_loss_name = 'mse' if 'mse' in loss_dict else 'ce'
                        kl_loss = loss_dict['kl'].item()
                        kl_weighted = loss_dict['kl_weighted'].item()
                        annealing_coef = loss_dict['annealing_coef']
                    else:
                        loss_dict = None
                        main_loss = None
                        main_loss_name = None
                        kl_loss = None
                        kl_weighted = None
                        annealing_coef = None

                    loss = loss.mean() # collapse all losses if they are scattered on multiple gpus
                    losses.append(loss.item())
                    if main_loss is not None:
                        main_losses.append(main_loss)
                        kl_losses.append(kl_weighted)

                if is_train:
                    # backprop and update the parameters
                    model.zero_grad()
                    self.accelerator.backward(loss)  # accelerator handles backward pass
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                    optimizer.step()

                    # decay the learning rate based on our progress
                    if config.lr_decay:
                        self.tokens += (y >= 0).sum() # number of tokens processed this step (i.e. label is not -100)
                        if self.tokens < config.warmup_tokens:
                            # linear warmup
                            lr_mult = float(self.tokens) / float(max(1, config.warmup_tokens))
                        else:
                            # cosine learning rate decay
                            progress = float(self.tokens - config.warmup_tokens) / float(max(1, config.final_tokens - config.warmup_tokens))
                            lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
                        lr = config.learning_rate * lr_mult
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr
                    else:
                        lr = config.learning_rate

                    # report progress
                    if loss_dict is not None:
                        wandb.log({
                            'step_train_loss': loss.item(),
                            f'step_{main_loss_name}_loss': main_loss,
                            'step_kl_loss': kl_loss,
                            'step_kl_weighted': kl_weighted,
                            'annealing_coef': annealing_coef,
                            'train_step': it + epoch*len(loader),
                            'learning_rate': lr
                        })
                        pbar.set_description(f"epoch {epoch+1} iter {it}: loss {loss.item():.4f} ({main_loss_name} {main_loss:.4f} + kl {kl_weighted:.4f}). lr {lr:e}")
                    else:
                        wandb.log({'step_train_loss': loss, 'train_step': it + epoch*len(loader), 'learning_rate': lr})
                        pbar.set_description(f"epoch {epoch+1} iter {it}: train loss {loss.item():.5f}. lr {lr:e}")
    
            if is_train:
                avg_loss = float(np.mean(losses))
                avg_main = float(np.mean(main_losses)) if main_losses else None
                avg_kl = float(np.mean(kl_losses)) if kl_losses else None
                return avg_loss, avg_main, avg_kl

            if not is_train:
                test_loss = float(np.mean(losses))
                logger.info("test loss: %f", test_loss)
                return test_loss

        best_loss = float('inf')
        self.tokens = 0 # counter used for learning rate decay
        molecules = []

        # CSV logging for loss components
        loss_history = []

        for epoch in range(config.max_epochs):

            train_result = run_epoch('train', epoch_num=epoch)
            train_loss, train_main, train_kl = train_result

            if self.test_dataset is not None:
                test_loss = run_epoch('test', epoch_num=epoch)
            else:
                test_loss = train_loss

            # Save loss components to history
            loss_history.append({
                'epoch': epoch + 1,
                'total_loss': train_loss,
                'main_loss': train_main,
                'kl_loss': train_kl,
                'test_loss': test_loss
            })

            # Save to CSV every 10 epochs
            if (epoch + 1) % 10 == 0 or epoch == config.max_epochs - 1:
                loss_df = pd.DataFrame(loss_history)
                loss_df.to_csv('../logs/loss_history.csv', index=False)

            wandb.log({'epoch_valid_loss': test_loss, 'epoch_train_loss': train_loss, 'epoch': epoch + 1})

            # supports early stopping based on the test loss, or just save always if no test set is provided
            good_model = self.test_dataset is None or test_loss < best_loss
            if self.config.ckpt_path is not None and good_model:
                best_loss = test_loss
                print(f'Saving at epoch {epoch + 1}')
                self.save_checkpoint()

            if self.config.generate:
                pattern =  r"(\[[^\]]+]|<|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
                regex = re.compile(pattern)
                context = "C"
                for i in range(2):
                    x = torch.tensor([self.stoi[s] for s in regex.findall(context)], dtype=torch.long)[None,...].repeat(512, 1).to(self.device)
                    p = None
                    sca = None

                    # Use EDL-aware sampling if EDL is enabled
                    if self.use_edl:
                        y, seq_uncertainty = sample_with_uncertainty(
                            model, x, self.config.block_size,
                            temperature=0.8, sample=True, top_k=10,
                            prop=p, scaffold=sca, return_uncertainty=True
                        )
                    else:
                        y = sample(model, x, self.config.block_size, temperature=0.8, sample=True, top_k=10, prop = p, scaffold = sca)

                    for gen_mol in y:
                        completion = ''.join([self.itos[int(i)] for i in gen_mol])
                        completion = completion.replace('<', '')
                        mol = get_mol(completion)
                        if mol:
                            smiles = Chem.MolToSmiles(mol)
                            molecules.append((mol, smiles, epoch))

        if self.config.generate:
            df = pd.DataFrame(molecules, columns = ['molecule', 'smiles', 'epoch'])
            return df

        return None
