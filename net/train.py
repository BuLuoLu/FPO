import os
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam, SGD, AdamW
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, ReduceLROnPlateau
from pathlib import Path
from tqdm import tqdm
import time
from datetime import datetime
import numpy as np
import argparse

from model import build_model
from loss import build_loss
from dataset import HairFollicleDataset
from utils import set_seed


class WarmupScheduler:
    def __init__(self, optimizer, warmup_epochs, base_lr, after_scheduler=None):
        self.optimizer = optimizer
        self.warmup_epochs = int(warmup_epochs)
        self.base_lr = float(base_lr)
        self.after_scheduler = after_scheduler
        self.current_epoch = 0
    
    def step(self, epoch=None):
        if epoch is not None:
            self.current_epoch = epoch
        else:
            self.current_epoch += 1
        
        if self.current_epoch < self.warmup_epochs:
            warmup_factor = 0.1 + 0.9 * (self.current_epoch / self.warmup_epochs)
            lr = self.base_lr * warmup_factor
            
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            
            return lr
        else:
            if self.after_scheduler:
                self.after_scheduler.step()
                return self.optimizer.param_groups[0]['lr']
            else:
                return self.base_lr
    
    def state_dict(self):
        state = {
            'current_epoch': self.current_epoch,
            'warmup_epochs': self.warmup_epochs,
            'base_lr': self.base_lr
        }
        if self.after_scheduler:
            state['after_scheduler'] = self.after_scheduler.state_dict()
        return state
    
    def load_state_dict(self, state_dict):
        self.current_epoch = state_dict['current_epoch']
        self.warmup_epochs = state_dict['warmup_epochs']
        self.base_lr = state_dict['base_lr']
        if self.after_scheduler and 'after_scheduler' in state_dict:
            self.after_scheduler.load_state_dict(state_dict['after_scheduler'])


class Trainer:
    def __init__(self, config_path='config.yaml'):
        self.config = self._load_config(config_path)
        set_seed(self.config['misc']['seed'])
        self.device = self._setup_device()
        self._create_output_dirs()
        self.train_loader, self.val_loader = self._build_dataloaders()
        self.model = self._build_model()
        self.criterion = self._build_criterion()
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()
        self.start_epoch = 0
        self.best_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        
        if self.config['misc']['resume']:
            self._resume_training()
        
        print(f"\nTraining Configuration:")
        print(f"  Device: {self.device}")
        print(f"  Train set size: {len(self.train_loader.dataset)}")
        print(f"  Val set size: {len(self.val_loader.dataset)}")
        print(f"  Batch size: {self.config['training']['batch_size']}")
        print(f"  Total epochs: {self.config['training']['num_epochs']}\n")
    
    def _load_config(self, config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    
    def _setup_device(self):
        device_config = self.config['device']
        if device_config['type'] == 'cuda' and torch.cuda.is_available():
            device = torch.device(f"cuda:{device_config['gpu_ids'][0]}")
        else:
            device = torch.device('cpu')
        return device
    
    def _create_output_dirs(self):
        self.checkpoint_dir = Path(self.config['output']['checkpoint_dir'])
        self.log_dir = Path(self.config['output']['log_dir'])
        
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.run_dir = self.log_dir / timestamp
        self.run_dir.mkdir(parents=True, exist_ok=True)
    
    def _build_dataloaders(self):
        data_config = self.config['data']
        
        train_list_path = data_config.get('train_list', None)
        val_list_path = data_config.get('val_list', None)
        
        train_file_list = None
        val_file_list = None
        
        if train_list_path and os.path.exists(train_list_path):
            with open(train_list_path, 'r', encoding='utf-8') as f:
                train_file_list = [line.strip() for line in f if line.strip()]
        
        if val_list_path and os.path.exists(val_list_path):
            with open(val_list_path, 'r', encoding='utf-8') as f:
                val_file_list = [line.strip() for line in f if line.strip()]
        
        train_dataset = HairFollicleDataset(
            image_dir=data_config['image_dir'],
            label_dir=data_config['label_dir'],
            image_size=tuple(data_config['image_size']),
            sigma_major=data_config['sigma_major'],
            sigma_minor=data_config['sigma_minor'],
            train=True,
            use_augmentation=data_config['use_augmentation'],
            heatmap_downsample_factor=data_config['heatmap_downsample_factor'],
            file_list=train_file_list
        )
        
        val_dataset = HairFollicleDataset(
            image_dir=data_config['image_dir'],
            label_dir=data_config['label_dir'],
            image_size=tuple(data_config['image_size']),
            sigma_major=data_config['sigma_major'],
            sigma_minor=data_config['sigma_minor'],
            train=False,
            use_augmentation=False,
            heatmap_downsample_factor=data_config['heatmap_downsample_factor'],
            file_list=val_file_list
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=True,
            num_workers=self.config['dataloader']['num_workers'],
            pin_memory=self.config['dataloader']['pin_memory'],
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=False,
            num_workers=self.config['dataloader']['num_workers'],
            pin_memory=self.config['dataloader']['pin_memory'],
            drop_last=False
        )
        
        return train_loader, val_loader
    
    def _build_model(self):
        model_config = self.config['model']
        model = build_model(
            backbone=model_config['backbone'],
            pretrained=model_config['pretrained'],
            pretrained_path=model_config['pretrained_path'],
            use_deformable=model_config.get('use_deformable', True),
            use_attention=model_config.get('use_attention', True),
            use_elongated=model_config.get('use_elongated', True),
            use_gate=model_config.get('use_gate', True)
        )
        model = model.to(self.device)
        
        if len(self.config['device']['gpu_ids']) > 1:
            model = nn.DataParallel(model, device_ids=self.config['device']['gpu_ids'])
        
        return model
    
    def _build_criterion(self):
        loss_config = self.config['loss']
        criterion = build_loss(loss_config)
        return criterion
    
    def _build_optimizer(self):
        opt_config = self.config['training']['optimizer']
        lr = float(self.config['training']['learning_rate'])
        weight_decay = float(opt_config.get('weight_decay', 1e-4))
        
        opt_type = opt_config['type'].lower()
        
        if opt_type == 'adam':
            betas = opt_config.get('betas', [0.9, 0.999])
            eps = float(opt_config.get('eps', 1e-8))
            optimizer = Adam(
                self.model.parameters(), 
                lr=lr, 
                betas=tuple(betas),
                eps=eps,
                weight_decay=weight_decay
            )
            
        elif opt_type == 'sgd':
            momentum = float(opt_config.get('momentum', 0.9))
            nesterov = opt_config.get('nesterov', False)
            optimizer = SGD(
                self.model.parameters(), 
                lr=lr, 
                momentum=momentum,
                weight_decay=weight_decay,
                nesterov=nesterov
            )
            
        elif opt_type == 'adamw':
            betas = opt_config.get('betas', [0.9, 0.999])
            eps = float(opt_config.get('eps', 1e-8))
            optimizer = AdamW(
                self.model.parameters(), 
                lr=lr, 
                betas=tuple(betas),
                eps=eps,
                weight_decay=weight_decay
            )
            
        else:
            raise ValueError(f"Unsupported optimizer: {opt_config['type']}")
        
        return optimizer
    
    def _build_scheduler(self):
        sched_config = self.config['training']['scheduler']
        
        if not sched_config['use_scheduler']:
            if sched_config.get('use_warmup', False):
                warmup_epochs = int(sched_config.get('warmup_epochs', 20))
                base_lr = float(self.config['training']['learning_rate'])
                return WarmupScheduler(self.optimizer, warmup_epochs, base_lr)
            return None
        
        sched_type = sched_config['type'].lower()
        
        if sched_type == 'step':
            main_scheduler = StepLR(
                self.optimizer,
                step_size=int(sched_config['step_size']),
                gamma=float(sched_config['gamma'])
            )
        elif sched_type == 'cosine':
            main_scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=int(sched_config['T_max']),
                eta_min=float(sched_config['eta_min'])
            )
        elif sched_type == 'plateau':
            main_scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                patience=int(sched_config['patience']),
                factor=float(sched_config['factor'])
            )
        else:
            raise ValueError(f"Unsupported scheduler: {sched_type}")
        
        if sched_config.get('use_warmup', False):
            warmup_epochs = int(sched_config.get('warmup_epochs', 20))
            base_lr = float(self.config['training']['learning_rate'])
            return WarmupScheduler(self.optimizer, warmup_epochs, base_lr, main_scheduler)
        
        return main_scheduler
    
    def _resume_training(self):
        resume_path = self.config['misc']['resume_path']
        if resume_path and os.path.exists(resume_path):
            checkpoint = torch.load(resume_path, map_location=self.device)
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            if self.scheduler and 'scheduler_state_dict' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            self.start_epoch = checkpoint['epoch'] + 1
            self.best_loss = checkpoint.get('best_loss', float('inf'))
            self.train_losses = checkpoint.get('train_losses', [])
            self.val_losses = checkpoint.get('val_losses', [])
            
            print(f"Resumed from epoch {self.start_epoch}, best loss: {self.best_loss:.6f}")
    
    def train_epoch(self, epoch):
        self.model.train()
        epoch_loss = 0.0
        epoch_heatmap_loss = 0.0
        epoch_direction_loss = 0.0
        epoch_unit_loss = 0.0
        epoch_angle_loss = 0.0
        epoch_offset_loss = 0.0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.config['training']['num_epochs']}", disable=True)
        
        for batch_idx, (images, heatmaps, directions, direction_masks, offsets) in enumerate(pbar):
            images = images.to(self.device)
            heatmaps = heatmaps.to(self.device)
            directions = directions.to(self.device)
            direction_masks = direction_masks.to(self.device)
            offsets = offsets.to(self.device)
            
            pred_heatmap, pred_direction, pred_offset = self.model(images)
            
            loss, loss_dict = self.criterion(
                pred_heatmap, pred_direction, pred_offset, 
                heatmaps, directions, offsets, direction_masks
            )

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            epoch_loss += loss.item()
            epoch_heatmap_loss += loss_dict['heatmap_loss']
            epoch_direction_loss += loss_dict['direction_loss']
            epoch_unit_loss += loss_dict['unit_loss']
            epoch_angle_loss += loss_dict['angle_loss']
            epoch_offset_loss += loss_dict['offset_loss']
            
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'hm': f"{loss_dict['heatmap_loss']:.4f}",
                'dir': f"{loss_dict['direction_loss']:.4f}",
                'unit': f"{loss_dict['unit_loss']:.4f}",
                'ang': f"{loss_dict['angle_loss']:.4f}",
                'off': f"{loss_dict['offset_loss']:.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.6f}"
            })
        
        avg_loss = epoch_loss / len(self.train_loader)
        avg_heatmap_loss = epoch_heatmap_loss / len(self.train_loader)
        avg_direction_loss = epoch_direction_loss / len(self.train_loader)
        avg_unit_loss = epoch_unit_loss / len(self.train_loader)
        avg_angle_loss = epoch_angle_loss / len(self.train_loader)
        avg_offset_loss = epoch_offset_loss / len(self.train_loader)
        
        return {
            'loss': avg_loss,
            'heatmap_loss': avg_heatmap_loss,
            'direction_loss': avg_direction_loss,
            'unit_loss': avg_unit_loss,
            'angle_loss': avg_angle_loss,
            'offset_loss': avg_offset_loss
        }
    
    def validate(self, epoch):
        if len(self.val_loader.dataset) == 0:
            return {
                'loss': 0.0,
                'heatmap_loss': 0.0,
                'direction_loss': 0.0,
                'unit_loss': 0.0,
                'angle_loss': 0.0,
                'offset_loss': 0.0
            }
        
        self.model.eval()
        val_loss = 0.0
        val_heatmap_loss = 0.0
        val_direction_loss = 0.0
        val_unit_loss = 0.0
        val_angle_loss = 0.0
        val_offset_loss = 0.0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"Validation", disable=True)
            
            for images, heatmaps, directions, direction_masks, offsets in pbar:
                images = images.to(self.device)
                heatmaps = heatmaps.to(self.device)
                directions = directions.to(self.device)
                direction_masks = direction_masks.to(self.device)
                offsets = offsets.to(self.device)
                
                pred_heatmap, pred_direction, pred_offset = self.model(images)
                
                loss, loss_dict = self.criterion(
                    pred_heatmap, pred_direction, pred_offset, 
                    heatmaps, directions, offsets, direction_masks
                )
                
                val_loss += loss.item()
                val_heatmap_loss += loss_dict['heatmap_loss']
                val_direction_loss += loss_dict['direction_loss']
                val_unit_loss += loss_dict['unit_loss']
                val_angle_loss += loss_dict['angle_loss']
                val_offset_loss += loss_dict['offset_loss']
                
                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'hm': f"{loss_dict['heatmap_loss']:.4f}",
                    'dir': f"{loss_dict['direction_loss']:.4f}",
                    'unit': f"{loss_dict['unit_loss']:.4f}",
                    'ang': f"{loss_dict['angle_loss']:.4f}",
                    'off': f"{loss_dict['offset_loss']:.4f}"
                })
        
        avg_loss = val_loss / len(self.val_loader)
        avg_heatmap_loss = val_heatmap_loss / len(self.val_loader)
        avg_direction_loss = val_direction_loss / len(self.val_loader)
        avg_unit_loss = val_unit_loss / len(self.val_loader)
        avg_angle_loss = val_angle_loss / len(self.val_loader)
        avg_offset_loss = val_offset_loss / len(self.val_loader)
        
        return {
            'loss': avg_loss,
            'heatmap_loss': avg_heatmap_loss,
            'direction_loss': avg_direction_loss,
            'unit_loss': avg_unit_loss,
            'angle_loss': avg_angle_loss,
            'offset_loss': avg_offset_loss
        }
    
    def save_checkpoint(self, epoch, val_loss, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_loss': self.best_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'config': self.config
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        if (epoch + 1) % self.config['output']['save_freq'] == 0:
            checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch+1}.pth"
            torch.save(checkpoint, checkpoint_path)
        
        if is_best and self.config['output']['save_best']:
            best_path = self.checkpoint_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
            print(f"Saved best model (val_loss: {val_loss:.6f})")
        
        latest_path = self.checkpoint_dir / "latest.pth"
        torch.save(checkpoint, latest_path)
    
    def save_logs(self):
        log_path = self.run_dir / "training_log.txt"
        with open(log_path, 'w') as f:
            f.write("Epoch,Train_Loss,Train_Heatmap,Train_Direction,Train_Unit,Train_Angle,Train_Offset,Val_Loss,Val_Heatmap,Val_Direction,Val_Unit,Val_Angle,Val_Offset,LR\n")
            for i in range(len(self.train_losses)):
                train_loss = self.train_losses[i]
                val_loss = self.val_losses[i] if i < len(self.val_losses) else {}
                lr = self.optimizer.param_groups[0]['lr']
                
                f.write(f"{i+1},"
                       f"{train_loss.get('loss', 0):.6f},"
                       f"{train_loss.get('heatmap_loss', 0):.6f},"
                       f"{train_loss.get('direction_loss', 0):.6f},"
                       f"{train_loss.get('unit_loss', 0):.6f},"
                       f"{train_loss.get('angle_loss', 0):.6f},"
                       f"{train_loss.get('offset_loss', 0):.6f},"
                       f"{val_loss.get('loss', 0):.6f},"
                       f"{val_loss.get('heatmap_loss', 0):.6f},"
                       f"{val_loss.get('direction_loss', 0):.6f},"
                       f"{val_loss.get('unit_loss', 0):.6f},"
                       f"{val_loss.get('angle_loss', 0):.6f},"
                       f"{val_loss.get('offset_loss', 0):.6f},"
                       f"{lr:.8f}\n")
    
    def train(self):
        print("\n" + "="*60)
        print("Training Started")
        print("="*60 + "\n")
        
        start_time = time.time()
        
        for epoch in range(self.start_epoch, self.config['training']['num_epochs']):
            epoch_start = time.time()
            
            train_metrics = self.train_epoch(epoch + 1)
            self.train_losses.append(train_metrics)
            
            has_val_set = len(self.val_loader.dataset) > 0
            if has_val_set:
                val_metrics = self.validate(epoch + 1)
                self.val_losses.append(val_metrics)
            else:
                val_metrics = {
                    'loss': train_metrics['loss'],
                    'heatmap_loss': train_metrics['heatmap_loss'],
                    'direction_loss': train_metrics['direction_loss'],
                    'unit_loss': train_metrics['unit_loss'],
                    'angle_loss': train_metrics['angle_loss'],
                    'offset_loss': train_metrics['offset_loss']
                }
                self.val_losses.append(val_metrics)
            
            if self.scheduler:
                if isinstance(self.scheduler, WarmupScheduler):
                    self.scheduler.step(epoch)
                elif isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['loss'])
                else:
                    self.scheduler.step()
            
            epoch_time = time.time() - epoch_start
            print(f"\nEpoch {epoch+1}/{self.config['training']['num_epochs']} - "
                  f"Train Loss: {train_metrics['loss']:.6f}, "
                  f"Val Loss: {val_metrics['loss']:.6f}, "
                  f"LR: {self.optimizer.param_groups[0]['lr']:.8f}, "
                  f"Time: {epoch_time:.2f}s")
            
            current_loss = val_metrics['loss'] if has_val_set else train_metrics['loss']
            is_best = current_loss < self.best_loss
            if is_best:
                self.best_loss = current_loss
            
            self.save_checkpoint(epoch, current_loss, is_best)
            self.save_logs()
        
        total_time = time.time() - start_time
        print("\n" + "="*60)
        print("Training Completed")
        print(f"Total time: {total_time/3600:.2f} hours")
        print(f"Best validation loss: {self.best_loss:.6f}")
        print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Train hair follicle detection model')
    parser.add_argument('--config', type=str, default='Code/config.yaml',
                       help='Path to config file (default: Code/config.yaml)')
    parser.add_argument('--exp_name', type=str, default=None,
                       help='Experiment name (optional)')
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("Hair Follicle Detection Model Training")
    print("="*60)
    print(f"Config file: {args.config}")
    if args.exp_name:
        print(f"Experiment name: {args.exp_name}")
    print("="*60 + "\n")
    
    trainer = Trainer(config_path=args.config)
    trainer.train()
