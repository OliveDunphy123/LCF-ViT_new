import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from datetime import datetime
from pathlib import Path
import random
import numpy as np
from torch.utils.tensorboard.writer import SummaryWriter

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.vit_model1_monthly_15 import create_model
from data.my_dataset import create_training_dataloaders

def calculate_accuracy_metrics(predictions, ground_truth):
    mae_per_class = torch.mean(torch.abs(predictions - ground_truth), dim=(0,2,3,4))
    rmse_per_class = torch.sqrt(torch.mean((predictions - ground_truth)**2, dim=(0,2,3,4)))
    tolerance = 0.1
    correct_predictions = torch.abs(predictions - ground_truth) <= tolerance
    overall_accuracy = torch.mean(correct_predictions.float())
    
    r2_scores = []
    for class_idx in range(7):
        y_true = ground_truth[:,class_idx].flatten()
        y_pred = predictions[:,class_idx].flatten()
        ss_tot = torch.sum((y_true - torch.mean(y_true))**2)
        ss_res = torch.sum((y_true - y_pred)**2)
        r2 = 1 - (ss_res / (ss_tot + 1e-8))
        r2_scores.append(r2.item())
    
    return {
        'mae_per_class': mae_per_class,
        'rmse_per_class': rmse_per_class,
        'overall_accuracy': overall_accuracy,
        'r2_scores': r2_scores
    }

def split_by_location(dataset, train_ratio=0.85):
    """Split dataset by unique locations."""
    # Get all unique location IDs
    location_ids = set()
    for item in dataset.unique_ids:
        loc_id = item.split('_')[0]
        location_ids.add(loc_id)
    location_ids = sorted(list(location_ids))
    
    # Randomly split locations
    num_train = int(len(location_ids) * train_ratio)
    train_locations = set(random.sample(location_ids, num_train))
    
    # Create indices for train and validation
    train_indices = []
    val_indices = []
    
    for idx, item in enumerate(dataset.unique_ids):
        loc_id = item.split('_')[0]
        if loc_id in train_locations:
            train_indices.append(idx)
        else:
            val_indices.append(idx)
    
    return train_indices, val_indices

class ViTTrainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader=None,
        learning_rate=1e-4,
        weight_decay=1e-4,
        device='cuda',
        num_epochs=50,
        criterion=None,
        scheduler_type='onecycle'
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.num_epochs = num_epochs
        self.custom_criterion = criterion
        
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir = Path(f"vit_monthly_results_{self.timestamp}")
        self.results_dir.mkdir(exist_ok=True)
        self.writer = SummaryWriter(self.results_dir / 'tensorboard')
        
        # Add TensorBoard custom scalar groups
        self.writer.add_custom_scalars({
            'Loss': {
                'Training': ['Multiline', [
                    'Train/total_loss',
                    'Train/main_loss',
                    'Train/smooth_loss'
                ]],
                'Validation': ['Multiline', [
                    'Val/total_loss',
                    'Val/main_loss',
                    'Val/smooth_loss'
                ]]
            },
            'Accuracy': {
                'Overall': ['Multiline', [
                    'Train/overall_accuracy',
                    'Val/overall_accuracy'
                ]],
                'MAE': ['Multiline', [
                    'Train/mae_avg',
                    'Val/mae_avg'
                ]]
            },
            'Class_Metrics': {
                'MAE': ['Multiline', [f'Train/mae_class_{i}' for i in range(7)]],
                'R2': ['Multiline', [f'Train/r2_class_{i}' for i in range(7)]]
            }
        })
        if criterion is None:
            self.mse_loss = nn.MSELoss()
            self.l1_loss = nn.L1Loss()
        
        para_groups = [
            {'params': model.patch_embed.parameters(), 'lr': learning_rate * 0.2},
            {'params': model.blocks.parameters(), 'lr': learning_rate * 1.5},
            {'params': model.temp_embedding.parameters(), 'lr': learning_rate * 5},
            {'params': model.temp_proj.parameters(), 'lr': learning_rate * 5},
            {'params': model.regression_head.parameters(), 'lr': learning_rate * 10},
        ]
        self.optimizer = optim.AdamW(para_groups, weight_decay=weight_decay)

        total_steps = len(train_loader) * num_epochs
        max_lrs = [group['lr'] for group in para_groups]
        
        if scheduler_type == 'onecycle':
            self.scheduler = optim.lr_scheduler.OneCycleLR(
                self.optimizer, max_lr=max_lrs, total_steps=total_steps,
                pct_start=0.3, div_factor=20, final_div_factor=1e3,
                anneal_strategy='cos'
            )
        else:
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=total_steps, eta_min=1e-6
            )

    def criterion(self, pred, target):
        if self.custom_criterion is not None:
            return self.custom_criterion(pred, target)
        mse = self.mse_loss(pred, target)
        l1 = self.l1_loss(pred, target)
        return 0.7 * mse + 0.3 * l1

    def temporal_smoothness_loss(self, pred):
        temp_diff = pred[:, :, 1:] - pred[:, :, :-1]
        return torch.mean(torch.abs(temp_diff))

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        total_main_loss = 0
        total_smooth_loss = 0
        epoch_predictions = []
        epoch_ground_truth = []

        progress_bar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
        for batch_idx, batch in enumerate(progress_bar):
            sentinel_data = batch['sentinel'].to(self.device)
            ground_truth = batch['ground_truth'].to(self.device)
            
            # Data augmentation
            noise = torch.randn_like(sentinel_data) * 0.01
            sentinel_data = sentinel_data + noise
            
            self.optimizer.zero_grad()
            predictions = self.model(sentinel_data)
            
            main_loss = self.criterion(predictions, ground_truth)
            smooth_loss = self.temporal_smoothness_loss(predictions)
            smooth_weight = min(0.5, 0.1 + epoch * 0.01)
            loss = main_loss + smooth_weight * smooth_loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step()
            
            epoch_predictions.append(predictions.detach())
            epoch_ground_truth.append(ground_truth.detach())
            
            total_loss += loss.item()
            total_main_loss += main_loss.item()
            total_smooth_loss += smooth_loss.item()
            
            global_step = (epoch - 1) * len(self.train_loader) + batch_idx
            self.writer.add_scalar('Batch/loss', loss.item(), global_step)
            self.writer.add_scalar('Batch/main_loss', main_loss.item(), global_step)
            self.writer.add_scalar('Batch/smooth_loss', smooth_loss.item(), global_step)
            self.writer.add_scalar('LR/learning_rate', self.scheduler.get_last_lr()[0], global_step)
            # self.writer.add_scalar('Train/total_loss', avg_loss, epoch)
            # self.writer.add_scalar('Train/main_loss', total_main_loss / len(self.train_loader), epoch)
            # self.writer.add_scalar('Train/smooth_loss', total_smooth_loss / len(self.train_loader), epoch)
            # self.writer.add_scalar('Train/mae_avg', torch.mean(metrics['mae_per_class']), epoch)

            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'main_loss': f'{main_loss.item():.4f}',
                'smooth_loss': f'{smooth_loss.item():.4f}',
                'lr': f'{self.scheduler.get_last_lr()[0]:.6f}'
            })

        epoch_predictions = torch.cat(epoch_predictions, dim=0)
        epoch_ground_truth = torch.cat(epoch_ground_truth, dim=0)
        metrics = calculate_accuracy_metrics(epoch_predictions, epoch_ground_truth)
        
        self.writer.add_scalar('Train/overall_accuracy', metrics['overall_accuracy'], epoch)
        for i, mae in enumerate(metrics['mae_per_class']):
            self.writer.add_scalar(f'Train/mae_class_{i}', mae, epoch)
        for i, r2 in enumerate(metrics['r2_scores']):
            self.writer.add_scalar(f'Train/r2_class_{i}', r2, epoch)
        
        avg_loss = total_loss / len(self.train_loader)
        print(f"\nEpoch {epoch} Training Metrics:")
        print(f"Overall Accuracy: {metrics['overall_accuracy']:.4f}")
        print("MAE per class:", ' '.join(f"{mae:.4f}" for mae in metrics['mae_per_class']))
        print("R² scores:", ' '.join(f"{r2:.4f}" for r2 in metrics['r2_scores']))
        
        return avg_loss, metrics

    def validate(self, epoch):
        if self.val_loader is None or len(self.val_loader) == 0:
            return None, None
        
        self.model.eval()
        total_loss = 0
        epoch_predictions = []
        epoch_ground_truth = []

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc=f'Validation Epoch {epoch}'):
                sentinel_data = batch['sentinel'].to(self.device)
                ground_truth = batch['ground_truth'].to(self.device)
                
                predictions = self.model(sentinel_data)
                loss = self.criterion(predictions, ground_truth)
                
                total_loss += loss.item()
                epoch_predictions.append(predictions.cpu())
                epoch_ground_truth.append(ground_truth.cpu())

        if len(epoch_predictions) > 0:
            epoch_predictions = torch.cat(epoch_predictions, dim=0)
            epoch_ground_truth = torch.cat(epoch_ground_truth, dim=0)
            metrics = calculate_accuracy_metrics(epoch_predictions, epoch_ground_truth)
            
            self.writer.add_scalar('Val/loss', total_loss / len(self.val_loader), epoch)
            self.writer.add_scalar('Val/overall_accuracy', metrics['overall_accuracy'], epoch)
            for i, mae in enumerate(metrics['mae_per_class']):
                self.writer.add_scalar(f'Val/mae_class_{i}', mae, epoch)
            for i, r2 in enumerate(metrics['r2_scores']):
                self.writer.add_scalar(f'Val/r2_class_{i}', r2, epoch)
            
            avg_loss = total_loss / len(self.val_loader)
            print(f"\nValidation Metrics:")
            print(f"Loss: {avg_loss:.4f}")
            print(f"Overall Accuracy: {metrics['overall_accuracy']:.4f}")
            print("MAE per class:", ' '.join(f"{mae:.4f}" for mae in metrics['mae_per_class']))
            print("R² scores:", ' '.join(f"{r2:.4f}" for r2 in metrics['r2_scores']))
            
            return avg_loss, metrics
            
        return None, None

    def train(self):
        print(f"\nStarting training for {self.num_epochs} epochs...")
        best_accuracy = 0.0

        for epoch in range(1, self.num_epochs + 1):
            print(f"\nEpoch {epoch}/{self.num_epochs}")
            
            train_loss, train_metrics = self.train_epoch(epoch)
            
            if self.val_loader is not None:
                val_results = self.validate(epoch)
                if val_results[0] is not None:
                    val_loss, val_metrics = val_results
                    current_accuracy = val_metrics['overall_accuracy']
                else:
                    val_metrics = None
                    current_accuracy = train_metrics['overall_accuracy']
            else:
                val_metrics = None
                current_accuracy = train_metrics['overall_accuracy']
            
            is_best = current_accuracy > best_accuracy
            if is_best:
                best_accuracy = current_accuracy
                
            if epoch % 5 == 0 or epoch == self.num_epochs or is_best:
                self.save_checkpoint(epoch, 
                                  {'train': train_metrics, 'val': val_metrics},
                                  is_best)
        
        print("\nTraining completed!")
        print(f"Best accuracy achieved: {best_accuracy:.4f}")
        self.writer.close()



    def save_checkpoint(self, epoch, metrics, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics
        }
        
        checkpoint_path = self.results_dir / f'checkpoint_epoch_{epoch}.pth'
        torch.save(checkpoint, checkpoint_path)
        
        if is_best:
            best_model_path = self.results_dir / 'best_model.pth'
            torch.save(checkpoint, best_model_path)



def main():
    # Initialize CUDA and clear cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    random.seed(42)
    torch.manual_seed(42)
    np.random.seed(42)
    
    model = create_model()
    monthly_loader, _ = create_training_dataloaders(
        base_path="/mnt/guanabana/raid/shared/dropbox/QinLennart",
        batch_size=32
    )
    
    train_indices, val_indices = split_by_location(monthly_loader.dataset)
    train_dataset = Subset(monthly_loader.dataset, train_indices)
    val_dataset = Subset(monthly_loader.dataset, val_indices)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    trainer = ViTTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        num_epochs=50
    )
    
    trainer.train()

if __name__ == "__main__":
    main()