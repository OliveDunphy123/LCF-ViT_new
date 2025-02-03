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
from models.vit_model3_yearly_15 import create_model
from data.my_whole_dataset import create_training_dataloaders


def calculate_accuracy_metrics(predictions, ground_truth):
        """
        Calculate accuracy metrics for land cover fraction predictions
        
        Args:
            predictions: tensor of shape [B, 7, T, 5, 5] 
            ground_truth: tensor of shape [B, 7, T, 5, 5]
            
        Returns:
            Dictionary containing various accuracy metrics
        """
        import torch
        
        # Mean Absolute Error for each class
        mae_per_class = torch.mean(torch.abs(predictions - ground_truth), dim=(0,2,3,4))
        
        # Root Mean Square Error for each class
        rmse_per_class = torch.sqrt(torch.mean((predictions - ground_truth)**2, dim=(0,2,3,4)))
        
        # Overall accuracy (considering predictions within 10% of ground truth as correct)
        tolerance = 0.1
        correct_predictions = torch.abs(predictions - ground_truth) <= tolerance
        overall_accuracy = torch.mean(correct_predictions.float())
        
        # R² score for each class
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
        print("Initializing ViTTrainer with parameters:")
        print(f"scheduler_type: {scheduler_type}")
        print(f"criterion: {criterion}")
    
        # initialize core components
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.num_epochs = num_epochs
        self.custom_criterion = criterion
        
        # Create timestamp and directory at initialization
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir = Path(f"vit_test_results_{self.timestamp}")
        self.results_dir.mkdir(exist_ok=True)

        # Initialize TensorBoard writer
        self.writer = SummaryWriter(self.results_dir / 'tensorboard')
        # Log hyperparameters
        self.writer.add_hparams(
            {
                'learning_rate': learning_rate,
                'weight_decay': weight_decay,
                'batch_size': train_loader.batch_size,
                'num_epochs': num_epochs,
                'scheduler_type': scheduler_type
            },
            {'dummy': 0}  # Required placeholder metric
        )
        # Add custom scalar groups
        self.writer.add_custom_scalars({
            'Training': {
                'losses': ['Multiline', [
                    'Loss/train_total',
                    'Loss/train_main',
                    'Loss/train_smooth'
                ]],
            },
            'Validation': {
                'losses': ['Multiline', [
                    'Loss/val_total',
                    'Loss/val_main',
                    'Loss/val_smooth'
                ]],
            },
            'Learning_Rate': {
                'lr': ['Multiline', ['LR/learning_rate']]
            }
        })

        #Loss function
        #self.custom_criterion = criterion
        if criterion is None:
            self.mse_loss = nn.MSELoss()
            self.l1_loss = nn.L1Loss()
            self.cross_loss = nn.CrossEntropyLoss()

        # Optimizer with different learning rates for different components
        para_groups=[
                {'params': model.patch_embed.parameters(), 'lr': learning_rate * 0.2},
                {'params': model.blocks.parameters(), 'lr': learning_rate * 1.5},
                {'params': model.year_embedding.parameters(), 'lr': learning_rate * 5},  # Changed from temporal_embed
                {'params': model.year_proj.parameters(), 'lr': learning_rate * 5},      # Added year_proj
                {'params': model.regression_head.parameters(), 'lr': learning_rate * 10},
            ]

        self.optimizer = optim.AdamW(para_groups, weight_decay=weight_decay)

       #learning rate scheduler
        total_steps = len(train_loader) * num_epochs
        max_lrs = [group['lr'] for group in para_groups]
        if scheduler_type == 'onecycle':
            self.scheduler = optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=max_lrs,
                total_steps=total_steps,
                pct_start=0.3,
                div_factor=20,
                final_div_factor=1e3,
                anneal_strategy='cos'
            )
        elif scheduler_type == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=total_steps,
                eta_min=1e-6
            )
        else:
            raise ValueError(f"Unsupported scheduler type: {scheduler_type}")

        print(f"Initialized trainer. Training results will be saved to: {self.results_dir}")


    # def log_metrics(writer, metrics, epoch, prefix='train'):
    #     """Log metrics to tensorboard"""
    #     writer.add_scalar(f'{prefix}/overall_accuracy', metrics['overall_accuracy'], epoch)
        
    #     for i, mae in enumerate(metrics['mae_per_class']):
    #         writer.add_scalar(f'{prefix}/mae_class_{i}', mae, epoch)
            
    #     for i, rmse in enumerate(metrics['rmse_per_class']):
    #         writer.add_scalar(f'{prefix}/rmse_class_{i}', rmse, epoch)
            
    #     for i, r2 in enumerate(metrics['r2_scores']):
    #         writer.add_scalar(f'{prefix}/r2_class_{i}', r2, epoch)

    def criterion(self, pred, target):
        """Combined loss function"""
        if self.custom_criterion is not None:
            return self.custom_criterion(pred, target)
        
        # Default loss if no custom criterion provided
        mse = self.mse_loss(pred, target)
        l1 = self.l1_loss(pred, target)
        return 0.7 * mse + 0.3 * l1

    def temporal_smoothness_loss(self, pred):
        """Calculate temporal smoothness loss"""
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
            
            # Add noise augmentation
            noise = torch.randn_like(sentinel_data) * 0.01
            sentinel_data = sentinel_data + noise

            self.optimizer.zero_grad()
            predictions = self.model(sentinel_data)

            # Calculate losses
            main_loss = self.criterion(predictions, ground_truth)
            smooth_loss = self.temporal_smoothness_loss(predictions)
            smooth_weight = min(0.5, 0.1+epoch * 0.01)
            loss = main_loss + smooth_weight * smooth_loss
            
            loss.backward()
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            self.scheduler.step()
            
            # Collect predictions and ground truth for epoch-level metrics
            epoch_predictions.append(predictions.detach())
            epoch_ground_truth.append(ground_truth.detach())

            #update metrics 
            total_loss += loss.item()
            total_main_loss += main_loss.item()
            total_smooth_loss += smooth_loss.item()
            
            # Log batch-level metrics
            global_step = (epoch - 1) * len(self.train_loader) + batch_idx
            self.writer.add_scalar('Batch/loss', loss.item(), global_step)
            self.writer.add_scalar('Batch/main_loss', main_loss.item(), global_step)
            self.writer.add_scalar('Batch/smooth_loss', smooth_loss.item(), global_step)
            self.writer.add_scalar('LR/learning_rate', self.scheduler.get_last_lr()[0], global_step)

            # Log gradients periodically
            if batch_idx % 10 == 0:
                for name, param in self.model.named_parameters():
                    if param.grad is not None:
                        self.writer.add_histogram(f'gradients/{name}', param.grad, global_step)

            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'main_loss': f'{main_loss.item():.4f}',
                'smooth_loss': f'{smooth_loss.item():.4f}',
                'lr': f'{self.scheduler.get_last_lr()[0]:.6f}'
            })
        
        # Calculate epoch metrics
        epoch_predictions = torch.cat(epoch_predictions, dim=0)
        epoch_ground_truth = torch.cat(epoch_ground_truth, dim=0)
        metrics = calculate_accuracy_metrics(epoch_predictions, epoch_ground_truth)
        
        # Log epoch metrics
        self.writer.add_scalar('Train/overall_accuracy', metrics['overall_accuracy'], epoch)
        for i, mae in enumerate(metrics['mae_per_class']):
            self.writer.add_scalar(f'Train/mae_class_{i}', mae, epoch)
        for i, r2 in enumerate(metrics['r2_scores']):
            self.writer.add_scalar(f'Train/r2_class_{i}', r2, epoch)
        
        # Calculate average losses
        avg_loss = total_loss / len(self.train_loader)
        avg_main_loss = total_main_loss / len(self.train_loader)
        avg_smooth_loss = total_smooth_loss / len(self.train_loader)
        
        # Log to TensorBoard
        self.writer.add_scalar('Loss/train_total', avg_loss, epoch)
        self.writer.add_scalar('Loss/train_main', avg_main_loss, epoch)
        self.writer.add_scalar('Loss/train_smooth', avg_smooth_loss, epoch)

        print(f"\nEpoch {epoch} Training Metrics:")
        print(f"Overall Accuracy: {metrics['overall_accuracy']:.4f}")
        print("MAE per class:", ' '.join(f"{mae:.4f}" for mae in metrics['mae_per_class']))
        print("R² scores:", ' '.join(f"{r2:.4f}" for r2 in metrics['r2_scores']))
        
        return avg_loss, avg_main_loss, avg_smooth_loss, metrics

    def validate(self, epoch):
        """Validation step"""
        if self.val_loader is None or len(self.val_loader) ==0:
            return None, None
            
        self.model.eval()
        total_loss = 0
        total_main_loss = 0
        total_smooth_loss = 0
        all_predictions = []
        all_ground_truth = []

        with torch.no_grad():
            for batch in self.val_loader:
                sentinel_data = batch['sentinel'].to(self.device)
                ground_truth = batch['ground_truth'].to(self.device)
                
                predictions = self.model(sentinel_data)
                
                main_loss = self.criterion(predictions, ground_truth)
                smooth_loss = self.temporal_smoothness_loss(predictions)
                loss = main_loss + 0.1 * smooth_loss
                
                total_loss += loss.item()
                total_main_loss += main_loss.item()
                total_smooth_loss += smooth_loss.item()

                # Store predictions and ground truth
                all_predictions.append(predictions.cpu())
                all_ground_truth.append(ground_truth.cpu())
        
        # Calculate validation metrics
        epoch_predictions = torch.cat(all_predictions, dim=0)
        epoch_ground_truth = torch.cat(all_ground_truth, dim=0)
        metrics = calculate_accuracy_metrics(epoch_predictions, epoch_ground_truth)
        
        # Log validation metrics
        self.writer.add_scalar('Val/loss', total_loss / len(self.val_loader), epoch)
        self.writer.add_scalar('Val/overall_accuracy', metrics['overall_accuracy'], epoch)
        for i, mae in enumerate(metrics['mae_per_class']):
            self.writer.add_scalar(f'Val/mae_class_{i}', mae, epoch)
        for i, r2 in enumerate(metrics['r2_scores']):
            self.writer.add_scalar(f'Val/r2_class_{i}', r2, epoch)

        avg_loss = total_loss / len(self.val_loader)
        avg_main_loss = total_main_loss / len(self.val_loader)
        avg_smooth_loss = total_smooth_loss / len(self.val_loader)
        
        # Log validation metrics
        self.writer.add_scalar('Loss/val_total', avg_loss, epoch)
        self.writer.add_scalar('Loss/val_main', avg_main_loss, epoch)
        self.writer.add_scalar('Loss/val_smooth', avg_smooth_loss, epoch)

        print(f"\nEpoch {epoch} Validation Metrics:")
        print(f"Loss: {avg_loss:.4f} (Main: {avg_main_loss:.4f}, Smooth: {avg_smooth_loss:.4f})")
        print(f"Overall Accuracy: {metrics['overall_accuracy']:.4f}")
        print("MAE per class:", ' '.join(f"{mae:.4f}" for mae in metrics['mae_per_class']))
        print("R² scores:", ' '.join(f"{r2:.4f}" for r2 in metrics['r2_scores']))
        
        return avg_loss, avg_main_loss, avg_smooth_loss, metrics
    
    def save_checkpoint(self, epoch, train_loss, metrics, val_loss=None, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'metrics': metrics
        }
        
        checkpoint_path = self.results_dir / f'checkpoint_epoch_{epoch}.pth'
        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")

        # Save best model if applicable
        if is_best:
            best_model_path = self.results_dir / 'best_model.pth'
            torch.save(checkpoint, best_model_path)
            print(f"Saved best model with loss: {train_loss:.4f}")

    def plot_training_curves(self, train_losses, val_losses, save_path):
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Losses')
        plt.legend()
        plt.grid(True)
        plt.savefig(save_path)
        plt.close()

    def train(self):
        """Main training loop"""
        print(f"\nStarting training for {self.num_epochs} epochs...")
        
        train_losses = []
        val_losses = []
        best_loss = float('inf')
        best_accuracy = 0.0

        for epoch in range(1, self.num_epochs + 1):
            print(f"\nEpoch {epoch}/{self.num_epochs}")
            
            train_loss, train_main_loss, train_smooth_loss, train_metrics = self.train_epoch(epoch)
            print(f"Train Loss: {train_loss:.4f} (Main: {train_main_loss:.4f}, Smooth: {train_smooth_loss:.4f})")
            

            # Validation
            if self.val_loader is not None:
                val_loss, val_main_loss, val_smooth_loss, val_metrics = self.validate(epoch)
                current_accuracy = val_metrics['overall_accuracy']
                print(f"Val Loss: {val_loss:.4f} (Main: {val_main_loss:.4f}, Smooth: {val_smooth_loss:.4f})")
            else:
                val_loss = None
                current_accuracy = train_metrics['overall_accuracy']
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
        #     # Save checkpoint 
        #     if epoch % 5 == 0 or epoch == self.num_epochs:
        #         self.save_checkpoint(epoch, train_loss)
        #         self.plot_training_curves(train_losses,
        #                              val_losses,
        #                              self.results_dir/f'learning_curves_epoch_{epoch}.png')

        #     if train_loss < best_loss:
        #         best_loss = train_loss
        #         self.save_checkpoint(epoch, train_loss)
        #         print(f"New best loss:{best_loss:.4f}")
        
        # print("\nTraining completed!")
        # print(f"Best loss achieved: {best_loss:.4f}")

        # Save checkpoints
            is_best = current_accuracy > best_accuracy
            if is_best:
                best_accuracy = current_accuracy
                
            if epoch % 5 == 0 or epoch == self.num_epochs or is_best:
                self.save_checkpoint(epoch, 
                                  {'train': train_metrics, 'val': val_metrics if self.val_loader else None},
                                  is_best)
        
        print("\nTraining completed!")
        print(f"Best accuracy achieved: {best_accuracy:.4f}")
        self.writer.close()

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


def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Set random seed for reproducibility
    random.seed(42)
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create model
    model = create_model()
    print("Model created successfully")
    
    # Create dataloaders
    _, yearly_loader = create_training_dataloaders(
        base_path="/mnt/guanabana/raid/shared/dropbox/QinLennart",
        batch_size=32
    )
    
    # Split data by locations
    train_indices, val_indices = split_by_location(yearly_loader.dataset, train_ratio=0.85)
    print(f"Number of training locations: {len(set(item.split('_')[0] for item in [yearly_loader.dataset.unique_ids[i] for i in train_indices]))}")
    print(f"Number of validation locations: {len(set(item.split('_')[0] for item in [yearly_loader.dataset.unique_ids[i] for i in val_indices]))}")
    
    # Create train and validation datasets
    train_dataset = Subset(yearly_loader.dataset, train_indices)
    val_dataset = Subset(yearly_loader.dataset, val_indices)  # noqa: F841
    
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
    
    # Create trainer
    trainer = ViTTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        num_epochs=50
    )
    
    # Train model
    trainer.train()

if __name__ == "__main__":
    main()




