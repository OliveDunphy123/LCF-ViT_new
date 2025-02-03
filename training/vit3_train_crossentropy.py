import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
import argparse
from tqdm import tqdm
from datetime import datetime
from pathlib import Path
import random
import numpy as np
import matplotlib.pyplot as plt

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.vit_model3_yearly_15 import create_model
from data.my_dataset import create_training_dataloaders

class SoftCrossEntropyLoss(nn.Module):
    """
    Cross entropy loss for soft targets (probability distributions)
    Assumes predictions and targets are probability distributions that sum to 1
    """
    def __init__(self, temperature=1.0):
        super().__init__()
        self.temperature = temperature

    def forward(self, predictions, targets):
        # Apply softmax to predictions with temperature
        pred_probs = F.softmax(predictions / self.temperature, dim=1)
        
        # Compute cross entropy with soft targets
        loss = -torch.sum(targets * torch.log(pred_probs + 1e-8), dim=1)
        return loss.mean()
    
class ViTTrainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader=None,
        learning_rate=1e-4,
        weight_decay=1e-4,  # L2 regularization through weight decay
        device='cuda',
        num_epochs=50,
        optimizer_class=None,
        scheduler_type='onecycle',
        criterion=None
    ):
        # Initialize core components
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.num_epochs = num_epochs
        
        # Create timestamp and directory
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir = Path(f"vit_test_results_{self.timestamp}") # put into model.pth
        self.results_dir.mkdir(exist_ok=True)

        # Initialize TensorBoard writer
        self.writer = SummaryWriter(self.results_dir / 'tensorboard')
        
        # Log hyperparameters
        self.writer.add_hparams(
            {
                'learning_rate': learning_rate,
                'weight_decay': weight_decay,
                'batch_size': train_loader.batch_size,
                'optimizer': optimizer_class.__name__ if optimizer_class else 'AdamW',
                'scheduler': scheduler_type,
            },
            {'dummy': 0}
        )

        # Set criterion (default to SoftCrossEntropyLoss if none provided)
        self.criterion = criterion if criterion else SoftCrossEntropyLoss()

        # Optimizer setup (using only AdamW for proper L2 regularization)
        param_groups = [
            {'params': model.patch_embed.parameters(), 'lr': learning_rate * 0.1},
            {'params': model.blocks.parameters(), 'lr': learning_rate},
            {'params': model.year_embedding.parameters(), 'lr': learning_rate * 5},
            {'params': model.year_proj.parameters(), 'lr': learning_rate * 5},
            {'params': model.regression_head.parameters(), 'lr': learning_rate * 10},
        ]

        # Apply weight decay to all parameter groups
        for group in param_groups:
            group['weight_decay'] = weight_decay

        self.optimizer = optim.AdamW(param_groups)

        # Learning rate scheduler
        total_steps = len(train_loader) * num_epochs
        max_lrs = [group['lr'] for group in param_groups]
        
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
        else:  # cosine
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=total_steps
            )

        # Initialize history for tracking metrics
        self.train_history = {'loss': [], 'accuracy': []}
        self.val_history = {'loss': [], 'accuracy': []}

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        total_correct = 0
        total_pixels = 0
        losses_per_batch = []

        progress_bar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
        for batch_idx, batch in enumerate(progress_bar):
            sentinel_data = batch['sentinel'].to(self.device)
            ground_truth = batch['ground_truth'].to(self.device) #put data to device
            
            self.optimizer.zero_grad() #Gradient zeroing
            predictions = self.model(sentinel_data) #forward propogation

            # Apply softmax to predictions if using cross entropy
            if isinstance(self.criterion, SoftCrossEntropyLoss):
                predictions = F.softmax(predictions, dim=1)

            loss = self.criterion(predictions, ground_truth) # claculate loss
            losses_per_batch.append(loss.item()) #get the loss from this batch???
            
            loss.backward() #backpropogation
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step() # update weights?
            self.scheduler.step()

            # Calculate accuracy (considering predictions as probability distributions)
            with torch.no_grad():
                correct = torch.abs(predictions - ground_truth) < 0.1
                total_correct += correct.sum().item()
                total_pixels += correct.numel()

            total_loss += loss.item()

            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{total_loss/(batch_idx+1):.4f}',
                'lr': f'{self.scheduler.get_last_lr()[0]:.6f}'
            })

            # Log batch metrics
            global_step = epoch * len(self.train_loader) + batch_idx
            self.writer.add_scalar('Batch/loss', loss.item(), global_step)
            self.writer.add_scalar('Batch/learning_rate', self.scheduler.get_last_lr()[0], global_step)

            # Log gradients periodically
            if batch_idx % 100 == 0:
                for name, param in self.model.named_parameters():
                    if param.grad is not None:
                        self.writer.add_histogram(f'gradients/{name}', param.grad, global_step)
                        grad_norm = torch.norm(param.grad)
                        self.writer.add_scalar(f'gradient_norms/{name}', grad_norm, global_step)

        avg_loss = total_loss / len(self.train_loader) # total_loss divide size of training dataset, epoch_loss?
        avg_accuracy = total_correct / total_pixels

        # Update history
        self.train_history['loss'].append(avg_loss)
        self.train_history['accuracy'].append(avg_accuracy)

        # Log epoch metrics
        self.writer.add_scalar('Epoch/train_loss', avg_loss, epoch)
        self.writer.add_scalar('Epoch/train_accuracy', avg_accuracy, epoch)

        # Log loss distribution
        self.writer.add_histogram('Epoch/loss_distribution', 
                                torch.tensor(losses_per_batch), epoch)

        return avg_loss, avg_accuracy

    def validate(self, epoch):
        if self.val_loader is None:
            return None, None

        self.model.eval() # model in evalation 
        total_loss = 0 #test loss start with 0
        total_correct = 0 # correct sample number is 0
        total_pixels = 0 #total sample number is 0
        val_predictions = []
        val_targets = []

        with torch.no_grad(): # no need to do gradient zeroing in eval 
            for batch in self.val_loader:
                sentinel_data = batch['sentinel'].to(self.device)
                ground_truth = batch['ground_truth'].to(self.device)
                
                predictions = self.model(sentinel_data)
                
                # Apply softmax if using cross entropy
                if isinstance(self.criterion, SoftCrossEntropyLoss):
                    predictions = F.softmax(predictions, dim=1)

                loss = self.criterion(predictions, ground_truth)
                
                # Calculate accuracy
                correct = torch.abs(predictions - ground_truth) < 0.1 ##??? the correct sample acculate 
                total_correct += correct.sum().item()
                total_pixels += correct.numel() # total number of samples
                
                total_loss += loss.item()
                
                # Store predictions for visualization
                if len(val_predictions) < 5:
                    val_predictions.append(predictions[:4].cpu())
                    val_targets.append(ground_truth[:4].cpu())

        avg_loss = total_loss / len(self.val_loader)
        avg_accuracy = total_correct / total_pixels

        # Update history
        self.val_history['loss'].append(avg_loss)
        self.val_history['accuracy'].append(avg_accuracy)

        # Log validation metrics
        self.writer.add_scalar('Epoch/val_loss', avg_loss, epoch)
        self.writer.add_scalar('Epoch/val_accuracy', avg_accuracy, epoch)

        # Log prediction visualizations
        if val_predictions:
            predictions = torch.cat(val_predictions[:5])
            targets = torch.cat(val_targets[:5])
            fig = self.create_prediction_figure(predictions, targets)
            self.writer.add_figure('Predictions/validation', fig, epoch)

        return avg_loss, avg_accuracy

    def create_prediction_figure(self, predictions, targets):
        """Create a matplotlib figure comparing predictions and targets"""
        
        fig, axes = plt.subplots(4, 2, figsize=(10, 20))
        for i in range(4):
            # Plot prediction
            axes[i, 0].imshow(predictions[i, 0].numpy())
            axes[i, 0].set_title(f'Prediction {i+1}')
            axes[i, 0].axis('off')
            
            # Plot target
            axes[i, 1].imshow(targets[i, 0].numpy())
            axes[i, 1].set_title(f'Target {i+1}')
            axes[i, 1].axis('off')
        
        plt.tight_layout()
        return fig

    def save_checkpoint(self, epoch, train_loss, val_loss=None, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_history': self.train_history,
            'val_history': self.val_history
        }
        
        checkpoint_path = self.results_dir / f'checkpoint_epoch_{epoch}.pth'
        torch.save(checkpoint, checkpoint_path)

        if is_best:
            best_model_path = self.results_dir / 'best_model.pth'
            torch.save(checkpoint, best_model_path)

    def train(self):
        """Main training loop"""
        print(f"\nStarting training for {self.num_epochs} epochs...")
        
        best_val_loss = float('inf')
        
        # Create custom figure for learning curves
        self.writer.add_custom_scalars({
            'Learning Curves': {
                'loss': ['Multiline', ['Epoch/train_loss', 'Epoch/val_loss']],
                'accuracy': ['Multiline', ['Epoch/train_accuracy', 'Epoch/val_accuracy']]
            }
        })

        for epoch in range(1, self.num_epochs + 1):
            print(f"\nEpoch {epoch}/{self.num_epochs}")
            
            train_loss, train_acc = self.train_epoch(epoch)
            print(f"Train - Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")

            if self.val_loader:
                val_loss, val_acc = self.validate(epoch)
                print(f"Val - Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")

                # Save checkpoint if validation improves
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.save_checkpoint(epoch, train_loss, val_loss, is_best=True) # do we save the accuracy in checkpoint?
                    print(f"New best validation loss: {best_val_loss:.4f}")
                    
            
            # Regular checkpoint every 5 epochs
            if epoch % 5 == 0:
                self.save_checkpoint(epoch, train_loss, val_loss if self.val_loader else None)

        self.writer.close()
        return self.train_history, self.val_history

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
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_epochs', type=int, default=50)
    args = parser.parse_args()

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model and dataloaders
    model = create_model()
    _, yearly_loader = create_training_dataloaders(
        base_path="/mnt/guanabana/raid/shared/dropbox/QinLennart",
        batch_size=args.batch_size
    )
    
    # Split data
    train_indices, val_indices = split_by_location(yearly_loader.dataset)
    
    train_dataset = Subset(yearly_loader.dataset, train_indices)
    val_dataset = Subset(yearly_loader.dataset, val_indices)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Create trainer with soft cross entropy loss
    trainer = ViTTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        device=device,
        num_epochs=args.num_epochs,
        criterion=SoftCrossEntropyLoss()
    )
    
    # Train model
    trainer.train()

if __name__ == "__main__":
    # num_epochs = 50
    # learning_rate = 5e-4
    # weight_decay = 1e-3
    # batch_size = 32
    main()