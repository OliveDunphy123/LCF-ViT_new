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
from data.my_dataset import create_training_dataloaders

class ViTTrainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader=None,
        learning_rate=1e-4,
        weight_decay=1e-4,
        device='cuda',
        num_epochs=50
    ):
    
        # initialize core components
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.num_epochs = num_epochs
        
        # Create timestamp and directory at initialization
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir = Path(f"vit_test_results_{self.timestamp}")
        self.results_dir.mkdir(exist_ok=True)

        # Initialize TensorBoard writer
        self.writer = SummaryWriter(self.results_dir / 'tensorboard')

        #Loss function
        self.mse_loss = nn.MSELoss() ## crossentropy()???
        self.l1_loss = nn.L1Loss()
        self.cross_loss = nn.CrossEntropyLoss()

        # Optimizer with different learning rates for different components
        para_groups=[
                {'params': model.patch_embed.parameters(), 'lr': learning_rate * 0.1},
                {'params': model.blocks.parameters(), 'lr': learning_rate},
                {'params': model.year_embedding.parameters(), 'lr': learning_rate * 5},  # Changed from temporal_embed
                {'params': model.year_proj.parameters(), 'lr': learning_rate * 5},      # Added year_proj
                {'params': model.regression_head.parameters(), 'lr': learning_rate * 10},
            ]

        self.optimizer = optim.AdamW(para_groups, weight_decay=weight_decay)

       #learning rate scheduler
        total_steps = len(train_loader) * num_epochs
        max_lrs = [group['lr'] for group in para_groups]
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=max_lrs,
            total_steps=total_steps,
            pct_start=0.3,  # 10% warmup
            div_factor=20,
            final_div_factor=1e3,
            anneal_strategy='cos'
        )
        print(f"Initialized trainer. Training results will be saved to: {self.results_dir}")

    def criterion(self, pred, target):
        """Combined loss function"""
        mse = self.mse_loss(pred, target)
        l1 = self.l1_loss(pred, target)
        #cross = self.cross_loss(pred,target)
        return mse

    def temporal_smoothness_loss(self, pred):
        """Calculate temporal smoothness loss"""
        temp_diff = pred[:, :, 1:] - pred[:, :, :-1]
        return torch.mean(torch.abs(temp_diff))
    
    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        total_main_loss = 0
        total_smooth_loss = 0

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
            
            #update metrics 
            total_loss += loss.item()
            total_main_loss += main_loss.item()
            total_smooth_loss += smooth_loss.item()
            
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'main_loss': f'{main_loss.item():.4f}',
                'smooth_loss': f'{smooth_loss.item():.4f}',
                'lr': f'{self.scheduler.get_last_lr()[0]:.6f}'
            })
        
        # Calculate average losses
        avg_loss = total_loss / len(self.train_loader)
        avg_main_loss = total_main_loss / len(self.train_loader)
        avg_smooth_loss = total_smooth_loss / len(self.train_loader)
        
        # Log to TensorBoard
        self.writer.add_scalar('Loss/train', avg_loss, epoch)
        self.writer.add_scalar('Loss/main', avg_main_loss, epoch)
        self.writer.add_scalar('Loss/smooth', avg_smooth_loss, epoch)
        
        return avg_loss, avg_main_loss, avg_smooth_loss

    def validate(self, epoch):
        """Validation step"""
        if self.val_loader is None:
            return None, None, None
            
        self.model.eval()
        total_loss = 0
        total_main_loss = 0
        total_smooth_loss = 0

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
        
        avg_loss = total_loss / len(self.val_loader)
        avg_main_loss = total_main_loss / len(self.val_loader)
        avg_smooth_loss = total_smooth_loss / len(self.val_loader)
        
        # Log validation metrics
        self.writer.add_scalar('Loss/val', avg_loss, epoch)
        
        return avg_loss, avg_main_loss, avg_smooth_loss
    
    def save_checkpoint(self, epoch, train_loss, val_loss=None, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss
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

        for epoch in range(1, self.num_epochs + 1):
            print(f"\nEpoch {epoch}/{self.num_epochs}")
            
            train_loss, train_main_loss, train_smooth_loss = self.train_epoch(epoch)
            print(f"Train Loss: {train_loss:.4f} (Main: {train_main_loss:.4f}, Smooth: {train_smooth_loss:.4f})")
            

            # Validation
            if self.val_loader is not None:
                val_loss, val_main_loss, val_smooth_loss = self.validate(epoch)
                print(f"Val Loss: {val_loss:.4f} (Main: {val_main_loss:.4f}, Smooth: {val_smooth_loss:.4f})")
            else:
                val_loss = None
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            # Save checkpoint 
            if epoch % 5 == 0 or epoch == self.num_epochs:
                self.save_checkpoint(epoch, train_loss)
                self.plot_training_curves(train_losses,
                                     val_losses,
                                     self.results_dir/f'learning_curves_epoch_{epoch}.png')

            if train_loss < best_loss:
                best_loss = train_loss
                self.save_checkpoint(epoch, train_loss)
                print(f"New best loss:{best_loss:.4f}")
        
        print("\nTraining completed!")
        print(f"Best loss achieved: {best_loss:.4f}")

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




