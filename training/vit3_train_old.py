import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from datetime import datetime
from pathlib import Path
import random
import math
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import r2_score
from torch.cuda.amp import GradScaler, autocast

PROJECT_ROOT = Path('/mnt/guanabana/raid/hdd1/qinxu/Python/LCF-ViT')
sys.path.append(str(PROJECT_ROOT))
from models.vit_model3_yearly_15 import create_model
from data.my_dataset import create_training_dataloaders

class CombinedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()

    # def correlation_loss(self, x, y):
    #     # Compute correlation loss for each fraction separately
    #     total_corr_loss = 0
    #     for i in range(x.size(1)):  # For each fraction
    #         x_frac = x[:, i].view(x.size(0), -1)
    #         y_frac = y[:, i].view(y.size(0), -1)
            
    #         x_centered = x_frac - x_frac.mean(dim=1, keepdim=True)
    #         y_centered = y_frac - y_frac.mean(dim=1, keepdim=True)
            
    #         corr = (x_centered * y_centered).sum(dim=1) / (
    #             torch.sqrt((x_centered ** 2).sum(dim=1)) * 
    #             torch.sqrt((y_centered ** 2).sum(dim=1)) + 1e-8
    #         )
    #         total_corr_loss += (1 - corr).mean()
            
    #     return total_corr_loss / x.size(1)

    def forward(self, pred, target):

        # Check for NaN values
        if torch.isnan(pred).any():
            # Instead of raising error, return high loss
            return torch.tensor(10.0, device=pred.device, requires_grad=True)
        
        # Basic losses with error checking
        mse_loss = self.mse(pred.clamp(0, 1), target)
        l1_loss = self.l1(pred.clamp(0, 1), target)

        # corr_loss = self.correlation_loss(pred, target)
        # Per-fraction normalization and loss
        # fraction_losses = 0
        # for i in range(pred.shape[1]):  # For each fraction
        #     pred_i = pred[:, i]
        #     target_i = target[:, i]
            
        #     # Normalize predictions and targets
        
        #     pred_i = (pred_i - pred_i.mean()) / (pred_i.std() + 1e-8)
        #     target_i = (target_i - target_i.mean()) / (target_i.std() + 1e-8)
            
        #     fraction_losses += self.mse(pred_i, target_i)

        # temporal_loss = 0
        # if pred.dim()>3:
        #     temp_diff = pred[:,:,1:] - pred[:, :, :-1]
        #     temporal_loss = torch.mean(torch.abs(temp_diff))
        
        return mse_loss + 0.5 * l1_loss
    
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        return self.early_stop
    
class ViTTrainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader=None,
        learning_rate=1e-6,
        weight_decay=1e-6,
        device='cuda',
        num_epochs=100
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

        # Initialize gradient scaler for mixed precision training
        self.scaler = GradScaler()

        # Initialize loss functions
        self.criterion = CombinedLoss()
        self.early_stopping = EarlyStopping(patience=7)

        # Optimizer with different learning rates for different components
        para_groups=[
                {'params': model.patch_embed.parameters(), 'lr': learning_rate * 0.01},
                {'params': model.blocks.parameters(), 'lr': learning_rate * 0.1},
                {'params': model.year_proj.parameters(), 'lr': learning_rate },      # Added year_proj
                {'params': model.regression_head.parameters(), 'lr': learning_rate * 2},
            ]

        self.optimizer = optim.AdamW(para_groups, weight_decay=weight_decay, eps=1e-8)

       #learning rate scheduler
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.9)

        print(f"Initialized trainer. Training results will be saved to: {self.results_dir}")

    
    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0

        progress_bar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
        for batch_idx, batch in enumerate(progress_bar):
            try:
                sentinel_data = batch['sentinel'].to(self.device)
                ground_truth = batch['ground_truth'].to(self.device)
                
                # Normalize input data per batch
                mean = sentinel_data.mean(dim=(2, 3, 4), keepdim=True)
                std = sentinel_data.std(dim=(2, 3, 4), keepdim=True) + 1e-6
                sentinel_data = (sentinel_data - mean) / std

                self.optimizer.zero_grad()

                predictions = self.model(sentinel_data)
                #predictions = predictions.clamp(0, 1)  # Ensure valid range
                loss = self.criterion(predictions, ground_truth)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.1)
            
                if self.scheduler is not None:
                    self.scheduler.step()

                
                #update metrics 
                total_loss += loss.item()
                
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'lr': f'{self.scheduler.get_last_lr()[0]:.6f}'
                })
            except RuntimeError as e:
                    print(f"Error in batch {batch_idx}: {str(e)}")
                    continue
    
        # Calculate average losses
        avg_loss = total_loss / len(self.train_loader)
        
        # Log to TensorBoard
        self.writer.add_scalar('Loss/train', avg_loss, epoch)
        self.scheduler.step()
        return avg_loss

    def validate(self, epoch):
        """Validation step"""
        if self.val_loader is None:
            return None, None
            
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in self.val_loader:
                sentinel_data = batch['sentinel'].to(self.device)
                ground_truth = batch['ground_truth'].to(self.device)
                
                 # Normalize input data
                mean = sentinel_data.mean(dim=(2, 3, 4), keepdim=True)
                std = sentinel_data.std(dim=(2, 3, 4), keepdim=True) + 1e-8
                sentinel_data = (sentinel_data - mean) / std

                predictions = self.model(sentinel_data)
                predictions = predictions.clamp(0,1)
                try:
                    loss = self.criterion(predictions, ground_truth)
                    total_loss += loss.item()
                    all_preds.append(predictions.cpu())
                    all_targets.append(ground_truth.cpu())
                except RuntimeError as e:
                    print(f"Error in validation: {str(e)}")
                    continue
        if not all_preds:
            return None, None
        # Calculate metrics
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        # Calculate R² scores for each fraction
        r2_scores = []
        for i in range(7):  # For each fraction
            pred_i = all_preds[:, i].flatten().numpy()
            target_i = all_targets[:, i].flatten().numpy()

            # Remove any NaN values
            mask = ~(np.isnan(pred_i) | np.isnan(target_i))
            pred_i = pred_i[mask]
            target_i = target_i[mask]

            if len(pred_i) > 0:
                r2 = r2_score(target_i, pred_i)
                r2_scores.append(r2)
                self.writer.add_scalar(f'R2_Score/fraction_{i+1}', r2, epoch)
            else:
                r2_scores.append(0.0)
           
        avg_loss = total_loss / len(self.val_loader)
        self.writer.add_scalar('Loss/val', avg_loss, epoch)
        
        return avg_loss, r2_scores
    
    def save_checkpoint(self, epoch, train_loss, val_loss=None, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'scaler_state_dict': self.scaler.state_dict()
        }
        
        checkpoint_path = self.results_dir / f'checkpoint_epoch_{epoch}.pth'
        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")

        # Save best model if applicable
        if is_best:
            best_model_path = self.results_dir / 'best_model.pth'
            torch.save(checkpoint, best_model_path)
            print(f"Saved best model with loss: {train_loss:.4f}")

    def train(self):
        """Main training loop"""
        print(f"\nStarting training for {self.num_epochs} epochs...")
        
        best_loss = float('inf')

        for epoch in range(1, self.num_epochs + 1):
            print(f"\nEpoch {epoch}/{self.num_epochs}")
            
            train_loss = self.train_epoch(epoch)
            print(f"Train Loss: {train_loss:.4f}")
            

            # Validation
            if self.val_loader is not None:
                val_loss, r2_scores = self.validate(epoch)
                print(f"Val Loss: {val_loss:.4f}")
                print("R² scores by fraction:")
                for i, r2 in enumerate(r2_scores):
                    print(f"Fraction {i+1}: {r2:.4f}")

                #early stopping check
                if self.early_stopping(val_loss):
                    print("early stopping triggered")
                    break
            
            # Save checkpoint 
            is_best = val_loss < best_loss
            if is_best:
                best_loss = val_loss
            
            if epoch % 5 == 0 or is_best:
                self.save_checkpoint(epoch, train_loss, val_loss, is_best)
        
        print("\nTraining completed!")
        if self.val_loader is not None:
            print(f"best validation loss: {best_loss:.4f}")

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
    
    # train_loader, val_loader = create_training_dataloaders(
    #     base_path="/mnt/guanabana/raid/shared/dropbox/QinLennart",
    #     batch_size=32
    # )
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
        learning_rate=1e-5,
        weight_decay=1e-4,
        device=device,
        num_epochs=100
    )
    
    # Train model
    trainer.train()

if __name__ == "__main__":
    main()