import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from my_dataset import create_training_dataloaders
import numpy as np
from tqdm import tqdm
import gc

class SimpleMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleMLP, self).__init__()
        self.flatten = nn.Flatten()
        
        # Dimension reduction first (smaller network for yearly data)
        self.reduce_dim = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
        )
        
        # Main network after dimension reduction
        self.main_network = nn.Sequential(
            nn.Linear(256, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.Sigmoid()  # Ensure output is between 0 and 1
        )
        
        # Initialize weights carefully
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu', a=0.1)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        x = self.flatten(x)
        x = (x - x.mean(dim=1, keepdim=True)) / (x.std(dim=1, keepdim=True) + 1e-8)
        x = self.reduce_dim(x)
        x = self.main_network(x)
        return x

def check_gradients(model):
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad = param.grad
            print(f"\n{name} gradients:")
            print(f"Mean: {grad.mean().item():.6f}")
            print(f"Std: {grad.std().item():.6f}")
            print(f"Max: {grad.max().item():.6f}")
            print(f"Min: {grad.min().item():.6f}")

def train_model(model, train_loader, criterion, optimizer, device, num_epochs=5):
    model.train()
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        valid_batches = 0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        
        for batch_idx, batch in enumerate(progress_bar):
            try:
                # Get data
                sentinel = batch['sentinel'].to(device)
                ground_truth = batch['ground_truth'].to(device)
                
                # Reshape
                B = sentinel.size(0)
                sentinel = sentinel.view(B, -1)
                ground_truth = ground_truth.view(B, -1)
                
                # Zero gradients
                optimizer.zero_grad(set_to_none=True)
                
                # Forward pass
                outputs = model(sentinel)
                
                # Loss with gradient clipping
                loss = criterion(outputs, ground_truth)
                
                if not torch.isnan(loss) and not torch.isinf(loss):
                    loss.backward()
                    
                    # Print gradient info for first batch
                    if batch_idx == 0:
                        check_gradients(model)
                    
                    # Clip gradients
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
                    
                    optimizer.step()
                    
                    # Update statistics
                    running_loss += loss.item()
                    valid_batches += 1
                    progress_bar.set_postfix({'loss': f'{running_loss/valid_batches:.4f}'})
                
            except RuntimeError as e:
                print(f"\nError in batch {batch_idx}: {str(e)}")
                continue
                
        if valid_batches > 0:
            epoch_loss = running_loss / valid_batches
            print(f'Epoch {epoch+1} Loss: {epoch_loss:.4f}')
            
            # Save best model
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                torch.save(model.state_dict(), 'best_yearly_model.pt')
        else:
            print(f'Epoch {epoch+1} had no valid batches')

def test_dataloader():
    # Parameters
    base_path = "/mnt/guanabana/raid/shared/dropbox/QinLennart"
    batch_size = 8  # Slightly larger batch size for yearly data
    num_epochs = 10  # More epochs since we have less data
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"\nUsing GPU: {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        device = torch.device("cpu")
        print("\nUsing CPU")
    
    try:
        # Set random seeds
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
            
        # Get dataloaders but we'll only use yearly
        _, yearly_loader = create_training_dataloaders(
            base_path=base_path,
            batch_size=batch_size,
            num_workers=2,
            debug=True
        )
        
        # Yearly training
        print("\nTesting Yearly Data:")
        sample_batch = next(iter(yearly_loader))
        
        # Print data shapes
        print("\nInput shape:", sample_batch['sentinel'].shape)
        print("Ground truth shape:", sample_batch['ground_truth'].shape)
        
        # Calculate sizes
        B = sample_batch['sentinel'].size(0)
        input_size = np.prod(sample_batch['sentinel'].shape[1:])
        output_size = np.prod(sample_batch['ground_truth'].shape[1:])
        hidden_size = 128
        
        yearly_model = SimpleMLP(input_size, hidden_size, output_size).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.AdamW(
            yearly_model.parameters(),
            lr=1e-4,  # Slightly larger learning rate for yearly data
            weight_decay=0.01,
            eps=1e-8
        )
        
        print(f"\nYearly Model Architecture:")
        print(yearly_model)
        print(f"Input Size: {input_size}")
        print(f"Output Size: {output_size}")
        
        print("\nTraining Yearly Model:")
        train_model(yearly_model, yearly_loader, criterion, optimizer, device, num_epochs)
        
    except Exception as e:
        print(f"Error during testing: {str(e)}")
        raise

if __name__ == "__main__":
    test_dataloader()