import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from my_dataset import create_training_dataloaders
import random

class SimpleMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleMLP, self).__init__()
        self.flatten = nn.Flatten()
        
        self.reduce_dim = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
        )
        
        self.main_network = nn.Sequential(
            nn.Linear(256, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = self.flatten(x)
        x = (x - x.mean(dim=1, keepdim=True)) / (x.std(dim=1, keepdim=True) + 1e-8)
        x = self.reduce_dim(x)
        return self.main_network(x)

def create_test_loader(yearly_dataset, batch_size=8):
    """Create test loader from 15% of the data"""
    dataset_size = len(yearly_dataset)
    test_size = int(0.15 * dataset_size)
    
    indices = list(range(dataset_size))
    random.seed(42)
    test_indices = random.sample(indices, test_size)
    
    test_dataset = Subset(yearly_dataset, test_indices)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )
    
    return test_loader

def evaluate_model(model, test_loader, device):
    """Evaluate model on test data"""
    model.eval()
    total_loss = 0
    criterion = nn.MSELoss()
    all_predictions = []
    all_targets = []
    location_ids = []
    
    print("\nEvaluating model on test data...")
    
    with torch.no_grad():
        for batch in test_loader:
            sentinel = batch['sentinel'].to(device)
            ground_truth = batch['ground_truth'].to(device)
            
            B = sentinel.size(0)
            sentinel = sentinel.view(B, -1)
            ground_truth = ground_truth.view(B, -1)
            
            outputs = model(sentinel)
            loss = criterion(outputs, ground_truth)
            total_loss += loss.item()
            
            all_predictions.append(outputs.cpu().numpy())
            all_targets.append(ground_truth.cpu().numpy())
            location_ids.extend(batch['location_id'])
    
    avg_loss = total_loss / len(test_loader)
    print(f"Average Test Loss: {avg_loss:.4f}")
    
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    return all_predictions, all_targets, location_ids, avg_loss

def plot_fraction_distributions(predictions, targets):
    """Plot distribution of predictions vs targets for each fraction"""
    num_fractions = 7
    fraction_size = targets.shape[1] // num_fractions
    
    plt.figure(figsize=(20, 10))
    for i in range(num_fractions):
        plt.subplot(2, 4, i+1)
        start_idx = i * fraction_size
        end_idx = (i + 1) * fraction_size
        
        # Calculate mean values for this fraction
        pred_mean = predictions[:, start_idx:end_idx].mean(axis=1)
        target_mean = targets[:, start_idx:end_idx].mean(axis=1)
        
        # Plot distributions
        sns.kdeplot(data=target_mean, label='Target', color='blue', alpha=0.5)
        sns.kdeplot(data=pred_mean, label='Prediction', color='red', alpha=0.5)
        
        plt.title(f'Fraction {i+1} Distribution')
        plt.xlabel('Value')
        plt.ylabel('Density')
        plt.legend()
    
    plt.tight_layout()
    plt.show()

def plot_error_analysis(predictions, targets):
    """Plot error analysis for each fraction"""
    num_fractions = 7
    fraction_size = targets.shape[1] // num_fractions
    
    # Calculate errors for each fraction
    errors = []
    for i in range(num_fractions):
        start_idx = i * fraction_size
        end_idx = (i + 1) * fraction_size
        fraction_errors = np.mean(np.abs(
            predictions[:, start_idx:end_idx] - targets[:, start_idx:end_idx]
        ), axis=1)
        errors.append(fraction_errors)
    
    # Plot error boxplots
    plt.figure(figsize=(12, 6))
    plt.boxplot(errors, labels=[f'Fraction {i+1}' for i in range(num_fractions)])
    plt.title('Error Distribution by Fraction')
    plt.ylabel('Absolute Error')
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_scatter_analysis(predictions, targets):
    """Plot scatter plots of predictions vs targets"""
    num_fractions = 7
    fraction_size = targets.shape[1] // num_fractions
    
    plt.figure(figsize=(20, 10))
    for i in range(num_fractions):
        plt.subplot(2, 4, i+1)
        start_idx = i * fraction_size
        end_idx = (i + 1) * fraction_size
        
        pred_mean = predictions[:, start_idx:end_idx].mean(axis=1)
        target_mean = targets[:, start_idx:end_idx].mean(axis=1)
        
        plt.scatter(target_mean, pred_mean, alpha=0.5)
        plt.plot([0, 1], [0, 1], 'r--', alpha=0.5)  # Perfect prediction line
        
        plt.title(f'Fraction {i+1}')
        plt.xlabel('Target')
        plt.ylabel('Prediction')
        
        # Calculate correlation
        corr = np.corrcoef(target_mean, pred_mean)[0, 1]
        plt.text(0.05, 0.95, f'Correlation: {corr:.3f}', 
                transform=plt.gca().transAxes)
    
    plt.tight_layout()
    plt.show()

def analyze_predictions(predictions, targets, location_ids):
    """Comprehensive analysis of predictions"""
    # Overall statistics
    mae = np.mean(np.abs(predictions - targets))
    mse = np.mean((predictions - targets) ** 2)
    rmse = np.sqrt(mse)
    
    print(f"\nPrediction Statistics:")
    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    
    # Per-fraction statistics
    fraction_size = targets.shape[1] // 7
    for i in range(7):
        start_idx = i * fraction_size
        end_idx = (i + 1) * fraction_size
        fraction_mae = np.mean(np.abs(predictions[:, start_idx:end_idx] - targets[:, start_idx:end_idx]))
        print(f"Fraction {i+1} MAE: {fraction_mae:.4f}")
    
    # Plot visualizations
    print("\nGenerating visualizations...")
    
    # 1. Distribution plots
    print("Plotting fraction distributions...")
    plot_fraction_distributions(predictions, targets)
    
    # 2. Error analysis
    print("Plotting error analysis...")
    plot_error_analysis(predictions, targets)
    
    # 3. Scatter plots
    print("Plotting scatter analysis...")
    plot_scatter_analysis(predictions, targets)
    
    # Print some example predictions
    print("\nExample predictions for first few samples:")
    for i in range(min(5, len(location_ids))):
        pred_means = [predictions[i, j*fraction_size:(j+1)*fraction_size].mean() for j in range(7)]
        target_means = [targets[i, j*fraction_size:(j+1)*fraction_size].mean() for j in range(7)]
        print(f"\nLocation {location_ids[i]}:")
        print("Predictions:", [f"{x:.3f}" for x in pred_means])
        print("Targets:    ", [f"{x:.3f}" for x in target_means])

def main():
    base_path = "/mnt/guanabana/raid/shared/dropbox/QinLennart"
    batch_size = 8
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        _, yearly_loader = create_training_dataloaders(
            base_path=base_path,
            batch_size=batch_size,
            debug=True
        )
        
        test_loader = create_test_loader(yearly_loader.dataset, batch_size)
        print(f"Test set size: {len(test_loader.dataset)} samples")
        
        sample_batch = next(iter(yearly_loader))
        input_size = np.prod(sample_batch['sentinel'].shape[1:])
        output_size = np.prod(sample_batch['ground_truth'].shape[1:])
        hidden_size = 128
        
        model = SimpleMLP(input_size, hidden_size, output_size).to(device)
        model.load_state_dict(torch.load('best_yearly_model.pt'))
        
        predictions, targets, location_ids, test_loss = evaluate_model(model, test_loader, device)
        analyze_predictions(predictions, targets, location_ids)
        
    except Exception as e:
        print(f"Error during testing: {str(e)}")
        raise

if __name__ == "__main__":
    main()