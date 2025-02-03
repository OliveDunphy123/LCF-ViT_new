import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import seaborn as sns

from vit_model3_yearly_15 import create_model
from my_dataset import create_training_dataloaders

class ModelEvaluator:
    def __init__(self, model_path, device='cuda'):
        self.device = device
        
        # Load model
        self.model = create_model()
        checkpoint = torch.load(model_path, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(device)
        self.model.eval()
        
        # Metrics
        self.criterion = nn.MSELoss()

    def calculate_metrics(self, predictions, ground_truth):
        """Calculate various metrics for evaluation"""
        # Convert to numpy for easier calculation
        pred = predictions.cpu().numpy()
        true = ground_truth.cpu().numpy()
        
        # Calculate metrics for each fraction
        metrics = {
            'mse': [],
            'rmse': [],
            'r2': [],
            'mae': []
        }
        
        for f in range(pred.shape[1]):  # For each fraction
            fraction_pred = pred[:, f].flatten()
            fraction_true = true[:, f].flatten()
            
            mse = np.mean((fraction_pred - fraction_true) ** 2)
            rmse = np.sqrt(mse)
            r2 = r2_score(fraction_true, fraction_pred)
            mae = np.mean(np.abs(fraction_pred - fraction_true))
            
            metrics['mse'].append(mse)
            metrics['rmse'].append(rmse)
            metrics['r2'].append(r2)
            metrics['mae'].append(mae)
            
            # Print metrics for each fraction
            print(f"\nFraction {f+1} metrics:")
            print(f"MSE: {mse:.4f}")
            print(f"RMSE: {rmse:.4f}")
            print(f"R²: {r2:.4f}")
            print(f"MAE: {mae:.4f}")
        
        return metrics

    def evaluate(self, dataloader):
        """Evaluate model on given dataloader"""
        all_predictions = []
        all_ground_truth = []
        total_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                sentinel_data = batch['sentinel'].to(self.device)
                ground_truth = batch['ground_truth'].to(self.device)
                
                predictions = self.model(sentinel_data)
                loss = self.criterion(predictions, ground_truth)
                total_loss += loss.item()
                
                all_predictions.append(predictions.cpu())
                all_ground_truth.append(ground_truth.cpu())
        
        all_predictions = torch.cat(all_predictions, dim=0)
        all_ground_truth = torch.cat(all_ground_truth, dim=0)
        avg_loss = total_loss / len(dataloader)
        
        timestep_metrics = []
        for t in range(all_predictions.shape[2]):  # For each timestep
            print(f"\nTimestep {t} metrics:")
            metrics = self.calculate_metrics(
                all_predictions[:, :, t], 
                all_ground_truth[:, :, t]
            )
            timestep_metrics.append(metrics)
        
        return avg_loss, timestep_metrics, all_predictions, all_ground_truth

    def plot_metrics(self, metrics, save_dir):
        """Plot and save evaluation metrics"""
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True)
        
        # Plot R² scores
        plt.figure(figsize=(12, 6))
        timesteps = range(len(metrics))
        fractions = ['Fraction ' + str(i+1) for i in range(7)]
        
        for f in range(7):
            r2_scores = [m['r2'][f] for m in metrics]
            plt.plot(timesteps, r2_scores, marker='o', label=fractions[f])
        
        plt.xlabel('Time Step')
        plt.ylabel('R² Score')
        plt.title('R² Score Over Time by Fraction')
        plt.legend()
        plt.grid(True)
        plt.savefig(save_dir / 'r2_scores.png')
        plt.close()
        
        # Plot RMSE
        plt.figure(figsize=(10, 6))
        avg_rmse = [np.mean([m['rmse'][f] for m in metrics]) for f in range(7)]
        plt.bar(fractions, avg_rmse)
        plt.xlabel('Fraction')
        plt.ylabel('Average RMSE')
        plt.title('Average RMSE by Fraction')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(save_dir / 'rmse_by_fraction.png')
        plt.close()

    def plot_full_comparison(self, predictions, ground_truth, sentinel_data, save_dir, sample_idx=0):
        """Plot Sentinel imagery, ground truth, and predictions side by side"""
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True)
        
        pred = predictions[sample_idx].numpy()
        true = ground_truth[sample_idx].numpy()
        sentinel = sentinel_data[sample_idx].numpy()
        
        # Create RGB composite
        rgb_sentinel = np.stack([
            sentinel[3],  # Red (band 4)
            sentinel[2],  # Green (band 3)
            sentinel[1],  # Blue (band 2)
        ], axis=-1)
        
        # Normalize RGB values
        for t in range(rgb_sentinel.shape[0]):
            for c in range(3):
                channel = rgb_sentinel[t, :, :, c]
                min_val, max_val = np.percentile(channel, (2, 98))
                rgb_sentinel[t, :, :, c] = np.clip((channel - min_val) / (max_val - min_val), 0, 1)
        
        # Plot comparisons
        for f in range(7):
            plt.figure(figsize=(20, 5*4))
            plt.suptitle(f'Fraction {f+1} Comparison', fontsize=16)
            
            for t in range(4):
                # Sentinel RGB
                plt.subplot(4, 3, t*3 + 1)
                plt.imshow(rgb_sentinel[t])
                plt.title(f'Sentinel RGB (t={t})')
                plt.axis('off')
                
                # Ground Truth
                plt.subplot(4, 3, t*3 + 2)
                sns.heatmap(true[f, t], vmin=0, vmax=1, cmap='YlOrRd', cbar_kws={'label': 'Fraction'})
                plt.title(f'Ground Truth (t={t})')
                
                # Prediction
                plt.subplot(4, 3, t*3 + 3)
                sns.heatmap(pred[f, t], vmin=0, vmax=1, cmap='YlOrRd', cbar_kws={'label': 'Fraction'})
                plt.title(f'Prediction (t={t})')
                
                # Add correlation
                corr = np.corrcoef(true[f, t].flatten(), pred[f, t].flatten())[0, 1]
                plt.xlabel(f'Correlation: {corr:.3f}')
            plt.tight_layout()
            plt.savefig(save_dir / f'fraction_{f+1}_full_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()

def main():
    try:
        # Set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        # Specify exact path to results directory
        results_dir = Path('/mnt/guanabana/raid/home/qinxu/Python/LCF-ViT/vit_test_results_20241212_144938')
        if not results_dir.exists():
            raise ValueError(f"Results directory not found: {results_dir}")
        print(f"Using results directory: {results_dir}")
        
        # Find latest checkpoint
        checkpoint_files = sorted(results_dir.glob('checkpoint_epoch_*.pth'))
        if not checkpoint_files:
            raise ValueError(f"No checkpoint files found in {results_dir}")
        
        latest_checkpoint = checkpoint_files[-1]
        plots_dir = results_dir / 'evaluation_plots'
        print(f"Using checkpoint: {latest_checkpoint}")
        
        # Create evaluator and load data
        evaluator = ModelEvaluator(latest_checkpoint, device)
        _, yearly_loader = create_training_dataloaders(
            base_path="/mnt/guanabana/raid/shared/dropbox/QinLennart",
            batch_size=32
        )
        
        # Evaluate model
        print("\nEvaluating model...")
        avg_loss, timestep_metrics, predictions, ground_truth = evaluator.evaluate(yearly_loader)
        print(f"\nAverage Loss: {avg_loss:.4f}")
        
        # Create visualizations
        plots_dir.mkdir(exist_ok=True)
        print("\nGenerating plots...")
        
        # Get sample data for visualization
        sample_batch = next(iter(yearly_loader))
        
        # Plot full comparison
        print("\nGenerating detailed comparison plots...")
        evaluator.plot_full_comparison(
            predictions[:8],
            ground_truth[:8],
            sample_batch['sentinel'][:8],
            plots_dir
        )
        
        # Plot metrics
        evaluator.plot_metrics(timestep_metrics, plots_dir)
        
        print(f"\nEvaluation complete. Plots saved to: {plots_dir}")
        
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        print(f"Current working directory: {Path.cwd()}")
        print("Available directories in LCF-ViT:")
        for item in Path('/mnt/guanabana/raid/home/qinxu/Python/LCF-ViT').iterdir():
            print(f"  - {item.name}")
        raise

if __name__ == "__main__":
    main()