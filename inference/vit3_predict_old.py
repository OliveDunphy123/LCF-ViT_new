import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import seaborn as sns

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.vit_model3_yearly_15 import create_model
from data.my_dataset import create_training_dataloaders

#plot one of the location one of the fraction for 4 years GT
# import rasterio
# import matplotlib.pyplot as plt

# # Define the file path
# file_path = "/mnt/guanabana/raid/shared/dropbox/QinLennart/training_subset_gt/stacked_2756118_fraction_3.tif"

# # Open the GeoTIFF file
# with rasterio.open(file_path) as src:
#     n_bands = src.count  # Number of bands
#     print(f"Number of bands: {n_bands}")
    
#     # Loop through each band and plot
#     for i in range(1, n_bands + 1):  # Bands are 1-indexed in rasterio
#         band_data = src.read(i)
        
#         plt.figure()
#         plt.imshow(band_data, cmap='viridis')  # Use a suitable colormap
#         plt.title(f"Year {2014 + i}")  # Years 2015-2018
#         plt.colorbar(label='Pixel Value')
#         plt.xlabel("X")
#         plt.ylabel("Y")
#         plt.show()


class ModelEvaluator:
    def __init__(self, model_path, device='cuda'):
        self.device = device
        
        # Load model
        self.model = create_model()
        checkpoint = torch.load(model_path, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        #self.optimizer.load_state_dict(checpoint['optimizer_state_dict'])
        #self.epoch = checkpoint['epoch']
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
                
                # Store predictions and ground truth for metrics
                all_predictions.append(predictions.cpu())
                all_ground_truth.append(ground_truth.cpu())
        
        # Concatenate all batches
        all_predictions = torch.cat(all_predictions, dim=0)
        all_ground_truth = torch.cat(all_ground_truth, dim=0)
        
        # Calculate average loss
        avg_loss = total_loss / len(dataloader)
        
        # Calculate metrics for each timestep
        timestep_metrics = []
        for t in range(all_predictions.shape[2]):  # For each timestep
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
        
        # Plot R² scores for each fraction over time
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
        
        # Plot RMSE for each fraction
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

    def plot_predictions(self, predictions, ground_truth, save_dir, sample_idx=0):
        """Plot example predictions vs ground truth"""
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True)
        
        # Plot for each fraction
        for f in range(7):
            plt.figure(figsize=(15, 5))
            
            # Get data for this fraction
            pred = predictions[sample_idx, f].numpy()
            true = ground_truth[sample_idx, f].numpy()
            
            # Create subplots for each timestep
            for t in range(4):
                plt.subplot(1, 4, t+1)
                
                # Plot ground truth
                plt.subplot(2, 4, t+1)
                sns.heatmap(true[t], vmin=0, vmax=1, cmap='YlOrRd')
                plt.title(f'Ground Truth\nTimestep {t}')
                
                # Plot prediction
                plt.subplot(2, 4, t+5)
                sns.heatmap(pred[t], vmin=0, vmax=1, cmap='YlOrRd')
                plt.title(f'Prediction\nTimestep {t}')
            
            plt.suptitle(f'Fraction {f+1} - Predictions vs Ground Truth')
            plt.tight_layout()
            plt.savefig(save_dir / f'fraction_{f+1}_comparison.png')
            plt.close()

    def plot_comparison_separate(self, predictions, ground_truth, sentinel_data, save_dir, sample_idx=0, location_ids=None):
        """
        Create three separate plots for Sentinel-2, ground truth, and predictions
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True)

        # Get location ID for title
        # loc_id = location_ids[sample_idx].split('_')[0] if location_ids is not None else "Unknown"
        location_str = ""
        if location_ids is not None:
            try:
                loc_id, year_str = location_ids[sample_idx].split('_')
                year = int(year_str)
                location_str = f" (Location ID: {loc_id}, Starting Year: {year})"
            except:
                pass
            
        # Plot for each fraction
        for f in range(7):
            # 1. Plot Sentinel-2 RGB Images
            plt.figure(figsize=(20, 5))
            plt.suptitle(f'Sentinel-2 RGB Images (Location ID: {loc_id}, Fraction {f+1})', fontsize=14)
            
            sentinel = sentinel_data[sample_idx].numpy()  # [10, T, 15, 15]
            for t in range(4):
                plt.subplot(1, 4, t+1)
                # Create RGB composite
                rgb = np.stack([
                    sentinel[3, t],  # Red (B04)
                    sentinel[2, t],  # Green (B03)
                    sentinel[1, t],  # Blue (B02)
                ], axis=-1)
                
                if np.any(rgb):
                    # Normalize non-zero values
                    for c in range(3):
                        channel = rgb[:, :, c]
                        if np.any(channel):
                            min_val, max_val = np.percentile(channel[channel > 0], (2, 98))
                            rgb[:, :, c] = np.clip((channel - min_val) / (max_val - min_val), 0, 1)
                    plt.imshow(rgb)
                else:
                    plt.text(0.5, 0.5, 'No data', ha='center', va='center')
                plt.title(f'Year {2015 + t}')
                plt.axis('off')
            plt.tight_layout()
            plt.savefig(save_dir / f'fraction_{f+1}_sentinel.png', dpi=300, bbox_inches='tight')
            plt.close()

            # 2. Plot Ground Truth
            plt.figure(figsize=(20, 5))
            plt.suptitle(f'Ground Truth (Location ID: {loc_id}, Fraction {f+1})', fontsize=14)
            
            true = ground_truth[sample_idx].numpy()  # [7, T, 5, 5]
            for t in range(4):
                plt.subplot(1, 4, t+1)
                im = plt.imshow(true[f, t], cmap='viridis', vmin=0, vmax=1)
                plt.colorbar(im, label='Fraction')
                plt.title(f'Year {2015 + t}')
                plt.axis('on')
                # Print value range for debugging
                print(f"Ground Truth Year {2015 + t} - Range: [{true[f, t].min():.3f}, {true[f, t].max():.3f}]")
            plt.tight_layout()
            plt.savefig(save_dir / f'fraction_{f+1}_ground_truth.png', dpi=300, bbox_inches='tight')
            plt.close()

            # 3. Plot Predictions
            plt.figure(figsize=(20, 5))
            plt.suptitle(f'Predictions (Location ID: {loc_id}, Fraction {f+1})', fontsize=14)
            
            pred = predictions[sample_idx].numpy()  # [7, T, 5, 5]
            for t in range(4):
                plt.subplot(1, 4, t+1)
                im = plt.imshow(pred[f, t], cmap='viridis', vmin=0, vmax=1)
                plt.colorbar(im, label='Fraction')
                plt.title(f'Year {2015 + t}')
                plt.axis('on')
                # Add correlation coefficient
                corr = np.corrcoef(true[f, t].flatten(), pred[f, t].flatten())[0, 1]
                plt.xlabel(f'Correlation: {corr:.3f}')
            plt.tight_layout()
            plt.savefig(save_dir / f'fraction_{f+1}_predictions.png', dpi=300, bbox_inches='tight')
            plt.close()

            # 4. Plot metrics for this fraction
            plt.figure(figsize=(10, 5))
            plt.suptitle(f'Metrics Over Time (Location ID: {loc_id}, Fraction {f+1})', fontsize=14)
            
            years = range(2015, 2019)
            metrics = []
            for t in range(4):
                true_t = true[f, t].flatten()
                pred_t = pred[f, t].flatten()
                corr = np.corrcoef(true_t, pred_t)[0, 1]
                rmse = np.sqrt(np.mean((true_t - pred_t) ** 2))
                mae = np.mean(np.abs(true_t - pred_t))
                metrics.append({'corr': corr, 'rmse': rmse, 'mae': mae})
            
            plt.subplot(1, 3, 1)
            plt.plot(years, [m['corr'] for m in metrics], 'o-')
            plt.title('Correlation')
            plt.ylim(0, 1)
            
            plt.subplot(1, 3, 2)
            plt.plot(years, [m['rmse'] for m in metrics], 'o-')
            plt.title('RMSE')
            
            plt.subplot(1, 3, 3)
            plt.plot(years, [m['mae'] for m in metrics], 'o-')
            plt.title('MAE')
            
            plt.tight_layout()
            plt.savefig(save_dir / f'fraction_{f+1}_metrics.png', dpi=300, bbox_inches='tight')
            plt.close()

    # def plot_full_comparison(self, predictions, ground_truth, sentinel_data, save_dir, sample_idx=0,location_ids=None):
    #     """
    #     Plot Sentinel imagery, ground truth, and predictions side by side with 
        
    #     Args:
    #         predictions: Model predictions [B, 7, T, 5, 5]
    #         ground_truth: Ground truth data [B, 7, T, 5, 5]
    #         sentinel_data: Sentinel imagery [B, 10, T, 15, 15]
    #         save_dir: Directory to save plots
    #         sample_idx: Index of sample to plot
    #         location_ids: List of location IDs corresponding to the samples
    #     """
    #     save_dir = Path(save_dir)
    #     save_dir.mkdir(exist_ok=True)
    
    #     # Get data for the selected sample
    #     pred = predictions[sample_idx].numpy()  # [7, T, 5, 5]
    #     true = ground_truth[sample_idx].numpy()  # [7, T, 5, 5]
    #     sentinel = sentinel_data[sample_idx].numpy()  # [10, T, 15, 15]


    #     # Extract location and year info
    #     location_str = ""
    #     if location_ids is not None:
    #         try:
    #             loc_id, year_str = location_ids[sample_idx].split('_')
    #             year = int(year_str)
    #             location_str = f" (Location ID: {loc_id}, Starting Year: {year})"
    #         except:
    #             pass


    #     # Create RGB composite from Sentinel bands (using bands 4,3,2 for RGB)
    #     rgb_sentinel = np.stack([
    #     sentinel[3],  # Red (band 4)
    #     sentinel[2],  # Green (band 3)
    #     sentinel[1],  # Blue (band 2)
    #     ], axis=-1)
    
    
    #     # Plot for each fraction
    #     for f in range(7):
    #         plt.figure(figsize=(20, 5*4))
    #         plt.suptitle(f'Fraction {f+1} Comparison{location_str}', fontsize=16)
        
    #         for t in range(4):
    #             year_label = f"{2015 + t}" if t < 4 else "Unknown"
                
    #             # Plot Sentinel RGB
    #             plt.subplot(4, 3, t*3 + 1)
    #             if np.any(rgb_sentinel[t]):
    #                 # Normalize the non-zero RGB values
    #                 rgb_norm = rgb_sentinel[t].copy()
    #                 for c in range(3):
    #                     channel = rgb_norm[:, :, c]
    #                     if np.any(channel):
    #                         min_val, max_val = np.percentile(channel[channel > 0], (2, 98))
    #                         rgb_norm[:, :, c] = np.clip((channel - min_val) / (max_val - min_val), 0, 1)
    #                 plt.imshow(rgb_norm)
    #             else:
    #                 plt.text(0.5, 0.5, 'No data', ha='center', va='center')
    #             plt.title(f'Sentinel RGB ({year})')
    #             plt.axis('off')
            
    #             # Plot Ground Truth using viridis colormap for consistency
    #             plt.subplot(4, 3, t*3 + 2)
    #             plt.imshow(true[f, t], cmap='viridis', vmin=0, vmax=1)
    #             plt.colorbar(label='Fraction')
    #             plt.title(f'Ground Truth ({year_label})')
    #             plt.axis('on')
            
    #             # Plot Prediction using same colormap
    #             plt.subplot(4, 3, t*3 + 3)
    #             plt.imshow(pred[f, t], cmap='viridis', vmin=0, vmax=1)
    #             plt.colorbar(label='Fraction')
    #             plt.title(f'Prediction ({year_label})')
    #             plt.axis('on')
            
    #             # Add correlation coefficient
    #             corr = np.corrcoef(true[f, t].flatten(), pred[f, t].flatten())[0, 1]
    #             plt.xlabel(f'Correlation: {corr:.3f}')
        
    #         plt.tight_layout()
    #         plt.savefig(save_dir / f'fraction_{f+1}_full_comparison.png', dpi=300, bbox_inches='tight')
    #         plt.close()

    #     # Plot average metrics across all timesteps
    #     plt.figure(figsize=(15, 5))
    #     correlations = []
    #     rmse_values = []
    #     mae_values = []

    #     for f in range(7):
    #         true_flat = true[f].flatten()
    #         pred_flat = pred[f].flatten()
        
    #         correlations.append(np.corrcoef(true_flat, pred_flat)[0, 1])
    #         rmse_values.append(np.sqrt(np.mean((true_flat - pred_flat) ** 2)))
    #         mae_values.append(np.mean(np.abs(true_flat - pred_flat)))

    #     # Plot metrics
    #     fractions = [f'F{i+1}' for i in range(7)]

    #     plt.subplot(1, 3, 1)
    #     plt.bar(fractions, correlations)
    #     plt.title(f'Correlation by Fraction{location_str}')
    #     plt.ylim(0, 1)
        
    #     plt.subplot(1, 3, 2)
    #     plt.bar(fractions, rmse_values)
    #     plt.title(f'RMSE by Fraction{location_str}')
        
    #     plt.subplot(1, 3, 3)
    #     plt.bar(fractions, mae_values)
    #     plt.title(f'MAE by Fraction{location_str}')
        
    #     plt.tight_layout()
    #     plt.savefig(save_dir / 'average_metrics.png', dpi=300, bbox_inches='tight')
    #     plt.close()

    #     # Save raw values for verification
    #     np.save(save_dir / f'ground_truth_raw_sample{sample_idx}.npy', true)
        
def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    try:
        # Get most recent model checkpoint
        base_dir = Path('/mnt/guanabana/raid/home/qinxu/Python/LCF-ViT')
        result_dirs = list(base_dir.glob('vit_test_results_*'))
        if not result_dirs:
            raise ValueError(f"No results directories found in {base_dir}")

        results_dir = sorted(result_dirs)[-1]
        print(f"Found results directory: {results_dir}")
        
        checkpoint_files = sorted(results_dir.glob('checkpoint_epoch_*.pth'))
        if not checkpoint_files:
            raise ValueError(f"No checkpoint files found in {results_dir}")
        latest_checkpoint = checkpoint_files[-1]
        plots_dir = results_dir / 'evaluation_plots'
        print(f"Using checkpoint: {latest_checkpoint}")
        
        # Create evaluator
        evaluator = ModelEvaluator(latest_checkpoint, device)
        
        # Load data
        _, yearly_loader = create_training_dataloaders(
            base_path="/mnt/guanabana/raid/shared/dropbox/QinLennart",
            batch_size=32
        )
        
        # Evaluate
        print("Evaluating model...")
        avg_loss, timestep_metrics, predictions, ground_truth = evaluator.evaluate(yearly_loader)
        
        print(f"\nAverage Loss: {avg_loss:.4f}")
        
        # Get a batch of data for visualization
        sample_batch = next(iter(yearly_loader))
        sentinel_data = sample_batch['sentinel']
        location_ids = sample_batch['location_id'] if 'location_id' in sample_batch else None
        
        # Create plots directory if it doesn't exist
        plots_dir = results_dir / 'evaluation_plots'
        plots_dir.mkdir(exist_ok=True)
        
        # Plot full comparison with location IDs
        print("\nGenerating detailed comparison plots...")
        evaluator.plot_comparison_separate(
        predictions[:8],  # First 8 samples
        ground_truth[:8],
        sentinel_data[:8],
        plots_dir,
        location_ids=location_ids[:8] if location_ids is not None else None
)

        # Print metrics for each timestep
        for t, metrics in enumerate(timestep_metrics):
            print(f"\nTimestep {t}:")
            print(f"Average R²: {np.mean([r2 for r2 in metrics['r2']]):.4f}")
            print(f"Average RMSE: {np.mean([rmse for rmse in metrics['rmse']]):.4f}")
            print(f"Average MAE: {np.mean([mae for mae in metrics['mae']]):.4f}")
        
        # Plot metrics
        print("\nGenerating plots...")
        evaluator.plot_metrics(timestep_metrics, plots_dir)
        
        # Plot sample predictions
        evaluator.plot_predictions(predictions, ground_truth, plots_dir)
        
        print(f"\nEvaluation complete. Plots saved to: {plots_dir}")
    
    except Exception as e:
        print(f"Error: {str(e)}")
        print(f"Current working directory: {Path.cwd()}")
        print(f"Available directories: {list(Path('/mnt/guanabana/raid/home/qinxu/Python/LCF-ViT').glob('*'))}")
        raise


if __name__ == "__main__":
    main()