import torch
import torch.nn as nn
from pathlib import Path
import numpy as np
from tqdm import tqdm
import os
import argparse
from datetime import datetime

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.vit_model3_yearly_15 import create_model
from data.my_dataset import create_training_dataloaders

class ViTPredictor:
    def __init__(
        self,
        model,
        checkpoint_path,
        device='cuda',
        output_dir=None
    ):
        self.device = device
        self.model = model.to(device)
        
        # Create output directory
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = Path(f"vit_predictions_{timestamp}")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        print(f"Loading checkpoint from: {checkpoint_path}")
        self._load_checkpoint(checkpoint_path)
        print(f"Predictions will be saved to: {self.output_dir}")

    def _load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Successfully loaded checkpoint from epoch {checkpoint['epoch']}")
            print(f"Training loss: {checkpoint['train_loss']:.4f}")
            if checkpoint.get('val_loss'):
                print(f"Validation loss: {checkpoint['val_loss']:.4f}")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            raise

    def predict_batch(self, sentinel_data):
        """Make predictions for a batch of data"""
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(sentinel_data)
        return predictions

    def calculate_metrics(self, predictions, ground_truth):
        """Calculate prediction metrics"""
        mse = nn.MSELoss()(predictions, ground_truth).item()
        mae = nn.L1Loss()(predictions, ground_truth).item()
        
        # Calculate R² per fraction
        r2_scores = []
        for i in range(predictions.shape[1]):  # For each fraction
            fraction_pred = predictions[:, i].flatten()
            fraction_true = ground_truth[:, i].flatten()
            
            # Calculate R²
            ss_tot = torch.sum((fraction_true - torch.mean(fraction_true))**2)
            ss_res = torch.sum((fraction_true - fraction_pred)**2)
            r2 = 1 - (ss_res / (ss_tot + 1e-8))  # Add small epsilon to avoid division by zero
            r2_scores.append(r2.item())
        
        return {
            'mse': mse,
            'mae': mae,
            'r2_scores': r2_scores
        }

    def predict_dataset(self, dataloader, save_predictions=True):
        """Make predictions for entire dataset"""
        print("\nStarting prediction...")
        
        all_predictions = []
        all_ground_truth = []
        all_locations = []
        metrics_per_batch = []

        for batch in tqdm(dataloader, desc="Predicting"):
            # Move data to device
            sentinel_data = batch['sentinel'].to(self.device)
            ground_truth = batch['ground_truth'].to(self.device)
            locations = batch['location_id']
            
            # Make predictions
            predictions = self.predict_batch(sentinel_data)
            
            # Calculate metrics
            batch_metrics = self.calculate_metrics(predictions, ground_truth)
            metrics_per_batch.append(batch_metrics)
            
            # Store results
            all_predictions.append(predictions.cpu())
            all_ground_truth.append(ground_truth.cpu())
            all_locations.extend(locations)

        # Combine results
        all_predictions = torch.cat(all_predictions, dim=0)
        all_ground_truth = torch.cat(all_ground_truth, dim=0)
        
        # Calculate overall metrics
        overall_metrics = self._calculate_overall_metrics(metrics_per_batch)
        self._save_metrics(overall_metrics)
        
        if save_predictions:
            self._save_predictions(all_predictions, all_ground_truth, all_locations)
        
        return overall_metrics

    def _calculate_overall_metrics(self, metrics_per_batch):
        """Calculate overall metrics from batch metrics"""
        n_batches = len(metrics_per_batch)
        
        # Calculate average MSE and MAE
        avg_mse = sum(m['mse'] for m in metrics_per_batch) / n_batches
        avg_mae = sum(m['mae'] for m in metrics_per_batch) / n_batches
        
        # Calculate average R² for each fraction
        avg_r2_scores = []
        for fraction_idx in range(7):  # Assuming 7 fractions
            avg_r2 = sum(m['r2_scores'][fraction_idx] for m in metrics_per_batch) / n_batches
            avg_r2_scores.append(avg_r2)
        
        return {
            'average_mse': avg_mse,
            'average_mae': avg_mae,
            'average_r2_scores': avg_r2_scores
        }

    def _save_metrics(self, metrics):
        """Save metrics to file"""
        metrics_path = self.output_dir / 'prediction_metrics.txt'
        with open(metrics_path, 'w') as f:
            f.write("Overall Prediction Metrics:\n")
            f.write(f"Average MSE: {metrics['average_mse']:.6f}\n")
            f.write(f"Average MAE: {metrics['average_mae']:.6f}\n")
            f.write("\nR² Scores per Fraction:\n")
            for i, r2 in enumerate(metrics['average_r2_scores'], 1):
                f.write(f"Fraction {i}: {r2:.6f}\n")
        print(f"\nMetrics saved to: {metrics_path}")

    def _save_predictions(self, predictions, ground_truth, locations):
        """Save predictions to file"""
        predictions_path = self.output_dir / 'predictions.npz'
        np.savez(
            predictions_path,
            predictions=predictions.numpy(),
            ground_truth=ground_truth.numpy(),
            locations=locations
        )
        print(f"Predictions saved to: {predictions_path}")

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='ViT Model Prediction Script')
    parser.add_argument('--checkpoint', type=str, required=True,
                      help='Path to model checkpoint')
    parser.add_argument('--data_path', type=str, required=True,
                      help='Path to data directory')
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Batch size for prediction')
    parser.add_argument('--output_dir', type=str, default=None,
                      help='Directory to save predictions')
    parser.add_argument('--device', type=str, default='cuda',
                      help='Device to use (cuda or cpu)')
    args = parser.parse_args()

    # Set device
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    print(f"Using device: {device}")

    try:
        # Create model
        model = create_model()
        print("Model created successfully")

        # Create predictor
        predictor = ViTPredictor(
            model=model,
            checkpoint_path=args.checkpoint,
            device=device,
            output_dir=args.output_dir
        )

        # Create dataloader
        _, yearly_loader = create_training_dataloaders(
            base_path=args.data_path,
            batch_size=args.batch_size,
            num_workers=4,
            debug=False
        )

        # Make predictions
        metrics = predictor.predict_dataset(yearly_loader)

        # Print final metrics
        print("\nPrediction Results:")
        print(f"Average MSE: {metrics['average_mse']:.6f}")
        print(f"Average MAE: {metrics['average_mae']:.6f}")
        print("\nR² Scores per Fraction:")
        for i, r2 in enumerate(metrics['average_r2_scores'], 1):
            print(f"Fraction {i}: {r2:.6f}")

    except Exception as e:
        print(f"Error during prediction: {e}")
        raise

if __name__ == "__main__":
    main()