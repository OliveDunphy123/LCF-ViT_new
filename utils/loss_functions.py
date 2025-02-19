import torch
import torch.nn as nn
import torch.nn.functional as F

class MSEAndL1Loss(nn.Module):
    """Combines MSE and L1 loss with weights"""
    def __init__(self, mse_weight=1.0, l1_weight=0.5):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        self.mse_weight = mse_weight
        self.l1_weight = l1_weight

    def forward(self, pred, target):
        return self.mse_weight * self.mse_loss(pred, target) + \
               self.l1_weight * self.l1_loss(pred, target)

class SmoothL1Loss(nn.Module):
    """Uses PyTorch's SmoothL1Loss with temporal smoothness"""
    def __init__(self, smooth_weight=0.1, beta=1.0):
        super().__init__()
        self.smooth_l1 = nn.SmoothL1Loss(beta=beta)
        self.smooth_weight = smooth_weight

    def forward(self, pred, target):
        main_loss = self.smooth_l1(pred, target)
        # Add temporal smoothness
        temp_diff = pred[:, :, 1:] - pred[:, :, :-1]
        smooth_loss = torch.mean(torch.abs(temp_diff))
        return main_loss + self.smooth_weight * smooth_loss

# class CrossEntropyLoss(nn.Module):
#     """Modified CrossEntropy for regression data"""
#     def __init__(self, num_bins=10):
#         super().__init__()
#         self.num_bins = num_bins
#         self.ce_loss = nn.CrossEntropyLoss()

#     def forward(self, pred, target):
#         # Convert continuous values to discrete bins
#         bins = torch.linspace(0, 1, self.num_bins).to(pred.device)
#         # Find nearest bin for both pred and target
#         pred_binned = torch.argmin(torch.abs(pred.unsqueeze(-1) - bins), dim=-1)
#         target_binned = torch.argmin(torch.abs(target.unsqueeze(-1) - bins), dim=-1)
#         return self.ce_loss(pred_binned.float(), target_binned)

class L2RegLoss(nn.Module):
    """MSE Loss with L2 regularization"""
    def __init__(self, lambda_reg=0.01):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        self.lambda_reg = lambda_reg

    def forward(self, pred, target, model=None):
        mse = self.mse_loss(pred, target)
        if model is not None:
            # Add L2 regularization
            l2_reg = torch.tensor(0.).to(pred.device)
            for param in model.parameters():
                l2_reg += torch.norm(param)
            return mse + self.lambda_reg * l2_reg
        return mse
class CrossEntropyLoss(nn.Module):
    def __init__(self, num_bins=20, smoothing=0.1):
        super().__init__()
        self.num_bins = num_bins
        self.smoothing = smoothing
        
    def forward(self, pred, target):
        # Flatten all dimensions except the last
        #pred_shape = pred.shape
        pred_flat = pred.reshape(-1)
        target_flat = target.reshape(-1)
        
        # Create bins and bin the targets
        bins = torch.linspace(0, 1, self.num_bins).to(pred.device)
        target_binned = torch.bucketize(target_flat, bins) - 1
        target_binned = torch.clamp(target_binned, 0, self.num_bins - 1)
        
        # Create logits
        pred_expanded = pred_flat.unsqueeze(-1).expand(-1, self.num_bins)
        bins_expanded = bins.view(1, -1).expand_as(pred_expanded)
        logits = -torch.abs(pred_expanded - bins_expanded) / 0.1
        
        # Apply label smoothing
        if self.smoothing > 0:
            target_one_hot = F.one_hot(target_binned, self.num_bins).float()
            target_smooth = (1 - self.smoothing) * target_one_hot + self.smoothing / self.num_bins
            loss = -(target_smooth * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
        else:
            loss = F.cross_entropy(logits, target_binned)
            
        return loss

# class HuberFocalLoss(nn.Module):
#     """Combines Huber loss with focal loss concept for handling outliers"""
#     def __init__(self, beta=1.0, alpha=2.0, gamma=0.5):
#         super().__init__()
#         self.beta = beta
#         self.alpha = alpha  # Controls focus on hard examples
#         self.gamma = gamma  # Modulates the focal weight
        
#     def forward(self, pred, target):
#         diff = torch.abs(pred - target)
#         huber_loss = torch.where(diff < self.beta,
#                                 0.5 * diff * diff,
#                                 self.beta * diff - 0.5 * self.beta * self.beta)
        
#         # Focal weight based on prediction error
#         focal_weight = (diff / diff.max()).pow(self.gamma)
#         focal_weight = self.alpha * focal_weight + (1 - self.alpha)
        
#         return (focal_weight * huber_loss).mean()

# class DistributionLoss(nn.Module):
#     """Loss that considers the distribution of land cover fractions"""
#     def __init__(self, kl_weight=0.1):
#         super().__init__()
#         self.kl_weight = kl_weight
#         self.mse_loss = nn.MSELoss()
        
#     def forward(self, pred, target):
#         # Basic MSE loss
#         mse = self.mse_loss(pred, target)
        
#         # KL divergence between predicted and target distributions
#         pred_dist = F.softmax(pred, dim=-1)
#         target_dist = F.softmax(target, dim=-1)
#         kl_div = F.kl_div(pred_dist.log(), target_dist, reduction='batchmean')
        
#         # Ensure predictions sum to approximately 1
#         sum_constraint = torch.abs(pred.sum(dim=-1) - 1.0).mean()
        
#         return mse + self.kl_weight * (kl_div + sum_constraint)

# class TemporalConsistencyLoss(nn.Module):
#     """Loss that emphasizes temporal consistency in predictions"""
#     def __init__(self, smooth_weight=0.1, trend_weight=0.05):
#         super().__init__()
#         self.smooth_weight = smooth_weight
#         self.trend_weight = trend_weight
#         self.mse_loss = nn.MSELoss()
        
#     def forward(self, pred, target):
#         # Base MSE loss
#         mse = self.mse_loss(pred, target)
        
#         # Temporal smoothness (penalize sudden changes)
#         temp_diff = pred[:, :, 1:] - pred[:, :, :-1]
#         smooth_loss = torch.abs(temp_diff).mean()
        
#         # Trend consistency (changes should follow target trends)
#         target_diff = target[:, :, 1:] - target[:, :, :-1]
#         trend_loss = self.mse_loss(temp_diff, target_diff)
        
#         return mse + self.smooth_weight * smooth_loss + self.trend_weight * trend_loss

# class CompositeMultiLoss(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.mse_loss = nn.MSELoss()
#         self.l1_loss = nn.L1Loss()
#         self.huber_loss = nn.HuberLoss(delta=1.0)
        
#         # Learnable weights
#         self.mse_weight = nn.Parameter(torch.tensor(1.0))
#         self.l1_weight = nn.Parameter(torch.tensor(1.0))
#         self.huber_weight = nn.Parameter(torch.tensor(1.0))
        
#     def forward(self, pred, target):
#         # Compute individual losses
#         mse = self.mse_loss(pred, target)
#         l1 = self.l1_loss(pred, target)
#         huber = self.huber_loss(pred, target)
        
#         # Get normalized weights
#         weights = F.softmax(torch.stack([
#             self.mse_weight,
#             self.l1_weight,
#             self.huber_weight
#         ]), dim=0)
        
#         # Combine losses
#         total_loss = (
#             weights[0] * mse +
#             weights[1] * l1 +
#             weights[2] * huber
#         )
        
#         return total_loss

# class BoundaryAwareLoss(nn.Module):
#     """Loss that pays special attention to boundary values (0 and 1)"""
#     def __init__(self, boundary_weight=2.0):
#         super().__init__()
#         self.boundary_weight = boundary_weight
#         self.mse_loss = nn.MSELoss(reduction='none')
        
#     def forward(self, pred, target):
#         mse = self.mse_loss(pred, target)
        
#         # Identify boundary cases (values close to 0 or 1)
#         boundary_mask = (target < 0.1) | (target > 0.9)
        
#         # Apply higher weight to boundary cases
#         weighted_mse = torch.where(boundary_mask,
#                                  mse * self.boundary_weight,
#                                  mse)
        
#         return weighted_mse.mean()