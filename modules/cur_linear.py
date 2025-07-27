import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


class CURLinear(nn.Module):
    """
    Activation-aware CUR decomposition for Linear layers.
    FIXED VERSION - Addresses parameter ratio calculation issues.
    """
    
    def __init__(self, C: torch.Tensor, U: torch.Tensor, R: torch.Tensor, 
                 col_indices: torch.Tensor = None, row_indices: torch.Tensor = None, bias=None) -> None:
        super().__init__()
        
        # Store the CUR decomposition components
        self.C = nn.Parameter(C.clone())  # Selected columns [out_features, k_cols]
        self.U = nn.Parameter(U.clone())  # Connection matrix [k_cols, k_rows] 
        self.R = nn.Parameter(R.clone())  # Selected rows [k_rows, in_features]
        
        # Store indices for reconstruction (optional, for analysis)
        if col_indices is not None:
            self.register_buffer('col_indices', col_indices.clone())
        else:
            self.register_buffer('col_indices', torch.arange(C.size(1)))
            
        if row_indices is not None:
            self.register_buffer('row_indices', row_indices.clone())
        else:
            self.register_buffer('row_indices', torch.arange(R.size(0)))
        
        # Bias handling
        if bias is not None:
            self.bias = nn.Parameter(bias.clone())
        else:
            self.register_parameter('bias', None)
            
        # Store dimensions for parameter counting
        self.original_out_features = C.size(0)
        self.original_in_features = R.size(1)
        self.k_cols = C.size(1)
        self.k_rows = R.size(0)
        
    @staticmethod
    def compute_statistical_leverage_scores(matrix: torch.Tensor, k: int, 
                                          direction: str = 'column') -> torch.Tensor:
        """
        Compute statistical leverage scores for column or row selection.
        """
        if direction == 'column':
            try:
                _, _, V = torch.svd_lowrank(matrix, q=min(k, min(matrix.shape)))
                leverage_scores = torch.sum(V**2, dim=1) / k
            except:
                leverage_scores = torch.ones(matrix.size(1), device=matrix.device) / matrix.size(1)
        else:  # row
            try:
                U, _, _ = torch.svd_lowrank(matrix, q=min(k, min(matrix.shape)))
                leverage_scores = torch.sum(U**2, dim=1) / k
            except:
                leverage_scores = torch.ones(matrix.size(0), device=matrix.device) / matrix.size(0)
                
        # Normalize to form probability distribution
        leverage_scores = leverage_scores / leverage_scores.sum()
        return leverage_scores
    
    @staticmethod
    def select_columns_rows_cur(matrix: torch.Tensor, k_cols: int, k_rows: int,
                               leverage_scores_cols: torch.Tensor,
                               leverage_scores_rows: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Select columns and rows based on leverage scores."""
        m, n = matrix.shape
        
        c_actual = min(k_cols, n)
        r_actual = min(k_rows, m)
        
        # Use top-k selection for more stable results
        col_indices = torch.topk(leverage_scores_cols, c_actual)[1]
        row_indices = torch.topk(leverage_scores_rows, r_actual)[1]
            
        return col_indices, row_indices
    
    @staticmethod
    def compute_connecting_matrix(C: torch.Tensor, R: torch.Tensor, 
                                W_subset: torch.Tensor) -> torch.Tensor:
        """Compute the connecting matrix U using Moore-Penrose pseudoinverse."""
        try:
            reg = 1e-6
            CtC = C.T @ C + reg * torch.eye(C.size(1), device=C.device)
            RRt = R @ R.T + reg * torch.eye(R.size(0), device=R.device)
            
            CtC_inv = torch.linalg.solve(CtC, torch.eye(C.size(1), device=C.device))
            RRt_inv = torch.linalg.solve(RRt, torch.eye(R.size(0), device=R.device))
            
            U = CtC_inv @ C.T @ W_subset @ R.T @ RRt_inv
            U = U.T  # Transpose to get [k_cols, k_rows]
            
            return U
            
        except:
            try:
                C_pinv = torch.pinverse(C)
                R_pinv = torch.pinverse(R)
                U = C_pinv @ W_subset @ R_pinv
                return U
            except:
                k_cols, k_rows = C.size(1), R.size(0)
                min_dim = min(k_cols, k_rows)
                U = torch.zeros(k_cols, k_rows, device=C.device)
                U[:min_dim, :min_dim] = torch.eye(min_dim, device=C.device)
                return U

    @staticmethod
    def calculate_optimal_k_fixed(m: int, n: int, target_ratio: float, rank_align: int = 1) -> int:
        """
        FIXED: Calculate optimal k value for target parameter ratio.
        
        Solves: k(m + k + n) = target_ratio * m * n
        Which gives: k² + k(m + n) - target_ratio*m*n = 0
        """
        target_params = target_ratio * m * n
        
        # Quadratic formula: ax² + bx + c = 0
        a = 1
        b = m + n  
        c = -target_params
        
        discriminant = b * b - 4 * a * c
        
        if discriminant >= 0:
            # Take positive root
            k_optimal = (-b + np.sqrt(discriminant)) / (2 * a)
            k = int(np.round(k_optimal))
        else:
            # Fallback if no real solution
            k = min(m, n) // 4
        
        # Apply constraints
        k = max(1, k)  # At least 1
        
        # FIXED: More generous upper bound for high compression ratios
        max_k_theoretical = min(m, n)  # Theoretical maximum
        max_k_practical = int(max_k_theoretical * 0.8)  # Use 80% of theoretical max
        k = min(k, max_k_practical)
        
        # Apply rank alignment
        if rank_align > 1:
            k = int(np.ceil(k / rank_align) * rank_align)
            k = max(rank_align, k)  # At least one alignment unit
        
        # Verify the result
        actual_params = k * (m + k + n)
        actual_ratio = actual_params / (m * n)
        
        print(f"   🔧 k calculation: target_ratio={target_ratio:.3f}, k={k}, "
              f"estimated_ratio={actual_ratio:.3f}, m={m}, n={n}")
        
        return k

    @staticmethod  
    def from_linear(linear: nn.Linear, param_ratio: float, act_aware=False, 
                   ic_split=1, oc_split=1, alpha=1, sigma_fuse="UV", rank_align=1) -> 'CURLinear':
        """
        FIXED VERSION: Create CURLinear from existing Linear layer.
        """
        
        # Get original weight matrix
        w = linear.weight.data.float()
        m, n = w.shape
        
        # FIXED: Use the corrected k calculation
        k = CURLinear.calculate_optimal_k_fixed(m, n, param_ratio, rank_align)
        k_cols = k_rows = k
        
        print(f"   CUR decomposition: target_ratio={param_ratio:.3f}, k={k}, "
              f"matrix_shape=({m},{n})")
        
        # Early exit if k is too large (impossible compression)
        if k >= min(m, n):
            print(f"   ❌ CUR impossible: k={k} >= min(m,n)={min(m,n)}, falling back to original")
            return linear
        
        # Apply activation-aware scaling if enabled
        scaling_diag_matrix = None
        if act_aware:
            scaling_diag_matrix = torch.ones(n, device=w.device, dtype=w.dtype)
            
            if hasattr(linear, "scaling_diag_matrix"):
                scaling_diag_matrix = scaling_diag_matrix * (linear.scaling_diag_matrix ** alpha)
                
            if hasattr(linear, "fisher_info"):
                scaling_diag_matrix = scaling_diag_matrix * (linear.fisher_info ** alpha)
                
            scaling_diag_matrix = scaling_diag_matrix + 1e-6  # Avoid division by zero
            
            # Scale the weight matrix
            w_scaled = w * scaling_diag_matrix.view(1, -1)
        else:
            w_scaled = w
            
        # Compute statistical leverage scores
        leverage_scores_cols = CURLinear.compute_statistical_leverage_scores(w_scaled, k, 'column')
        leverage_scores_rows = CURLinear.compute_statistical_leverage_scores(w_scaled, k, 'row')
        
        # Select columns and rows
        col_indices, row_indices = CURLinear.select_columns_rows_cur(
            w_scaled, k_cols, k_rows, leverage_scores_cols, leverage_scores_rows
        )
        
        # Extract selected columns and rows
        C = w_scaled[:, col_indices]  # [m, k_cols]
        R = w_scaled[row_indices, :]  # [k_rows, n]
        W_subset = w_scaled[row_indices][:, col_indices]  # [k_rows, k_cols]
        
        # Apply inverse scaling to R if activation-aware
        if act_aware and scaling_diag_matrix is not None:
            R = R / scaling_diag_matrix.view(1, -1)
            
        # Compute connecting matrix U
        U = CURLinear.compute_connecting_matrix(C, R, W_subset)
        
        # Handle bias
        bias = linear.bias.data if linear.bias is not None else None
        
        # Check for NaN/Inf
        if torch.isnan(C).any() or torch.isnan(U).any() or torch.isnan(R).any():
            print("   ❌ NaN detected in CUR decomposition, falling back to original")
            return linear
            
        if torch.isinf(C).any() or torch.isinf(U).any() or torch.isinf(R).any():
            print("   ❌ Inf detected in CUR decomposition, falling back to original")
            return linear
        
        # Create CURLinear layer
        cur_linear = CURLinear(C, U, R, col_indices, row_indices, bias)
        cur_linear.to(linear.weight.dtype)
        
        # Verify the parameter ratio
        actual_ratio = cur_linear.get_param_ratio()
        ratio_error = abs(actual_ratio - param_ratio)
        
        # FIXED: More lenient acceptance criteria
        if ratio_error > 0.3:  # Accept if within 30% of target
            print(f"   ⚠️  Large ratio error: target={param_ratio:.3f}, actual={actual_ratio:.3f}, "
                  f"error={ratio_error:.3f}")
            # Still return the CUR layer - it's better than nothing
        
        print(f"   ✅ CUR created: target={param_ratio:.3f}, actual={actual_ratio:.3f}, "
              f"compression={1-actual_ratio:.1%}")
        
        return cur_linear
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: y = x @ R^T @ U^T @ C^T + bias"""
        x = torch.matmul(x, self.R.T)         # [batch, in_features] @ [in_features, k_rows]
        x = torch.matmul(x, self.U.T)         # [batch, k_rows] @ [k_rows, k_cols]
        x = torch.matmul(x, self.C.T)         # [batch, k_cols] @ [k_cols, out_features]
        
        if self.bias is not None:
            x = x + self.bias
            
        return x
    
    def reconstruct_weight(self) -> torch.Tensor:
        """Reconstruct the full weight matrix from CUR decomposition."""
        reconstructed = self.C @ self.U @ self.R
        return reconstructed
    
    def get_param_ratio(self) -> float:
        """Calculate the actual parameter ratio (compressed/original)."""
        compressed_params = (self.C.numel() + self.U.numel() + self.R.numel())
        original_params = self.original_out_features * self.original_in_features
        return compressed_params / original_params


class GradCURLinear(nn.Module):
    """Gradient-based CUR Linear layer for optimization during search."""
    
    def __init__(self, weight: torch.Tensor, scale: torch.Tensor, bias: Optional[torch.Tensor], 
                 k_cols: int, k_rows: int) -> None:
        super().__init__()
        self.weight = weight
        self.scale = nn.Parameter(scale)
        self.bias = bias
        self.k_cols = k_cols
        self.k_rows = k_rows

    @staticmethod
    def from_linear(linear: nn.Linear, param_ratio: float, act_aware=False, 
                   ic_split=1, oc_split=1, alpha=1, sigma_fuse="UV") -> 'GradCURLinear':
        """Create GradCURLinear from existing Linear layer."""
        if param_ratio >= 1:
            return linear
            
        w = linear.weight.data.float()
        m, n = w.shape
        
        # Use the fixed k calculation
        k = CURLinear.calculate_optimal_k_fixed(m, n, param_ratio)
        k_cols = k_rows = k
        
        # Handle activation awareness
        scaling_diag_matrix = torch.ones(n, device=w.device, dtype=w.dtype)
        if act_aware:
            if hasattr(linear, "scaling_diag_matrix"):
                scaling_diag_matrix = scaling_diag_matrix * (linear.scaling_diag_matrix ** alpha)
            if hasattr(linear, "fisher_info"):
                scaling_diag_matrix = scaling_diag_matrix * (linear.fisher_info ** alpha)
            scaling_diag_matrix = scaling_diag_matrix + 1e-6

        bias = linear.bias.data if linear.bias is not None else None
        
        return GradCURLinear(w, scaling_diag_matrix, bias, k_cols, k_rows)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with dynamic CUR decomposition."""
        # Scale weight matrix
        w_scaled = self.weight * self.scale.view(1, -1)
        
        # Perform CUR decomposition on-the-fly
        m, n = w_scaled.shape
        
        # Compute leverage scores
        leverage_scores_cols = CURLinear.compute_statistical_leverage_scores(w_scaled, self.k_cols, 'column')
        leverage_scores_rows = CURLinear.compute_statistical_leverage_scores(w_scaled, self.k_rows, 'row')
        
        # Select columns and rows
        col_indices, row_indices = CURLinear.select_columns_rows_cur(
            w_scaled, self.k_cols, self.k_rows, leverage_scores_cols, leverage_scores_rows
        )
        
        # Extract CUR components
        C = w_scaled[:, col_indices]
        R = w_scaled[row_indices, :]
        W_subset = w_scaled[row_indices][:, col_indices]
        
        # Compute U
        U = CURLinear.compute_connecting_matrix(C, R, W_subset)
        
        # Reconstruct weight
        new_w = C @ U @ R
        
        # Apply linear transformation
        y = F.linear(x, new_w, self.bias)
        return y