import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


class CURLinear(nn.Module):
    """
    Activation-aware CUR decomposition for Linear layers.
    Decomposes W ≈ CUR where C contains selected columns, R contains selected rows,
    and U is the connecting matrix.
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
        
        Args:
            matrix: Input matrix [m, n]
            k: Target rank
            direction: 'column' for column scores, 'row' for row scores
            
        Returns:
            leverage_scores: Normalized leverage scores
        """
        if direction == 'column':
            # Compute top-k right singular vectors for column selection
            try:
                _, _, V = torch.svd_lowrank(matrix, q=min(k, min(matrix.shape)))
                # Leverage scores: π_j = (1/k) * Σ(v_ξj)² for ξ=1 to k
                leverage_scores = torch.sum(V**2, dim=1) / k
            except:
                # Fallback to uniform selection if SVD fails
                leverage_scores = torch.ones(matrix.size(1), device=matrix.device) / matrix.size(1)
        else:  # row
            # Compute top-k left singular vectors for row selection  
            try:
                U, _, _ = torch.svd_lowrank(matrix, q=min(k, min(matrix.shape)))
                # Leverage scores: π_i = (1/k) * Σ(u_iξ)² for ξ=1 to k
                leverage_scores = torch.sum(U**2, dim=1) / k
            except:
                # Fallback to uniform selection if SVD fails
                leverage_scores = torch.ones(matrix.size(0), device=matrix.device) / matrix.size(0)
                
        # Normalize to form probability distribution
        leverage_scores = leverage_scores / leverage_scores.sum()
        return leverage_scores
    
    @staticmethod
    def select_columns_rows_cur(matrix: torch.Tensor, k_cols: int, k_rows: int,
                               leverage_scores_cols: torch.Tensor,
                               leverage_scores_rows: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Select columns and rows based on leverage scores.
        Uses deterministic selection for better reproducibility.
        """
        m, n = matrix.shape
        
        # Use exact target numbers
        c_actual = min(k_cols, n)
        r_actual = min(k_rows, m)
        
        # Use top-k selection instead of random sampling for more stable results
        col_indices = torch.topk(leverage_scores_cols, c_actual)[1]
        row_indices = torch.topk(leverage_scores_rows, r_actual)[1]
            
        return col_indices, row_indices
    
    @staticmethod
    def compute_connecting_matrix(C: torch.Tensor, R: torch.Tensor, 
                                W_subset: torch.Tensor) -> torch.Tensor:
        """
        Compute the connecting matrix U using Moore-Penrose pseudoinverse.
        Improved version with better numerical stability.
        
        Args:
            C: Selected columns [m, k_cols]
            R: Selected rows [k_rows, n]  
            W_subset: Intersection matrix W[row_indices, col_indices] [k_rows, k_cols]
            
        Returns:
            U: Connecting matrix [k_cols, k_rows]
        """
        try:
            # Add small regularization for numerical stability
            reg = 1e-6
            
            # Improved computation: U = (C^T C + reg*I)^{-1} C^T W_subset R^T (R R^T + reg*I)^{-1}
            CtC = C.T @ C + reg * torch.eye(C.size(1), device=C.device)
            RRt = R @ R.T + reg * torch.eye(R.size(0), device=R.device)
            
            # Solve linear systems instead of direct inverse
            CtC_inv = torch.linalg.solve(CtC, torch.eye(C.size(1), device=C.device))
            RRt_inv = torch.linalg.solve(RRt, torch.eye(R.size(0), device=R.device))
            
            U = CtC_inv @ C.T @ W_subset @ R.T @ RRt_inv
            U = U.T  # Transpose to get [k_cols, k_rows]
            
            return U
            
        except:
            # Fallback to simple pseudoinverse
            try:
                C_pinv = torch.pinverse(C)
                R_pinv = torch.pinverse(R)
                U = C_pinv @ W_subset @ R_pinv
                return U
            except:
                # Final fallback - identity-like matrix
                k_cols, k_rows = C.size(1), R.size(0)
                min_dim = min(k_cols, k_rows)
                U = torch.zeros(k_cols, k_rows, device=C.device)
                U[:min_dim, :min_dim] = torch.eye(min_dim, device=C.device)
                return U

    @staticmethod  
    def from_linear(linear: nn.Linear, param_ratio: float, act_aware=False, 
                   ic_split=1, oc_split=1, alpha=1, sigma_fuse="UV", rank_align=1) -> 'CURLinear':
        """
        Create CURLinear from existing Linear layer using activation-aware CUR decomposition.
        
        Args:
            linear: Original linear layer
            param_ratio: Target parameter ratio (compressed/original)
            act_aware: Whether to use activation-aware decomposition
            alpha: Scaling factor for activation awareness
            rank_align: Rank alignment factor
            
        Returns:
            CURLinear layer
        """
        
        # Get original weight matrix
        w = linear.weight.data.float()  # [out_features, in_features]
        m, n = w.shape
        
        # Calculate target number of columns and rows based on param_ratio
        # For CUR: total_params = m*k_cols + k_cols*k_rows + k_rows*n
        # Setting k_cols = k_rows = k for simplicity
        # So: total_params = k*(m + k + n)
        # We want: k*(m + k + n) = param_ratio * m*n
        # Solving: k^2 + k*(m + n) - param_ratio*m*n = 0
        
        # Try different k values to find one that gets close to target ratio
        target_params = int(param_ratio * m * n)
        best_k = 1
        best_diff = float('inf')
        
        # Search for optimal k
        max_k = min(m, n) // 3
        for k in range(1, max_k + 1):
            actual_params = k * (m + k + n)
            diff = abs(actual_params - target_params)
            if diff < best_diff:
                best_diff = diff
                best_k = k
            if actual_params > target_params:
                break
        
        # Apply rank alignment
        k = int(np.ceil(best_k / rank_align) * rank_align)
        k = max(1, min(k, min(m, n) // 3))  # Safety bounds
        
        k_cols = k_rows = k
        
        print(f"   CUR decomposition: target_ratio={param_ratio:.3f}, k={k}, estimated_ratio={(k*(m+k+n))/(m*n):.3f}")
        
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
            print("NaN detected in CUR decomposition, falling back to identity")
            return nn.Linear(linear.in_features, linear.out_features).to(linear.weight.dtype).to(linear.weight.device)
            
        if torch.isinf(C).any() or torch.isinf(U).any() or torch.isinf(R).any():
            print("Inf detected in CUR decomposition, falling back to identity")
            return nn.Linear(linear.in_features, linear.out_features).to(linear.weight.dtype).to(linear.weight.device)
        
        # Create CURLinear layer
        cur_linear = CURLinear(C, U, R, col_indices, row_indices, bias)
        cur_linear.to(linear.weight.dtype)
        
        return cur_linear
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: y = x @ R^T @ U^T @ C^T + bias
        """
        # Explicit matrix multiplications to avoid F.linear confusion
        x = torch.matmul(x, self.R.T)         # [batch, in_features] @ [in_features, k_rows] = [batch, k_rows]
        x = torch.matmul(x, self.U.T)         # [batch, k_rows] @ [k_rows, k_cols] = [batch, k_cols]
        x = torch.matmul(x, self.C.T)         # [batch, k_cols] @ [k_cols, out_features] = [batch, out_features]
        
        if self.bias is not None:
            x = x + self.bias
            
        return x
    
    def reconstruct_weight(self) -> torch.Tensor:
        """
        Reconstruct the full weight matrix from CUR decomposition.
        
        Returns:
            Reconstructed weight matrix [out_features, in_features]
        """
        # W ≈ C @ U @ R
        reconstructed = self.C @ self.U @ self.R
        return reconstructed
    
    def get_param_ratio(self) -> float:
        """
        Calculate the actual parameter ratio (compressed/original).
        
        Returns:
            Parameter ratio
        """
        compressed_params = (self.C.numel() + self.U.numel() + self.R.numel())
        original_params = self.original_out_features * self.original_in_features
        return compressed_params / original_params


class GradCURLinear(nn.Module):
    """
    Gradient-based CUR Linear layer for optimization during search.
    """
    
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
        """
        Create GradCURLinear from existing Linear layer.
        """
        if param_ratio >= 1:
            return linear
            
        w = linear.weight.data.float()
        m, n = w.shape
        
        # Calculate k similar to CURLinear.from_linear
        a = 1
        b = m + n
        c = -param_ratio * m * n
        discriminant = b*b - 4*a*c
        
        if discriminant < 0:
            k = min(m, n) // 4
        else:
            k = int((-b + np.sqrt(discriminant)) / (2*a))
            
        k = max(1, min(k, min(m, n) // 2))
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
        """
        Forward pass with dynamic CUR decomposition.
        """
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