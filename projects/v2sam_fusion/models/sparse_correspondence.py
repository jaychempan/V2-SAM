"""
Sparse Correspondence Module using DINOv3 features
Extracts and matches features between ego and exo views
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import numpy as np
from typing import Tuple, Optional, List
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
import os


class SparseCorrespondenceMatcher(nn.Module):
    """
    A module for establishing sparse correspondences between two images using DINOv3 features.
    
    Args:
        dinov3_model: Pre-trained DINOv3 model
        patch_size: Size of patches (default: 16)
        image_size: Target image size for processing (default: 768)
        mask_fg_threshold: Threshold for foreground mask filtering (default: 0.6)
        stratify_distance_threshold: Distance threshold for point stratification (default: 150.0)
        n_layers: Number of layers in DINOv3 model (default: 24 for ViT-L)
    """
    
    def __init__(
        self,
        dinov3_model: nn.Module,
        patch_size: int = 16,
        image_size: int = 768,
        mask_fg_threshold: float = 0.8, # debug: new feat
        stratify_distance_threshold: float = 180.0, # debug: new feat  
        n_layers: int = 24,
        max_points_per_object: Optional[int] = 1,  # debug: new feat,每个物体最多返回的点数
        outlier_removal_ratio: float = 0.25,  # debug: new feat,离群点移除比例（移除最远的25%）
    ):
        super().__init__()
        
        self.dinov3_model = dinov3_model
        self.patch_size = patch_size
        self.image_size = image_size
        self.mask_fg_threshold = mask_fg_threshold
        self.stratify_distance_threshold = stratify_distance_threshold
        self.n_layers = n_layers
        self.max_points_per_object = max_points_per_object
        self.outlier_removal_ratio = outlier_removal_ratio
        
        # ImageNet normalization constants
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1))
        
        # Patch quantization filter (box blur)
        # 用来将mask打成patches，并且计算每个patch的前景比例
        patch_quant_filter = nn.Conv2d(1, 1, patch_size, stride=patch_size, bias=False)
        patch_quant_filter.weight.data.fill_(1.0 / (patch_size * patch_size))
        patch_quant_filter.requires_grad_(False)
        self.patch_quant_filter = patch_quant_filter
    
    def _resize_transform(
        self,
        input_data,  # PIL Image or numpy array or torch.Tensor
    ) -> torch.Tensor:
        """
        Resize image/mask to dimensions divisible by patch size.
        
        Args:
            input_data: PIL Image, numpy array, or torch.Tensor
            
        Returns:
            Resized tensor [C, H, W]
        """
        # Convert to PIL Image if needed
        if isinstance(input_data, np.ndarray):
            if input_data.ndim == 2:  # Binary mask
                input_data = Image.fromarray((input_data * 255).astype(np.uint8))
            else:  # RGB image
                input_data = Image.fromarray(input_data)
        elif isinstance(input_data, torch.Tensor):
            # Assume already in correct format
            if input_data.ndim == 2:
                input_data = Image.fromarray((input_data.cpu().numpy() * 255).astype(np.uint8))
            else:
                input_data = TF.to_pil_image(input_data.cpu())
        
        w, h = input_data.size
        h_patches = int(self.image_size / self.patch_size)
        w_patches = int((w * self.image_size) / (h * self.patch_size))
        
        target_h = h_patches * self.patch_size
        target_w = w_patches * self.patch_size
        
        return TF.to_tensor(TF.resize(input_data, (target_h, target_w)))
    
    def _extract_features(
        self,
        image: torch.Tensor,
    ) -> torch.Tensor:
        """
        Extract DINOv3 features from an image.
        
        Args:
            image: Image tensor [1, 3, H, W] (already normalized)
            
        Returns:
            Feature tensor [D, H_patches, W_patches]
        """
        with torch.inference_mode():
            feats = self.dinov3_model.get_intermediate_layers(
                image, n=range(self.n_layers), reshape=True, norm=True
            )
            return feats[-1].squeeze(0)  # [D, H, W]
    
    def _compute_distances_l2(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
        X_squared_norm: torch.Tensor,
        Y_squared_norm: torch.Tensor,
    ) -> torch.Tensor:
        """Compute pairwise L2 distances."""
        distances = -2 * X @ Y.T
        distances.add_(X_squared_norm[:, None]).add_(Y_squared_norm[None, :])
        return distances
    
    def _stratify_points(
        self,
        pts_2d: torch.Tensor,
        threshold: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Stratify points to avoid clustering.
        
        Args:
            pts_2d: Points tensor [N, 2]
            threshold: Distance threshold (squared)
            
        Returns:
            Tuple of (indices_to_exclude, indices_to_keep)
        """
        n = len(pts_2d)
        max_value = threshold + 1
        
        pts_2d_sq_norms = torch.linalg.vector_norm(pts_2d, dim=1)
        pts_2d_sq_norms.square_()
        
        distances = self._compute_distances_l2(
            pts_2d, pts_2d, pts_2d_sq_norms, pts_2d_sq_norms
        )
        distances.fill_diagonal_(max_value)
        
        distances_mask = torch.le(distances, threshold)
        ones_vec = torch.ones(n, device=pts_2d.device, dtype=pts_2d.dtype)
        counts_vec = torch.mv(distances_mask.float(), ones_vec)
        
        indices_mask = np.ones(n)
        while torch.any(counts_vec).item():
            index_max = torch.argmax(counts_vec).item()
            indices_mask[index_max] = 0
            distances[index_max, :] = max_value
            distances[:, index_max] = max_value
            distances_mask = torch.le(distances, threshold)
            counts_vec = torch.mv(distances_mask.float(), ones_vec)
        
        indices_to_exclude = np.nonzero(indices_mask == 0)[0]
        indices_to_keep = np.nonzero(indices_mask > 0)[0]
        
        return indices_to_exclude, indices_to_keep
    
    # def forward(
    #     self,
    #     image_ego,  # PIL Image, numpy array, or torch.Tensor [H, W, 3]
    #     image_exo,  # PIL Image, numpy array, or torch.Tensor [H, W, 3]
    #     mask_ego: Optional[np.ndarray] = None,  # Binary mask [H, W] or [n_obj, H, W]
    #     mask_exo: Optional[np.ndarray] = None,  # Binary mask [H, W]
    #     return_features: bool = False,
    # ) -> dict:
    #     """
    #     Establish sparse correspondences between ego and exo images.
        
    #     Args:
    #         image_ego: Ego-view image
    #         image_exo: Exo-view image
    #         mask_ego: Binary mask for ego image (optional, if None, uses full image)
    #                  - Can be [H, W] for single object or [n_obj, H, W] for multiple objects
    #         mask_exo: Binary mask for exo image (optional, if None, uses full image)
    #         return_features: Whether to return intermediate features
            
    #     Returns:
    #         Dictionary containing:
    #             - 'points_ego': Matched points in ego image
    #                 * If mask_ego is [H, W]: [N, 2] (x, y) in original image coordinates
    #                 * If mask_ego is [n_obj, H, W]: [n_obj, N, 2] - may have different N per object
    #             - 'points_exo': Matched points in exo image (same shape as points_ego)
    #             - 'num_matches': Number of matches
    #                 * If single object: int
    #                 * If multiple objects: list of ints [n_obj]
    #             - 'features_ego': (optional) Feature maps for ego image
    #             - 'features_exo': (optional) Feature maps for exo image
    #     """
    #     # Convert images to PIL if needed for size information
    #     if isinstance(image_ego, torch.Tensor):
    #         h_ego, w_ego = image_ego.shape[-2:]
    #     elif isinstance(image_ego, np.ndarray):
    #         h_ego, w_ego = image_ego.shape[:2]
    #     else:  # PIL Image
    #         w_ego, h_ego = image_ego.size
        
    #     if isinstance(image_exo, torch.Tensor):
    #         h_exo, w_exo = image_exo.shape[-2:]
    #     elif isinstance(image_exo, np.ndarray):
    #         h_exo, w_exo = image_exo.shape[:2]
    #     else:  # PIL Image
    #         w_exo, h_exo = image_exo.size
        
    #     # Check if mask_ego is multi-object or single-object
    #     is_multi_object = False
    #     n_objects = 1
        
    #     if mask_ego is not None:
    #         if isinstance(mask_ego, torch.Tensor):
    #             mask_ego = mask_ego.cpu().numpy()
            
    #         if mask_ego.ndim == 3:  # Multi-object: [n_obj, H, W]
    #             is_multi_object = True
    #             n_objects = mask_ego.shape[0]
    #         elif mask_ego.ndim == 2:  # Single object: [H, W]
    #             # Convert to [1, H, W] for uniform processing
    #             mask_ego = mask_ego[np.newaxis, ...]
    #             n_objects = 1
    #         else:
    #             raise ValueError(f"mask_ego must be 2D [H, W] or 3D [n_obj, H, W], got shape {mask_ego.shape}")
    #     else:
    #         # Create default mask for full image
    #         mask_ego = np.ones((1, h_ego, w_ego), dtype=np.uint8)
    #         n_objects = 1
        
    #     # Create default masks if not provided
    #     if mask_exo is None:
    #         mask_exo = np.ones((h_exo, w_exo), dtype=np.uint8)
    #     elif isinstance(mask_exo, torch.Tensor):
    #         mask_exo = mask_exo.cpu().numpy()
        
    #     # Process images (only once)
    #     image_ego_tensor = self._resize_transform(image_ego).to(self.mean.device)
    #     image_exo_tensor = self._resize_transform(image_exo).to(self.mean.device)
        
    #     # Normalize
    #     image_ego_normalized = (image_ego_tensor - self.mean) / self.std
    #     image_exo_normalized = (image_exo_tensor - self.mean) / self.std
        
    #     # Extract features (only once)
    #     features_ego = self._extract_features(image_ego_normalized.unsqueeze(0))
    #     features_exo = self._extract_features(image_exo_normalized.unsqueeze(0))
        
    #     dim = features_ego.shape[0]
        
    #     # Normalize features
    #     features_ego = F.normalize(features_ego, p=2, dim=0)
    #     features_exo = F.normalize(features_exo, p=2, dim=0)
        
    #     # Process exo mask (only once)
    #     mask_exo_resized = self._resize_transform(mask_exo).to(self.mean.device)
    #     mask_exo_quantized = self.patch_quant_filter(
    #         mask_exo_resized.unsqueeze(0)
    #     ).squeeze(0).squeeze(0)
        
    #     # Compute similarity heatmaps (only once)
    #     heatmaps = torch.einsum(
    #         "k f, f h w -> k h w",
    #         features_ego.view(dim, -1).permute(1, 0),
    #         features_exo,
    #     )
        
    #     # Compute 2D patch locations in ego image (only once)
    #     n_patches_ego = features_ego.shape[1] * features_ego.shape[2]
    #     patch_indices_ego = torch.arange(n_patches_ego, device=features_ego.device)
    #     locs_2d_ego = (
    #         torch.stack(
    #             (
    #                 patch_indices_ego // features_ego.shape[2],  # row
    #                 patch_indices_ego % features_ego.shape[2]    # column
    #             ),
    #             dim=-1
    #         ) + 0.5
    #     ) * self.patch_size
        
    #     # Compute corresponding 2D patch locations in exo image (only once)
    #     patch_indices_exo = torch.flatten(heatmaps, start_dim=-2).argmax(dim=-1)
    #     locs_2d_exo = (
    #         torch.stack(
    #             (
    #                 patch_indices_exo // features_exo.shape[2],  # row
    #                 patch_indices_exo % features_exo.shape[2]    # column
    #             ),
    #             dim=-1
    #         ) + 0.5
    #     ) * self.patch_size
        
    #     # Process each object's mask and compute correspondences
    #     all_points_ego = []
    #     all_points_exo = []
    #     num_matches_list = []
        
    #     for obj_idx in range(n_objects):
    #         # Get current object's mask
    #         current_mask_ego = mask_ego[obj_idx]  # [H, W]
            
    #         # Process ego mask for current object
    #         mask_ego_resized = self._resize_transform(current_mask_ego).to(self.mean.device)
    #         mask_ego_quantized = self.patch_quant_filter(
    #             mask_ego_resized.unsqueeze(0)
    #         ).squeeze(0).squeeze(0)
            
    #         # Foreground selection for current object
    #         patches_ego_fg_selection = (
    #             mask_ego_quantized.view(-1) > self.mask_fg_threshold
    #         )
    #         patches_exo_fg_selection = (
    #             mask_exo_quantized.view(-1)[patch_indices_exo] > self.mask_fg_threshold
    #         )
    #         patches_fg_selection = patches_ego_fg_selection * patches_exo_fg_selection
            
    #         # Select foreground matched patches
    #         locs_2d_ego_fg = locs_2d_ego[patches_fg_selection, :]
    #         locs_2d_exo_fg = locs_2d_exo[patches_fg_selection, :]
            
    #         # Compute image scales
    #         scale_ego = h_ego / self.image_size
    #         scale_exo = h_exo / self.image_size
            
    #         # Stratify points
    #         if len(locs_2d_ego_fg) > 0:
    #             _, indices_to_keep = self._stratify_points(
    #                 locs_2d_ego_fg * scale_ego,
    #                 self.stratify_distance_threshold ** 2
    #             )
                
    #             # Get final sparse points in original image coordinates
    #             sparse_points_ego_yx = locs_2d_ego_fg[indices_to_keep, :].cpu().numpy() * scale_ego
    #             sparse_points_exo_yx = locs_2d_exo_fg[indices_to_keep, :].cpu().numpy() * scale_exo

    #             # debug: 新增加功能
    #             # === 离群点过滤 + 随机采样 ===
    #             if len(sparse_points_ego_yx) > 0:
    #                 # 1. 移除离群点：计算每个点到所有其他点的平均距离
    #                 points_array = sparse_points_ego_yx  # [N, 2]
    #                 n_points = len(points_array)
                    
    #                 if n_points > 3:  # 只有点数足够多时才进行离群点过滤
    #                     # 计算两两距离
    #                     distances = np.linalg.norm(
    #                         points_array[:, None, :] - points_array[None, :, :], 
    #                         axis=2
    #                     )  # [N, N]
                        
    #                     # 计算每个点到其他点的平均距离
    #                     avg_distances = distances.sum(axis=1) / (n_points - 1)  # [N]
                        
    #                     # 根据离群点移除比例，移除距离最大的点
    #                     n_remove = int(n_points * self.outlier_removal_ratio)
    #                     if n_remove > 0:
    #                         # 按平均距离排序，保留距离较小的点
    #                         sorted_indices = np.argsort(avg_distances)
    #                         inlier_indices = sorted_indices[:n_points - n_remove]
                            
    #                         sparse_points_ego_yx = sparse_points_ego_yx[inlier_indices]
    #                         sparse_points_exo_yx = sparse_points_exo_yx[inlier_indices]
                    
    #                 # 2. 随机采样到固定数量
    #                 if self.max_points_per_object is not None and len(sparse_points_ego_yx) > self.max_points_per_object:
    #                     # 随机采样
    #                     selected_indices = np.random.choice(
    #                         len(sparse_points_ego_yx), 
    #                         self.max_points_per_object, 
    #                         replace=False
    #                     )
    #                     sparse_points_ego_yx = sparse_points_ego_yx[selected_indices]
    #                     sparse_points_exo_yx = sparse_points_exo_yx[selected_indices]
    #                 elif self.max_points_per_object is not None and len(sparse_points_ego_yx) < self.max_points_per_object:
    #                     # 如果点数不足，进行上采样（重复采样）
    #                     selected_indices = np.random.choice(
    #                         len(sparse_points_ego_yx), 
    #                         self.max_points_per_object, 
    #                         replace=True  # 允许重复
    #                     )
    #                     sparse_points_ego_yx = sparse_points_ego_yx[selected_indices]
    #                     sparse_points_exo_yx = sparse_points_exo_yx[selected_indices]
    #             # === 结束 ===
                
    #             # Convert from (row, col) to (x, y) format
    #             points_ego = sparse_points_ego_yx[:, [1, 0]]  # [N, 2] (x, y)
    #             points_exo = sparse_points_exo_yx[:, [1, 0]]  # [N, 2] (x, y)
    #         else:
    #             # No valid points for this object
    #             points_ego = np.zeros((0, 2), dtype=np.float32)
    #             points_exo = np.zeros((0, 2), dtype=np.float32)
            
    #         all_points_ego.append(points_ego)
    #         all_points_exo.append(points_exo)
    #         num_matches_list.append(len(points_ego))
        
    #     # Format output based on whether multi-object or single-object
    #     result = {
    #         'points_ego': all_points_ego,  # list of [N_i, 2] arrays
    #         'points_exo': all_points_exo,  # list of [N_i, 2] arrays
    #         'num_matches': num_matches_list,  # list of ints
    #     }
    
        
    #     if return_features:
    #         result['features_ego'] = features_ego
    #         result['features_exo'] = features_exo
        
    #     return result

    def forward(
        self,
        image_ego,
        image_exo,
        mask_ego: Optional = None,
        mask_exo: Optional = None,
        return_features: bool = False,
    ) -> dict:
        """
        支持：
          - 单对输入 (单图像、单mask)
          - 多对输入 (list/tuple，每对可能有多个mask)
        """

        # ========== ✅ Step 1. 递归支持 list 输入 ==========
        if isinstance(image_ego, (list, tuple)):
            n_pairs = len(image_ego)
            # 检查输入长度一致
            assert len(image_exo) == n_pairs, f"image_exo len {len(image_exo)} != image_ego len {n_pairs}"
            if mask_ego is not None:
                assert len(mask_ego) == n_pairs, f"mask_ego len {len(mask_ego)} != image_ego len {n_pairs}"
            if mask_exo is not None:
                assert len(mask_exo) == n_pairs, f"mask_exo len {len(mask_exo)} != image_ego len {n_pairs}"

            results = []
            for i in range(n_pairs):
                res = self.forward(
                    image_ego=image_ego[i],
                    image_exo=image_exo[i],
                    mask_ego=None if mask_ego is None else mask_ego[i],
                    mask_exo=None if mask_exo is None else mask_exo[i],
                    return_features=return_features,
                )
                results.append(res)

            # 合并输出结构
            merged = {
                "points_ego": [r["points_ego"] for r in results],
                "points_exo": [r["points_exo"] for r in results],
                "num_matches": [r["num_matches"] for r in results],
            }
            if return_features:
                merged["features_ego"] = [r["features_ego"] for r in results]
                merged["features_exo"] = [r["features_exo"] for r in results]
            return merged
        # ========== ✅ Step 1 END ==========

        # ========== ✅ Step 2. 单对输入 (保持原逻辑) ==========
        # --- 处理图像尺寸 ---
        if isinstance(image_ego, torch.Tensor):
            h_ego, w_ego = image_ego.shape[-2:]
        elif isinstance(image_ego, np.ndarray):
            h_ego, w_ego = image_ego.shape[:2]
        else:
            w_ego, h_ego = image_ego.size

        if isinstance(image_exo, torch.Tensor):
            h_exo, w_exo = image_exo.shape[-2:]
        elif isinstance(image_exo, np.ndarray):
            h_exo, w_exo = image_exo.shape[:2]
        else:
            w_exo, h_exo = image_exo.size

        # --- 处理 mask_ego (可能多个) ---
        if mask_ego is not None:
            if isinstance(mask_ego, torch.Tensor):
                mask_ego = mask_ego.cpu().numpy()
            if mask_ego.ndim == 2:  # 单个
                mask_ego = mask_ego[np.newaxis, ...]
            elif mask_ego.ndim == 3:
                pass  # [n_obj, H, W]
            else:
                raise ValueError(f"mask_ego must be [H,W] or [n_obj,H,W]")
        else:
            mask_ego = np.ones((1, h_ego, w_ego), dtype=np.uint8)

        # --- 处理 mask_exo ---
        if mask_exo is not None:
            if isinstance(mask_exo, torch.Tensor):
                mask_exo = mask_exo.cpu().numpy()
            if mask_exo.ndim == 2:
                pass  # 单个
            elif mask_exo.ndim == 3:
                # 如果 exo mask 多个，只取并集（或可以自定义逻辑）
                mask_exo = np.any(mask_exo, axis=0).astype(np.uint8)
            else:
                raise ValueError(f"mask_exo must be [H,W] or [n_obj,H,W]")
        else:
            mask_exo = np.ones((h_exo, w_exo), dtype=np.uint8)

        # --- 图像预处理 & 特征提取 ---
        image_ego_tensor = self._resize_transform(image_ego).to(self.mean.device)
        image_exo_tensor = self._resize_transform(image_exo).to(self.mean.device)
        image_ego_normalized = (image_ego_tensor - self.mean) / self.std
        image_exo_normalized = (image_exo_tensor - self.mean) / self.std

        features_ego = self._extract_features(image_ego_normalized.unsqueeze(0))
        features_exo = self._extract_features(image_exo_normalized.unsqueeze(0))

        dim = features_ego.shape[0]
        features_ego = F.normalize(features_ego, p=2, dim=0)
        features_exo = F.normalize(features_exo, p=2, dim=0)

        # --- 预计算 exo patch mask & 相似度 ---
        mask_exo_resized = self._resize_transform(mask_exo).to(self.mean.device)
        mask_exo_quantized = self.patch_quant_filter(mask_exo_resized.unsqueeze(0)).squeeze(0).squeeze(0)
        heatmaps = torch.einsum(
            "k f, f h w -> k h w",
            features_ego.view(dim, -1).permute(1, 0),
            features_exo,
        )

        n_patches_ego = features_ego.shape[1] * features_ego.shape[2]
        patch_indices_ego = torch.arange(n_patches_ego, device=features_ego.device)
        locs_2d_ego = (
            torch.stack(
                (
                    patch_indices_ego // features_ego.shape[2],
                    patch_indices_ego % features_ego.shape[2],
                ),
                dim=-1,
            )
            + 0.5
        ) * self.patch_size

        patch_indices_exo = torch.flatten(heatmaps, start_dim=-2).argmax(dim=-1)
        locs_2d_exo = (
            torch.stack(
                (
                    patch_indices_exo // features_exo.shape[2],
                    patch_indices_exo % features_exo.shape[2],
                ),
                dim=-1,
            )
            + 0.5
        ) * self.patch_size

        # --- 遍历 ego 中的每个对象 mask ---
        all_points_ego, all_points_exo, num_matches_list = [], [], []
        scale_ego = h_ego / self.image_size
        scale_exo = h_exo / self.image_size

        for obj_idx in range(mask_ego.shape[0]):
            cur_mask = mask_ego[obj_idx]

            mask_ego_resized = self._resize_transform(cur_mask).to(self.mean.device)
            mask_ego_quantized = self.patch_quant_filter(mask_ego_resized.unsqueeze(0)).squeeze(0).squeeze(0)

            patches_ego_fg = mask_ego_quantized.view(-1) > self.mask_fg_threshold
            patches_exo_fg = mask_exo_quantized.view(-1)[patch_indices_exo] > self.mask_fg_threshold
            patches_fg = patches_ego_fg & patches_exo_fg

            locs_ego_fg = locs_2d_ego[patches_fg]
            locs_exo_fg = locs_2d_exo[patches_fg]

            if len(locs_ego_fg) == 0:
                all_points_ego.append(np.zeros((0, 2), dtype=np.float32))
                all_points_exo.append(np.zeros((0, 2), dtype=np.float32))
                num_matches_list.append(0)
                continue

            _, keep_idx = self._stratify_points(locs_ego_fg * scale_ego, self.stratify_distance_threshold**2)
            pts_ego = locs_ego_fg[keep_idx].cpu().numpy() * scale_ego
            pts_exo = locs_exo_fg[keep_idx].cpu().numpy() * scale_exo

            # --- 离群点过滤 + 采样逻辑保持不变 ---
            if len(pts_ego) > 3:
                d = np.linalg.norm(pts_ego[:, None, :] - pts_ego[None, :, :], axis=2)
                avg_d = d.mean(axis=1)
                n_remove = int(len(pts_ego) * self.outlier_removal_ratio)
                if n_remove > 0:
                    inliers = np.argsort(avg_d)[:-n_remove]
                    pts_ego = pts_ego[inliers]
                    pts_exo = pts_exo[inliers]

            if self.max_points_per_object is not None:
                n_pts = len(pts_ego)
                if n_pts > self.max_points_per_object:
                    sel = np.random.choice(n_pts, self.max_points_per_object, replace=False)
                else:
                    sel = np.random.choice(n_pts, self.max_points_per_object, replace=True)
                pts_ego = pts_ego[sel]
                pts_exo = pts_exo[sel]

            all_points_ego.append(pts_ego[:, [1, 0]])
            all_points_exo.append(pts_exo[:, [1, 0]])
            num_matches_list.append(len(pts_ego))

        result = {
            "points_ego": all_points_ego,
            "points_exo": all_points_exo,
            "num_matches": num_matches_list,
        }
        if return_features:
            result["features_ego"] = features_ego
            result["features_exo"] = features_exo
        return result



def load_dinov3_model(
    model_name: str = 'dinov3_vitl16',
    repo_path: str = None,
    weights_path: str = None,
    device: str = 'cuda',
) -> nn.Module:
    """
    Helper function to load DINOv3 model.
    
    Args:
        model_name: Name of the DINOv3 model
        repo_path: Path to local DINOv3 repository (if None, loads from GitHub)
        weights_path: Path to model weights (optional)
        device: Device to load model on
        
    Returns:
        Loaded DINOv3 model
    """
    if repo_path is not None:
        if weights_path is not None:
            model = torch.hub.load(
                repo_path, model_name, source='local', weights=weights_path
            )
        else:
            model = torch.hub.load(repo_path, model_name, source='local')
    else:
        model = torch.hub.load('facebookresearch/dinov3', model_name)
    
    # model = model.to(device)
    # model.eval()
    return model


# Model name to number of layers mapping
MODEL_TO_NUM_LAYERS = {
    'dinov3_vits16': 12,
    'dinov3_vits16plus': 12,
    'dinov3_vitb16': 12,
    'dinov3_vitl16': 24,
    'dinov3_vith16plus': 32,
    'dinov3_vit7b16': 40,
}


def visualize_correspondences(
    image_ego,
    image_exo,
    points_ego,
    points_exo,
    title="Sparse Correspondences",
    save_path=None,           # e.g. "outputs/corr_vis.png"
    dpi=150,                  # 保存分辨率
    show=True,                # 是否在屏幕显示
    close_after=True,         # 显示/保存后是否关闭 Figure 以释放内存
    transparent=False,        # 是否透明背景
    tight=True,               # 是否紧凑布局保存
):
    """Visualize sparse correspondences between two images and optionally save to file."""
    fig = plt.figure(figsize=(20, 10))
    
    ax1 = fig.add_subplot(121)
    ax1.imshow(image_ego)
    ax1.set_axis_off()
    ax1.set_title("Ego Image", fontsize=16)
    
    ax2 = fig.add_subplot(122)
    ax2.imshow(image_exo)
    ax2.set_axis_off()
    ax2.set_title("Exo Image", fontsize=16)
    
    # 用彩虹色绘制对应连线
    n = len(points_ego)
    colors = plt.cm.rainbow(np.linspace(0, 1, max(n, 1)))  # 避免 n=0 报错

    for i, ((x_ego, y_ego), (x_exo, y_exo)) in enumerate(zip(points_ego, points_exo)):
        con = ConnectionPatch(
            xyA=(x_ego, y_ego),
            xyB=(x_exo, y_exo),
            coordsA="data",
            coordsB="data",
            axesA=ax1,
            axesB=ax2,
            color=colors[i],
            linewidth=2,
        )
        ax2.add_artist(con)
    
    fig.suptitle(f"{title} ({n} matches)", fontsize=18)
    plt.tight_layout()

    # —— 保存到文件（如果需要） ——
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(
            save_path,
            dpi=dpi,
            transparent=transparent,
            bbox_inches="tight" if tight else None,
            pad_inches=0.1 if tight else None,
        )
        print(f"[Saved] {save_path}")

    # —— 显示 / 关闭 ——
    if show:
        plt.show()
    if close_after:
        plt.close(fig)