import cv2
from typing import Literal
import torch
import torch.nn as nn
import torch.nn.functional as F
from third_parts.mmdet.models.losses import CrossEntropyLoss
from xtuner.registry import BUILDER
from xtuner.model.utils import get_peft_model_state_dict
from xtuner.utils import PROMPT_TEMPLATE
from xtuner.tools.utils import get_stop_criteria
from transformers import GenerationConfig
from projects.v2sam.models.preprocess.image_resize import DirectResize
import numpy as np
from .utils import dynamic_preprocess
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from pycocotools import mask as _mask
from types import MethodType
from xtuner.model.utils import guess_load_checkpoint
from mmcv.ops import point_sample
from third_parts.mmdet.models.utils import get_uncertain_point_coords_with_randomness
from PIL import Image
import os
import torchvision.transforms.functional as TF
from transformers.cache_utils import Cache, DynamicCache
from mmengine.model import BaseModel
from .region_pooling import RegionPooling
from .vp_matcher import VPFeatureMatcher
from .sparse_correspondence import SparseCorrespondenceMatcher, load_dinov3_model, visualize_correspondences
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg




class V2SAM(BaseModel):
    def __init__(self,
                 grounding_encoder,
                 loss_mask=None,
                 loss_dice=None,
                 pretrained_pth=None,
                 frozen_sam2_decoder=True,
                 loss_sample_points=False,
                 num_points=12544,
                 use_fast_supervision=False,
                 bs=1,
                 dinov3_cfg=None,
                 sparse_corr_cfg=None,
                 ):
        super(V2SAM, self).__init__()
        self.grounding_encoder = BUILDER.build(grounding_encoder)
        self.grounding_encoder.requires_grad_(False)
        if not frozen_sam2_decoder:
            self.grounding_encoder.sam2_model.sam_mask_decoder.requires_grad_(True)
            self.grounding_encoder.sam2_model.sam_mask_decoder.iou_prediction_head.requires_grad_(False)
            self.grounding_encoder.sam2_model.sam_mask_decoder.pred_obj_score_head.requires_grad_(False)

        self.region_sampler = RegionPooling(num_sample_point=256)
        self.matcher = VPFeatureMatcher(dim=256)

        self.constr_prompt_fcs = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 256), nn.Dropout(0.0)
        )
        
        dinov3_cfg = dinov3_cfg or {}
        sparse_corr_cfg = sparse_corr_cfg or {}
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

        # wrz: load dinov3 model for sparse correspondence
        dinov3_repo_path = dinov3_cfg.get('repo_path', os.path.join(project_root, 'third_parts', 'dinov3'))
        if not os.path.isabs(dinov3_repo_path):
            dinov3_repo_path = os.path.abspath(os.path.join(project_root, dinov3_repo_path))

        dinov3_weights_path = dinov3_cfg.get('weights_path')
        if dinov3_weights_path:
            if not os.path.isabs(dinov3_weights_path):
                dinov3_weights_path = os.path.abspath(os.path.join(project_root, dinov3_weights_path))
        else:
            raise ValueError("DINOv3 weights_path must be provided via dinov3_cfg in the config.")

        self.dinov3_model = load_dinov3_model(
            model_name=dinov3_cfg.get('model_name', 'dinov3_vitl16'),
            repo_path=dinov3_repo_path,
            weights_path=dinov3_weights_path,
        )

        # wrz: initialize sparse correspondence matcher
        self.sparse_correspondence = SparseCorrespondenceMatcher(
            dinov3_model=self.dinov3_model,
            patch_size=sparse_corr_cfg.get('patch_size', 16),
            image_size=sparse_corr_cfg.get('image_size', 768),
            mask_fg_threshold=sparse_corr_cfg.get('mask_fg_threshold', 0.6),
            stratify_distance_threshold=sparse_corr_cfg.get('stratify_distance_threshold', 150.0),
            n_layers=sparse_corr_cfg.get('n_layers', 24),
            max_points_per_object=sparse_corr_cfg.get('max_points_per_object', 1),
            outlier_removal_ratio=sparse_corr_cfg.get('outlier_removal_ratio', 0.25),
        )
        # wrz: freeze dinov3 model
        for param in self.sparse_correspondence.dinov3_model.parameters():
                param.requires_grad = False
        print("DINOv3 Loaded!!!!!!!")

        
        self.loss_mask = BUILDER.build(loss_mask)
        self.loss_dice = BUILDER.build(loss_dice)
        if use_fast_supervision:
            self.loss_exists = BUILDER.build(dict(
                type=CrossEntropyLoss,
                use_sigmoid=True,
                reduction='mean',
                loss_weight=1.0)
            )
        if pretrained_pth is not None:
            pretrained_state_dict = guess_load_checkpoint(pretrained_pth)
            self.load_state_dict(pretrained_state_dict, strict=False)
            print(f'Load pretrained weight from {pretrained_pth}')

        self.loss_sample_points = loss_sample_points
        self.num_points = num_points
        self.oversample_ratio = 3.0
        self.importance_sample_ratio = 0.75

        self.bs = bs

    def _prep_sparse_correspondence_points(
        self, 
        points_list,  # list of np.ndarray [N_i, 2], each in original image coordinates
        orig_hw,      # tuple (H, W) - original image size
        device
    ):
        """
        将 sparse correspondence 的点从原始图片坐标系转换到 SAM2 内部的 1024x1024 坐标系。
        
        Args:
            points_list: list of np.ndarray, 每个元素是 [N_i, 2] 的点坐标 (x, y)，在原始图像坐标系中
            orig_hw: tuple (H, W), 原始图像的高度和宽度
            device: torch device
            
        Returns:
            dict with:
                'point_coords': torch.Tensor [B, P, 2] - 转换后的点坐标
                'point_labels': torch.Tensor [B, P] - 点标签（全为1，表示前景点）
        """
        from third_parts.sam2.utils.transforms import SAM2Transforms
        
        # 创建 SAM2Transforms 实例
        transforms = SAM2Transforms(
            resolution=1024,  # SAM2 的内部分辨率
            mask_threshold=0.0,
        )
        
        batch_coords = []
        batch_labels = []
        
        for points in points_list:
            if len(points) == 0:
                # # 如果没有点，创建一个空的 tensor
                batch_coords.append(torch.zeros((1, 2), dtype=torch.float32, device=device))
                batch_labels.append(torch.zeros((1,), dtype=torch.int, device=device))
                continue
            
            # 转换为 torch tensor
            points_tensor = torch.as_tensor(points, dtype=torch.float32, device=device)
            
            # 使用 SAM2Transforms 的 transform_coords 方法
            # normalize=True 表示输入是原始图像坐标，需要归一化
            transformed_points = transforms.transform_coords(
                points_tensor, 
                normalize=True, 
                orig_hw=orig_hw
            )
            
            # 创建标签（全为1，表示前景点）
            labels = torch.ones(len(points), dtype=torch.int, device=device)
            
            batch_coords.append(transformed_points)
            batch_labels.append(labels)
        
        # Stack 成 batch
        # 注意：如果每个对象的点数不同，需要 padding 或者分别处理
        # 这里假设可以直接 stack（即所有对象的点数相同）
        # 如果点数不同，需要特殊处理
        
        # 检查是否所有点数相同
        point_counts = [len(coords) for coords in batch_coords]
        if len(set(point_counts)) == 1 and point_counts[0] > 0:
            # 所有对象的点数相同，可以直接 stack
            point_coords = torch.stack(batch_coords, dim=0)  # [B, P, 2]
            point_labels = torch.stack(batch_labels, dim=0)  # [B, P]
        else:
            # 点数不同，需要 padding 到最大点数
            max_points = max(point_counts) if point_counts else 0
            if max_points == 0:
                point_coords = torch.zeros((len(points_list), 1, 2), dtype=torch.float32, device=device)
                point_labels = torch.zeros((len(points_list), 1), dtype=torch.int, device=device)
            else:
                padded_coords = []
                padded_labels = []
                for coords, labels in zip(batch_coords, batch_labels):
                    if len(coords) < max_points:
                        # Padding: 使用 -1 标签让 SAM2 忽略这些点
                        # 坐标仍然 padding 为 (0, 0)，但标签设为 -1 表示"忽略此点"
                        pad_size = max_points - len(coords)
                        coords_padded = torch.cat([
                            coords, 
                            torch.zeros((pad_size, 2), dtype=torch.float32, device=device)
                        ], dim=0)
                        # 使用 -1 标签表示 padding 的点应该被忽略
                        labels_padded = torch.cat([
                            labels,
                            torch.full((pad_size,), -1, dtype=torch.int, device=device)
                        ], dim=0)
                        padded_coords.append(coords_padded)
                        padded_labels.append(labels_padded)
                    else:
                        padded_coords.append(coords)
                        padded_labels.append(labels)
                
                point_coords = torch.stack(padded_coords, dim=0)  # [B, P, 2]
                point_labels = torch.stack(padded_labels, dim=0)  # [B, P]
        
        return {
            'point_coords': point_coords,
            'point_labels': point_labels
        }

    def _get_pesudo_data(self, dtype, device):
        g_pixel_values = torch.zeros((3, 1024, 1024), dtype=dtype, device=device)
        g_pixel_values = [g_pixel_values] * self.bs
        frames_per_batch = [1] * self.bs
        gt_masks = torch.zeros((5, 256, 256), dtype=torch.uint8, device=device)
        gt_masks = [gt_masks] * self.bs
        return g_pixel_values, frames_per_batch, gt_masks

    def process_video_gt_masks(self, gt_masks, frames_per_batch):
        gt_masks_video = []

        assert len(gt_masks) == len(frames_per_batch)
        for gt_masks_batch, frames_num in zip(gt_masks, frames_per_batch):
            N, H, W = gt_masks_batch.shape
            assert N % frames_num == 0
            gt_masks_batch = gt_masks_batch.reshape(
                N // frames_num, frames_num, H, W)
            for i in range(frames_num):
                gt_masks_video.append(gt_masks_batch[:, i])
        return gt_masks_video

    def sample_points(self, mask_pred, gt_masks):
        gt_masks = gt_masks.unsqueeze(1)
        gt_masks = gt_masks.to(mask_pred)
        mask_pred = mask_pred.unsqueeze(1)
        # (N, 1, h, w)

        with torch.no_grad():
            points_coords = get_uncertain_point_coords_with_randomness(
                mask_pred.to(torch.float32), None, self.num_points,
                self.oversample_ratio, self.importance_sample_ratio)
            # shape (num_total_gts, h, w) -> (num_total_gts, num_points)
            mask_point_targets = point_sample(
                gt_masks.float(), points_coords).squeeze(1)
        # shape (num_queries, h, w) -> (num_queries, num_points)
        mask_point_preds = point_sample(
            mask_pred.to(torch.float32), points_coords.to(torch.float32)).squeeze(1)
        return mask_point_preds.to(mask_pred.dtype), mask_point_targets.to(mask_pred.dtype)

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        return super().load_state_dict(state_dict, strict, assign)

    
    ### 保存所有参数
    def state_dict(self, *args, **kwargs):
        return super().state_dict(*args, **kwargs)
    
    # Debug: 检查可训练参数
    def print_trainable_parameters(self):
        """
        打印模型参数信息
        """
        print("\n" + "="*80)
        print("V2SAM Model Parameters:")
        print("="*80)
        
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(f"[Trainable] {name}")


    def forward(self, data, data_samples=None, mode='loss'):
        if mode == 'loss':
            g_pixel_values = data.pop('g_pixel_values', None)
            prompt_g_pixel_values = data.pop('prompt_g_pixel_values', None)
            gt_masks = data.pop('masks', None)
            prompt_masks = data.pop('prompt_masks', None)
            frames_per_batch = data.pop('frames_per_batch', None)

            # wrz: 取出图片路径和raw_prompt_masks用于sparse correspondence
            query_img_path_list = data.pop('query_img_path', None)
            target_img_path_list = data.pop('target_img_path', None)
            raw_prompt_masks_list = data.pop('raw_prompt_masks', None)

            # wrz: 只有当两个视角图片路径都存在时才进行sparse correspondence
            if query_img_path_list is not None and target_img_path_list is not None:
                # print(len(query_img_path_list))
                # print(len(query_img_path_list[0]))
                # print(len(target_img_path_list))
                # print(len(target_img_path_list[0]))
                # image_query = Image.open(query_img_path_list[0])
                # image_target = Image.open(target_img_path_list[0])

                image_query = [query_img_path[0] for query_img_path in query_img_path_list]
                image_target = [target_img_path[0] for target_img_path in target_img_path_list]
                # print(f"query: {query_img_path_list[0]}")
                # print(f"target: {target_img_path_list[0]}")

                # wrz: 从列表中取出query_mask, 并转换为二值numpy格式
                if raw_prompt_masks_list is not None and len(raw_prompt_masks_list) > 0:
                    # 判断是 list 还是 tensor
                    if isinstance(raw_prompt_masks_list, (list, tuple)):
                        # 处理 list[tensor1, tensor2, ...]
                        mask_query = []
                        for m in raw_prompt_masks_list:
                            m_np = m.detach().cpu().numpy()
                            m_np = (m_np > 0).astype(np.uint8)
                            # 去掉多余的通道维
                            mask_query.append(m_np)
                            
                    else:
                        raise TypeError(f"Unexpected type for raw_prompt_masks_list: {type(raw_prompt_masks_list)}")
                    # print(mask_query[0].shape)
                # wrz: 进行sparse correspondence, 获取目标图像中的points
                if mask_query is not None:
                    with torch.cuda.amp.autocast(enabled=False):
                        with torch.no_grad():  
                            corr_result = self.sparse_correspondence(
                                image_ego=image_query,
                                image_exo=image_target,
                                mask_ego=mask_query,  # [n_obj, H, W] 或 [H, W]
                                mask_exo=None,
                                return_features=False,
                            )
                    
                    # # wrz: 处理返回结果
                    # points_exo = corr_result['points_exo']  # list of [N_i, 2] arrays
                    # points_ego = corr_result['points_ego']  # list of [N_i, 2] arrays
                    # # wrz: 每个物体的匹配点数量
                    # num_matches = corr_result['num_matches']  # list of ints
                    
                    # # wrz: 将 points_exo 转换为 SAM2 内部坐标系
                    # # 获取 target 图像的原始尺寸
                    # target_h, target_w = image_target.size[1], image_target.size[0]  # PIL Image: (W, H)
                    # target_orig_hw = (target_h, target_w)
                    
                    # # 转换点坐标到 SAM2 1024x1024 坐标系
                    # sparse_points_dict = self._prep_sparse_correspondence_points(
                    #     points_list=points_exo,
                    #     orig_hw=target_orig_hw,
                    #     device=g_pixel_values[0].device
                    # )
                    points_exo_batch, points_ego_batch, num_matches_batch = self._flatten_corr_results(corr_result)
                    
                    # ------------------------------------------------------------
                    # ✅ wrz: 遍历每对图像 / target 一一对应处理
                    # ------------------------------------------------------------
                    sparse_points_batch = []
                    
                    
                    device = (
                        prompt_g_pixel_values[0].device
                        if prompt_g_pixel_values is not None
                        else "cuda"
                    )
                    
                    for i, (points_exo, image_target_i) in enumerate(zip(points_exo_batch, image_target)):
                        # PIL.Image.size -> (W, H)
                        target_h, target_w = image_target_i.size[1], image_target_i.size[0]
                        target_orig_hw = (target_h, target_w)
                    
                        sparse_points = self._prep_sparse_correspondence_points(
                            points_list=points_exo,
                            orig_hw=target_orig_hw,
                            device=device,
                        )
                        sparse_points_batch.append(sparse_points)
                        
                    sparse_points_dict = {}
                    point_coords_list = []
                    point_labels_list = []
                    for sparse_points in sparse_points_batch:
                        point_coords_list.append(sparse_points["point_coords"])
                        point_labels_list.append(sparse_points["point_labels"])
                        
                    # print(f"point_labels_list{len(point_labels_list)}")
                    # print(f"point_labels_list{point_labels_list[0].shape}")

                    # print("Number of tensors in point_coords_list:", len(point_coords_list))
                    # for i, t in enumerate(point_coords_list):
                    #     print(f"Tensor {i} shape:", t.shape)
                        
                    sparse_points_dict["point_coords"] = torch.cat(point_coords_list, dim=0)
                    sparse_points_dict["point_labels"] = torch.cat(point_labels_list, dim=0)



                    # 现在 sparse_points_dict 包含:
                    # 'point_coords': torch.Tensor [B, P, 2] - SAM2 内部坐标系
                    # 'point_labels': torch.Tensor [B, P] - 标签（全为1）
                    
                    # Debug: 可视化检查特定样本
                    # if query_img_path_list[0] == "xxxxxx":
                    #     with open("xxx/points_exo.txt", "w") as f:
                    #         for obj_idx, pts in enumerate(points_exo):
                    #             f.write(f"# Object {obj_idx}:\n")
                    #             for pt in pts:
                    #                 f.write(f"{pt[0].item()}, {pt[1].item()}\n")
                    #     # 可视化对应关系
                    #     visualize_correspondences(
                    #                     image_ego=image_query,
                    #                     image_exo=image_target,
                    #                     points_ego=points_ego[0],
                    #                     points_exo=points_exo[0],
                    #                     save_path="xxx/vis.png",
                    #                     title="Sparse Correspondences (Module Output)")


            assert frames_per_batch, "Video require frames_per_batch !!!"
            # print('frmaes_per_batch: ', frames_per_batch)
            ori_size_list = []
            for i_bs, mask in enumerate(gt_masks):
                mask_shape = mask.shape[-2:]
                ori_size_list += [mask_shape] * frames_per_batch[i_bs]


            gt_masks_video = self.process_video_gt_masks(gt_masks, frames_per_batch)

            g_pixel_values = torch.stack([
                self.grounding_encoder.preprocess_image(pixel) for pixel in g_pixel_values
            ])
            prompt_g_pixel_values = torch.stack([
                self.grounding_encoder.preprocess_image(prompt_pixel) for prompt_pixel in prompt_g_pixel_values
            ])
            # print(f"Done, {g_pixel_values.device} !!!\n\n")
            num_objs = gt_masks[0].shape[0]
            # print(num_objs)
            num_frames = len(frames_per_batch)
            # language_embeddings = torch.cat(pred_embeddings_list_video, dim=0)[:, None] # torch.Size([40, 1, 256])
            sam_states, prompt_vision_features, vision_features = self.grounding_encoder.get_sam2_embeddings(g_pixel_values, prompt_g_pixel_values, expand_size=num_objs)

            prompt_vp_embeds = self.region_sampler(prompt_vision_features, prompt_masks,
                                original_dtype=prompt_vision_features.dtype,
                                return_dtype=prompt_vision_features.dtype)
            vp_embeds = self.region_sampler(vision_features, gt_masks,
                                original_dtype=vision_features.dtype,
                                return_dtype=vision_features.dtype)
            prompt_vp_embeds = torch.cat(prompt_vp_embeds, dim=0)
            vp_embeds = torch.cat(vp_embeds, dim=0)
            prompt_masks_tensor = torch.cat(prompt_masks, dim=0) # debug: new feat

            # ===对齐dtype===
            if prompt_vp_embeds.dtype != next(self.matcher.parameters()).dtype:
                prompt_vp_embeds = prompt_vp_embeds.to(next(self.matcher.parameters()).dtype)
                vp_embeds = vp_embeds.to(next(self.matcher.parameters()).dtype)

            # ===预测与注入===
            predict_vp_embeds, pred_masks_tensor = self.matcher(prompt_vp_embeds, prompt_masks_tensor, vision_features) # 用来生成我们需要的特征，可以采用特征筛选的方式 
            pred_masks_tensor_list = [m.squeeze(0) for m in pred_masks_tensor.split(1, dim=0)]
            pred_mask_vp_embeds = self.region_sampler(vision_features, pred_masks_tensor_list,
                    original_dtype=vision_features.dtype,
                    return_dtype=vision_features.dtype)
            pred_mask_vp_embeds = torch.cat(pred_mask_vp_embeds, dim=0)
            loss_contr = self.get_contr_loss(vp_embeds, predict_vp_embeds)

            # wrz: 将 sparse_points_dict 点作为额外的点提示注入
            inject_feat = self.constr_prompt_fcs(torch.cat([predict_vp_embeds, pred_mask_vp_embeds], dim=-1))
            pred_masks = self.grounding_encoder.inject_language_embd(sam_states, inject_feat, sparse_points_dict, nf_nobj=(num_frames, num_objs))

            gt_masks_huge = [F.interpolate(gt_mask.unsqueeze(0).float(), size=pred_masks[0].shape[-2:], mode='bilinear', align_corners=False).squeeze(0) for gt_mask in gt_masks_video]
            gt_masks = torch.cat(gt_masks_huge, dim=0)
            # print(gt_masks.shape) # torch.Size([40, 256, 256])
            pred_masks = pred_masks.flatten(0, 1)

            # # === 简单可视化：每 10 step 保存一次预测 mask 与 GT 对比 ===
            # if not hasattr(V2SAM, "_vis_step"):
            #     V2SAM._vis_step = 0
            # V2SAM._vis_step += 1

            # if (V2SAM._vis_step % 10 == 0) and (not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0):
            #     try:
            #         save_dir = "vis_results_training"
            #         os.makedirs(save_dir, exist_ok=True)

            #         # 取第一个输入图像
            #         img_t = g_pixel_values[0]
            #         if img_t.ndim == 4:
            #             img_t = img_t[0]
            #         img_t = img_t.detach().float().cpu()
            #         img_t = (img_t - img_t.min()) / (img_t.max() - img_t.min() + 1e-5)
            #         base_img = TF.to_pil_image(img_t)

            #         # === 预测 mask 可视化 ===
            #         m_pred = pred_masks[0]
            #         if m_pred.ndim == 3:
            #             m_pred = m_pred[0]
            #         m_pred = (m_pred.sigmoid() > 0.5).to(torch.uint8).cpu().numpy() * 255
            #         mask_pred = Image.fromarray(m_pred, mode="L").resize(base_img.size, resample=Image.NEAREST)

            #         overlay_pred = base_img.copy().convert("RGBA")
            #         color_pred = Image.new("RGBA", overlay_pred.size, (255, 0, 0, 0))
            #         alpha_pred = mask_pred.point(lambda p: 100 if p > 0 else 0)
            #         color_pred.putalpha(alpha_pred)
            #         overlay_pred.alpha_composite(color_pred)

            #         # === GT mask 可视化 ===
            #         m_gt = gt_masks_[0]
            #         if m_gt.ndim == 3:
            #             m_gt = m_gt[0]
            #         m_gt = (m_gt > 0.5).to(torch.uint8).cpu().numpy() * 255
            #         mask_gt = Image.fromarray(m_gt, mode="L").resize(base_img.size, resample=Image.NEAREST)

            #         overlay_gt = base_img.copy().convert("RGBA")
            #         color_gt = Image.new("RGBA", overlay_gt.size, (0, 255, 0, 0))
            #         alpha_gt = mask_gt.point(lambda p: 100 if p > 0 else 0)
            #         color_gt.putalpha(alpha_gt)
            #         overlay_gt.alpha_composite(color_gt)

            #         # === 拼接左右子图 ===
            #         w, h = base_img.size
            #         combined = Image.new("RGB", (w * 2, h))
            #         combined.paste(overlay_pred.convert("RGB"), (0, 0))
            #         combined.paste(overlay_gt.convert("RGB"), (w, 0))

            #         save_path = os.path.join(save_dir, f"step_{V2SAM._vis_step:06d}.png")
            #         combined.save(save_path)
            #         print(f"[Vis] saved: {save_path}")
            #     except Exception as e:
            #         print(f"[Vis] failed: {e}")

            
            bs = len(pred_masks)
            loss_mask, loss_dice = 0, 0
            if len(pred_masks) != len(gt_masks):
                # drop this data
                print(f"Pred mask shape {pred_masks.shape} is not equal to gt_mask shape {gt_masks.shape} !!!")
                min_num = min(len(pred_masks), len(gt_masks))
                pred_masks = pred_masks[:min_num]
                gt_masks = gt_masks[:min_num]


            if self.loss_sample_points:
                sampled_pred_mask, sampled_gt_mask = self.sample_points(pred_masks, gt_masks)
                sam_loss_dice = self.loss_dice(
                    sampled_pred_mask,
                    sampled_gt_mask, avg_factor=(len(gt_masks) + 1e-4))
                sam_loss_mask = self.loss_mask(
                    sampled_pred_mask.reshape(-1),
                    sampled_gt_mask.reshape(-1),
                    avg_factor=(pred_masks.shape[0] * sampled_pred_mask.shape[1] + 1e-4))
            else:
                sam_loss_mask = self.loss_mask(pred_masks, gt_masks)
                sam_loss_dice = self.loss_dice(pred_masks, gt_masks)
            loss_mask += sam_loss_mask
            loss_dice += sam_loss_dice


            _scale = 10.0
            # print(type(loss_mask))
            loss_mask = loss_mask * _scale
            loss_dice = loss_dice * _scale

            if not hasattr(V2SAM, "_constr_step"):
                V2SAM._constr_step = 0

            V2SAM._constr_step += 1

            if V2SAM._constr_step >= 4000:
                _contr_scale = 1.0
            else:
                _contr_scale = 100.0

            loss_contr = loss_contr * _contr_scale

            # loss_mse = 0.0
            small_loss_mask, small_loss_dice = 0, 0

            small_loss_mask += self.loss_mask(pred_masks_tensor.flatten(0, 1), gt_masks)
            small_loss_dice += self.loss_dice(pred_masks_tensor.flatten(0, 1), gt_masks)

            
            # loss_mse += self.loss_mse(pred_masks_tensor.flatten(0, 1), self._normalize_mask(gt_masks, vision_features.dtype))


            
            loss_dict = {
                "loss_mask": loss_mask,
                "loss_dice": loss_dice,
                "small_loss_mask": small_loss_mask,
                "small_loss_dice": small_loss_dice,
                "loss_contr": loss_contr,
            }
            return loss_dict
        elif mode in ['eval', 'predict', 'test']:
            preds = self.predict(data)
            return  preds

    def get_contr_loss(self, image_feat, text_feat, idx=None, label=None, config=None):
        """
        Args:
            image_feat, text_feat: normalized
        Returns: contrastive loss
        """
        # assert image_feat.size(-1) == self.embed_dim
        # assert text_feat.size(-1) == self.embed_dim

        # image_feat_all = allgather(image_feat, torch.distributed.get_rank(), torch.distributed.get_world_size())
        # text_feat_all = allgather(text_feat, torch.distributed.get_rank(), torch.distributed.get_world_size())
        image_feat_all = F.normalize(image_feat[:,0,:])
        text_feat_all = F.normalize(text_feat[:,0,:])
        logits = image_feat_all @ text_feat_all.t() / 0.07
        bsz = image_feat_all.shape[0]

        if idx is None:
            labels = torch.arange(bsz, device=image_feat.device)  # 注意这里batchsize一定要大于1，不然loss为0
            loss_i2t = F.cross_entropy(logits, labels)
            loss_t2i = F.cross_entropy(logits.t(), labels)
        else:
            idx = idx.view(-1, 1)
            # assert idx.size(0) == image_feat.size(0)

            ## 生成对角阵
            # idx_all = allgather(idx, torch.distributed.get_rank(), torch.distributed.get_world_size())
            idx_all = idx
            pos_idx = torch.eq(idx_all, idx_all.t()).float()
            labels = pos_idx / pos_idx.sum(dim=1, keepdim=True)

            loss_i2t = -torch.sum(F.log_softmax(logits, dim=1) * labels, dim=1).mean()
            loss_t2i = -torch.sum(F.log_softmax(logits.t(), dim=1) * labels, dim=1).mean()

        loss = (loss_i2t + loss_t2i) / 2
        # loss_dict = {"contr_loss": loss}
        return loss

    def gradient_checkpointing_enable(self):
        self.activation_checkpointing_enable()

    def gradient_checkpointing_disable(self):
        self.activation_checkpointing_disable()

    @torch.no_grad()
    def predict(self, data):
        """
        推理函数（不计算 loss, 不还原原图尺寸）
        返回与训练时 mask 相同分辨率的预测结果。
        Args:
            data: dict, 需要字段:
                - g_pixel_values
                - prompt_g_pixel_values
                - prompt_masks
                - frames_per_batch
        Returns:
            dict:
                pred_probs: torch.Tensor [N, H, W] 概率图
                pred_masks: torch.Tensor [N, H, W] 二值图(>0.5)
        """
        # self.print_trainable_parameters() # debug
        g_pixel_values = data.pop('g_pixel_values', None)
        prompt_g_pixel_values = data.pop('prompt_g_pixel_values', None)
        gt_masks = data.pop('masks', None)
        prompt_masks = data.pop('prompt_masks', None)
        frames_per_batch = data.pop('frames_per_batch', None)

        # wrz: 取出图片路径和raw_prompt_masks用于sparse correspondence
        query_img_path_list = data.pop('query_img_path', None)
        target_img_path_list = data.pop('target_img_path', None)
        raw_prompt_masks_list = data.pop('raw_prompt_masks', None)

        
        # wrz: 只有当两个视角图片路径都存在时才进行sparse correspondence
        if query_img_path_list is not None and target_img_path_list is not None:
            # print(len(query_img_path_list))
            # print(len(query_img_path_list[0]))
            # print(len(target_img_path_list))
            # print(len(target_img_path_list[0]))
            # print(len(raw_prompt_masks_list))
            image_query = [query_img_path[i] for i in range(len(frames_per_batch)) for query_img_path in query_img_path_list]
            image_target = [target_img_path[i] for i in range(len(frames_per_batch)) for target_img_path in target_img_path_list]
            # print(len(image_query[0][0]))
            # image_query = Image.open(query_img_path_list[0])
            # image_target = Image.open(target_img_path_list[0])
            # print("图片加载成功", image_query.size, image_target.size)
            # print(f"query: {query_img_path_list[0]}")
            # print(f"target: {target_img_path_list[0]}")

            # wrz: 从列表中取出query_mask, 并转换为二值numpy格式
            # if raw_prompt_masks_list is not None and len(raw_prompt_masks_list) > 0:
            #     raw_prompt_masks = raw_prompt_masks_list # [n_obj, 704, 704]
            #     print(raw_prompt_masks_list[0].shape)
            #     # print("raw_prompt_masks-type:", type(raw_prompt_masks))  #  <class 'torch.Tensor'>
            #     # wrz: 转换为numpy并确保是0-1的二值mask
            #     mask_query = raw_prompt_masks.detach().cpu().numpy() # [n_obj, 704, 704]
            #     mask_query = (mask_query > 0).astype(np.uint8)  
            if raw_prompt_masks_list is not None and len(raw_prompt_masks_list) > 0:
                # 判断是 list 还是 tensor
                if isinstance(raw_prompt_masks_list, (list, tuple)):
                    # 处理 list[tensor1, tensor2, ...]
                    mask_query = []
                    for m in raw_prompt_masks_list:
                        m_np = m.detach().cpu().numpy()
                        m_np = (m_np > 0).astype(np.uint8)
                        # 去掉多余的通道维
                        mask_query.append(m_np)
                else:
                    raise TypeError(f"Unexpected type for raw_prompt_masks_list: {type(raw_prompt_masks_list)}")
            
                # print("mask_query shape:", mask_query.shape)
    
                # Debug: 保存特定图片的mask，用于可视化
                # if query_img_path_list[0] == "xxx":
                #     print("FOUND IT", "!"*30)
                #     mask_vis = mask_query[0] * 255
                #     mask_vis_uint8 = mask_vis.astype(np.uint8)
                #     print("mask_vis", mask_vis_uint8.shape)
                #     cv2.imwrite("xxx", mask_vis_uint8)
               

            # wrz: 进行sparse correspondence, 获取目标图像中的points
            if mask_query is not None:
                with torch.no_grad():  
                    corr_result = self.sparse_correspondence(
                        image_ego=image_query,
                        image_exo=image_target,
                        mask_ego=mask_query,  # [n_obj, H, W] 或 [H, W]
                        mask_exo=None,
                        return_features=False,
                    )
                
                # # wrz: 处理返回结果
                # points_exo = corr_result['points_exo']  # list of [N_i, 2] arrays
                # points_ego = corr_result['points_ego']  # list of [N_i, 2] arrays
                # # wrz: 每个物体的匹配点数量
                # num_matches = corr_result['num_matches']  # list of ints
                
                # # wrz: 将 points_exo 转换为 SAM2 内部坐标系
                # # 获取 target 图像的原始尺寸
                # target_h, target_w = image_target.size[1], image_target.size[0]  # PIL Image: (W, H)
                # target_orig_hw = (target_h, target_w)
                
                # # 转换点坐标到 SAM2 1024x1024 坐标系
                # sparse_points_dict = self._prep_sparse_correspondence_points(
                #     points_list=points_exo,
                #     orig_hw=target_orig_hw,
                #     device=prompt_g_pixel_values[0].device if prompt_g_pixel_values is not None else 'cuda'
                # )
                # print(sparse_points_dict)


                points_exo_batch, points_ego_batch, num_matches_batch = self._flatten_corr_results(corr_result)
                
                # ------------------------------------------------------------
                # ✅ wrz: 遍历每对图像 / target 一一对应处理
                # ------------------------------------------------------------
                sparse_points_batch = []
                
                
                device = (
                    prompt_g_pixel_values[0].device
                    if prompt_g_pixel_values is not None
                    else "cuda"
                )
                
                for i, (points_exo, image_target_i) in enumerate(zip(points_exo_batch, image_target)):
                    # PIL.Image.size -> (W, H)
                    target_h, target_w = image_target_i.size[1], image_target_i.size[0]
                    target_orig_hw = (target_h, target_w)
                
                    sparse_points = self._prep_sparse_correspondence_points(
                        points_list=points_exo,
                        orig_hw=target_orig_hw,
                        device=device,
                    )
                    sparse_points_batch.append(sparse_points)
                    
                sparse_points_dict = {}
                point_coords_list = []
                point_labels_list = []
                for sparse_points in sparse_points_batch:
                    point_coords_list.append(sparse_points["point_coords"])
                    point_labels_list.append(sparse_points["point_labels"])
                    
                sparse_points_dict["point_coords"] = torch.cat(point_coords_list, dim=0)
                sparse_points_dict["point_labels"] = torch.cat(point_labels_list, dim=0)

                # print(sparse_points_dict["point_coords"].shape)
                # print(sparse_points_dict["point_labels"].shape)
                # print(sparse_points_batch)
                # print(sparse_points_batch)
                
                # 现在 sparse_points_dict 包含:
                # 'point_coords': torch.Tensor [B, P, 2] - SAM2 内部坐标系
                # 'point_labels': torch.Tensor [B, P] - 标签（全为1）
                
                # Debug: 检查特定样本的输出, 保存点坐标为txt文件,并可视化检查
                # if query_img_path_list[0] == "xxxxxx":
                #     with open("xxx/points_exo.txt", "w") as f:
                #         for obj_idx, pts in enumerate(points_exo):
                #             f.write(f"# Object {obj_idx}:\n")
                #             for pt in pts:
                #                 f.write(f"{pt[0].item()}, {pt[1].item()}\n")
                #     # 可视化对应关系
                #     visualize_correspondences(
                #                     image_ego=image_query,
                #                     image_exo=image_target,
                #                     points_ego=points_ego,
                #                     points_exo=points_exo,
                #                     save_path="xxx/vis.png",
                #                     title="Sparse Correspondences (Module Output)")
                                                

        assert frames_per_batch, "Video require frames_per_batch !!!"

        # === 预处理图像 ===
        g_pixel_values = torch.stack([
            self.grounding_encoder.preprocess_image(pixel) for pixel in g_pixel_values
        ])
        prompt_g_pixel_values = torch.stack([
            self.grounding_encoder.preprocess_image(prompt_pixel) for prompt_pixel in prompt_g_pixel_values
        ])

        # === 获取对象数与帧数 ===
        num_objs = prompt_masks[0].shape[0]
        num_frames = len(frames_per_batch)
        # debug
        # print("对象数:", num_objs)
        # print("帧数:", num_frames)

        # === 提取特征 ===
        sam_states, prompt_vision_features, vision_features = \
            self.grounding_encoder.get_sam2_embeddings(
                g_pixel_values, prompt_g_pixel_values, expand_size=num_objs
            )
        
        # === prompt 区域采样 ===
        prompt_vp_embeds = self.region_sampler(
            prompt_vision_features, prompt_masks,
            original_dtype=prompt_vision_features.dtype,
            return_dtype=prompt_vision_features.dtype
        )
        # debug
        # print("region_pooling后vp_emb:", type(prompt_vp_embeds), len(prompt_vp_embeds))
        prompt_vp_embeds = torch.cat(prompt_vp_embeds, dim=0)
        prompt_masks_tensor = torch.cat(prompt_masks, dim=0) # debug: new feat

        # === dtype 对齐 ===
        if prompt_vp_embeds.dtype != next(self.matcher.parameters()).dtype:
            prompt_vp_embeds = prompt_vp_embeds.to(next(self.matcher.parameters()).dtype)

        # === 预测与注入 ===
        # debug: new feat
        predict_vp_embeds, pred_masks_tensor = self.matcher(prompt_vp_embeds, prompt_masks_tensor, vision_features) # 用来生成我们需要的特征，可以采用特征筛选的方式
        # pred_masks_tensor_05 = (pred_masks_tensor.sigmoid() > 0.5).float()
        pred_masks_tensor_list = [m.squeeze(0) for m in pred_masks_tensor.split(1, dim=0)]
        pred_mask_vp_embeds = self.region_sampler(vision_features, pred_masks_tensor_list,
                original_dtype=vision_features.dtype,
                return_dtype=vision_features.dtype)
        pred_mask_vp_embeds = torch.cat(pred_mask_vp_embeds, dim=0)
        inject_feat = self.constr_prompt_fcs(torch.cat([predict_vp_embeds, pred_mask_vp_embeds], dim=-1))
        # print(len(sparse_points_dict))
        # print(sparse_points_dict[0].shape)
        # wrz: 将 sparse_points_dict 点作为额外的点提示注入
        pred_masks = self.grounding_encoder.inject_language_embd(
            sam_states, inject_feat, sparse_points_dict, nf_nobj=(num_frames, num_objs)
        )
        # wrz：这里必须采用bilinear插值，否则预测mask会有锯齿
        pred_masks = [F.interpolate(pred_mask.unsqueeze(0), size=gt_masks[0].shape[-2:], mode='bilinear').squeeze(0) for pred_mask in pred_masks]
        pred_masks = torch.cat(pred_masks, dim=0)
        # print("*"*80)
        # print(gt_masks.shape) # torch.Size([40, 256, 256])
        # pred_masks = pred_masks.flatten(0, 1)  # [N, 1, H, W]

        # === sigmoid + 二值化 ===
        pred_probs = pred_masks.sigmoid()
        pred_masks = (pred_probs > 0.5).to(torch.uint8)
        pred_masks = self.fill_holes_in_masks(pred_masks) # debug: fill holes

        # # === 推理可视化：保存预测 mask 与 GT 对比 ===
        # if not hasattr(V2SAM, "_pred_vis_step"):
        #     V2SAM._pred_vis_step = 0
        # V2SAM._pred_vis_step += 1

        # try:
        #     save_dir = "/gemini/code/vis_results_inference2"
        #     os.makedirs(save_dir, exist_ok=True)

        #     # 取第一个输入图像
        #     img_t = g_pixel_values[0]
        #     if img_t.ndim == 4:
        #         img_t = img_t[0]
        #     img_t = img_t.detach().float().cpu()
        #     img_t = (img_t - img_t.min()) / (img_t.max() - img_t.min() + 1e-5)
        #     base_img = TF.to_pil_image(img_t)

        #     # === 预测 mask 可视化 ===
        #     m_pred = pred_masks[0]
        #     if m_pred.ndim == 3:
        #         m_pred = m_pred[0]
        #     m_pred = m_pred.cpu().numpy() * 255
        #     mask_pred = Image.fromarray(m_pred.astype(np.uint8), mode="L").resize(base_img.size, resample=Image.NEAREST)

        #     overlay_pred = base_img.copy().convert("RGBA")
        #     color_pred = Image.new("RGBA", overlay_pred.size, (255, 0, 0, 0))
        #     alpha_pred = mask_pred.point(lambda p: 100 if p > 0 else 0)
        #     color_pred.putalpha(alpha_pred)
        #     overlay_pred.alpha_composite(color_pred)
            
        #     # === 在预测图上绘制 sparse correspondence 点 ===
        #     if sparse_points_dict is not None and 'point_coords' in sparse_points_dict:
        #         # 转换为 PIL 可绘制的格式
        #         overlay_pred_with_points = overlay_pred.convert("RGB")
                
        #         # 创建 matplotlib figure
        #         fig, ax = plt.subplots(1, 1, figsize=(10, 10), dpi=100)
        #         ax.imshow(overlay_pred_with_points)
        #         ax.axis('off')
                
        #         # 获取点坐标 (需要从 SAM2 1024x1024 转回原图坐标)
        #         point_coords = sparse_points_dict['point_coords'].cpu().numpy()  # [B, P, 2]
        #         point_labels = sparse_points_dict['point_labels'].cpu().numpy()  # [B, P]
                
        #         # 计算缩放比例（从 SAM2 1024 坐标系转回原图）
        #         img_h, img_w = base_img.size[1], base_img.size[0]  # PIL: (W, H)
        #         scale_x = img_w / 1024
        #         scale_y = img_h / 1024
                
        #         # 绘制第一个对象的点（可以改为绘制所有对象）
        #         if len(point_coords) > 0:
        #             points = point_coords[0]  # [P, 2] (x, y) in SAM2 coords
        #             labels = point_labels[0]  # [P]
                    
        #             # 过滤有效点 (label != -1)
        #             valid_mask = labels != -1
        #             valid_points = points[valid_mask]
        #             valid_labels = labels[valid_mask]
                    
        #             if len(valid_points) > 0:
        #                 # 转换坐标到原图尺寸
        #                 valid_points_orig = valid_points.copy()
        #                 valid_points_orig[:, 0] *= scale_x  # x
        #                 valid_points_orig[:, 1] *= scale_y  # y
                        
        #                 # 绘制点：前景点(绿色星), 背景点(红色星)
        #                 for pt, label in zip(valid_points_orig, valid_labels):
        #                     if label == 1:  # 前景点
        #                         ax.scatter(pt[0], pt[1], c='lime', marker='*', s=200, 
        #                                  edgecolors='white', linewidths=1.5, zorder=10)
        #                     elif label == 0:  # 背景点
        #                         ax.scatter(pt[0], pt[1], c='red', marker='*', s=200,
        #                                  edgecolors='white', linewidths=1.5, zorder=10)
                
        #         # 转换 matplotlib figure 为 PIL Image
        #         canvas = FigureCanvasAgg(fig)
        #         canvas.draw()
        #         buf = canvas.buffer_rgba()
        #         overlay_pred_with_points = Image.frombytes('RGBA', canvas.get_width_height(), buf)
        #         overlay_pred = overlay_pred_with_points.convert("RGBA")
        #         plt.close(fig)

        #     # === GT mask 可视化 ===
        #     if gt_masks is not None and len(gt_masks) > 0:
        #         m_gt = gt_masks[0]
        #         if m_gt.ndim == 3:
        #             m_gt = m_gt[0]
        #         m_gt = (m_gt > 0.5).to(torch.uint8).cpu().numpy() * 255
        #         mask_gt = Image.fromarray(m_gt, mode="L").resize(base_img.size, resample=Image.NEAREST)

        #         overlay_gt = base_img.copy().convert("RGBA")
        #         color_gt = Image.new("RGBA", overlay_gt.size, (0, 255, 0, 0))
        #         alpha_gt = mask_gt.point(lambda p: 100 if p > 0 else 0)
        #         color_gt.putalpha(alpha_gt)
        #         overlay_gt.alpha_composite(color_gt)

        #         # === 拼接左右子图：预测(红) vs GT(绿) ===
        #         w, h = base_img.size
        #         combined = Image.new("RGB", (w * 2, h))
        #         combined.paste(overlay_pred.convert("RGB"), (0, 0))
        #         combined.paste(overlay_gt.convert("RGB"), (w, 0))
        #     else:
        #         # 如果没有 GT，只保存预测结果
        #         combined = overlay_pred.convert("RGB")

        #     # 从路径中提取样本信息作为文件名
        #     if target_img_path_list is not None and len(target_img_path_list) > 0:
        #         sample_name = os.path.basename(target_img_path_list[0]).split('.')[0]
        #         save_path = os.path.join(save_dir, f"pred_{V2SAM._pred_vis_step:06d}_{sample_name}.png")
        #     else:
        #         save_path = os.path.join(save_dir, f"pred_{V2SAM._pred_vis_step:06d}.png")
            
        #     combined.save(save_path)
        #     print(f"[Inference Vis] saved: {save_path}")
        # except Exception as e:
        #     print(f"[Inference Vis] failed: {e}")

        # 当前nf_no的所有目标都存在这里了
        results = [dict(
                    pred_masks=pred_masks, 
                    pred_probs=pred_probs,
                    # data_sample=data_samples[i] if data_samples is not None else None
                )]
        
        return results

    def _flatten_corr_results(self, corr_result):
        """兼容单对 / 多对输入的统一展开逻辑"""
        points_exo = corr_result["points_exo"]
        points_ego = corr_result["points_ego"]
        num_matches = corr_result["num_matches"]
    
        # 多对时：list of list
        if len(points_exo) > 0 and isinstance(points_exo[0], list):
            return points_exo, points_ego, num_matches
        # 单对时：直接包装成长度为1的 list
        else:
            return [points_exo], [points_ego], [num_matches]
            
    # mask后处理
    def fill_holes_in_masks(self, pred_masks: torch.Tensor) -> torch.Tensor:
        """
        对 [C, H, W] 的多通道mask进行洞填充。
        每个通道独立处理，填满内部空洞。
        输出保持原 dtype / device / shape。
        """
        assert pred_masks.ndim == 3, f"Expected [C, H, W], got {pred_masks.shape}"
    
        device = pred_masks.device
        dtype = pred_masks.dtype
        masks_np = (pred_masks.detach().cpu().numpy() > 0).astype(np.uint8)
        C, H, W = masks_np.shape
    
        filled = np.zeros_like(masks_np, dtype=np.uint8)
    
        for c in range(C):
            # OpenCV版填洞
            contours, _ = cv2.findContours(masks_np[c], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(filled[c], contours, -1, color=1, thickness=-1)
    
        out = torch.from_numpy(filled).to(device=device, dtype=torch.uint8)
        if dtype == torch.bool:
            out = out.to(torch.bool)
        return out
