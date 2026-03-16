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
import cv2

class V2SAM(BaseModel):
    def __init__(self,
                 grounding_encoder,
                 loss_mask=None,
                 loss_dice=None,
                 loss_mse=None,
                 pretrained_pth=None,
                 frozen_sam2_decoder=True,
                 loss_sample_points=False,
                 num_points=12544,
                 use_fast_supervision=False,
                 bs=1,
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

        self.loss_mask = BUILDER.build(loss_mask)
        self.loss_dice = BUILDER.build(loss_dice)
        # self.loss_mse = BUILDER.build(loss_mse)
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

    # def state_dict(self, *args, **kwargs):
    #     state_dict = super().state_dict(*args, **kwargs)
    #     from collections import OrderedDict

    #     # for k, v in state_dict.items():
    #     #     print(k)

    #     to_return = OrderedDict()

    #     to_return.update(
    #         {k: v
    #          for k, v in state_dict.items() if 'sam_mask_decoder' in k})
    #     to_return.update(
    #         {k: v
    #          for k, v in state_dict.items() if 'text_exist_fcs' in k}
    #     )
    #     to_return.update(
    #         {k: v
    #          for k, v in state_dict.items() if 'constr_prompt_fcs' in k})
    #     return to_return
    
    ### 保存所有参数
    def state_dict(self, *args, **kwargs):
        return super().state_dict(*args, **kwargs)


    def forward(self, data, data_samples=None, mode='loss'):
        if mode == 'loss':
            g_pixel_values = data.pop('g_pixel_values', None)
            prompt_g_pixel_values = data.pop('prompt_g_pixel_values', None)
            gt_masks = data.pop('masks', None)
            prompt_masks = data.pop('prompt_masks', None)
            frames_per_batch = data.pop('frames_per_batch', None)
            # input_ids = data['input_ids']


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
            # print(len(prompt_masks))
            # print(prompt_masks[0].shape)
            prompt_masks_tensor = torch.cat(prompt_masks, dim=0)
            
            if prompt_vp_embeds.dtype != next(self.matcher.parameters()).dtype:
                prompt_vp_embeds = prompt_vp_embeds.to(next(self.matcher.parameters()).dtype)
                vp_embeds = vp_embeds.to(next(self.matcher.parameters()).dtype)
            # print("vp_embeds: ", vp_embeds.shape, vp_embeds.dtype, vp_embeds.device)
            
            # print(prompt_masks_tensor.shape)
            predict_vp_embeds, pred_masks_tensor = self.matcher(prompt_vp_embeds, prompt_masks_tensor, vision_features) # 用来生成我们需要的特征，可以采用特征筛选的方式
            # pred_masks_tensor_05 = (pred_masks_tensor.sigmoid() > 0.5).float()
            pred_masks_tensor_list = [m.squeeze(0) for m in pred_masks_tensor.split(1, dim=0)]

            pred_mask_vp_embeds = self.region_sampler(vision_features, pred_masks_tensor_list,
                    original_dtype=vision_features.dtype,
                    return_dtype=vision_features.dtype)
            pred_mask_vp_embeds = torch.cat(pred_mask_vp_embeds, dim=0)
            # predict_vp_embeds = self.matcher(prompt_vp_embeds, vision_features) # 用来生成我们需要的特征，可以采用特征筛选的方式
            loss_contr = self.get_contr_loss(vp_embeds, predict_vp_embeds)
  
            # print("vp_embeds_: ", vp_embeds_.shape, vp_embeds_.dtype, vp_embeds_.device)
            # print(f"Done, {g_pixel_values.device} !!! {num_frames}---{num_objs}, {language_embeddings.shape}, {vp_embeds.shape}\n\n")
            # pred_masks = self.grounding_encoder.inject_language_embd(sam_states, vp_embeds, nf_nobj=(num_frames, num_objs))
            # pred_masks = self.grounding_encoder.inject_language_embd(sam_states, self.constr_prompt_fcs(predict_vp_embeds), nf_nobj=(num_frames, num_objs))
            pred_masks = self.grounding_encoder.inject_language_embd(sam_states, self.constr_prompt_fcs(torch.cat([predict_vp_embeds, pred_mask_vp_embeds], dim=-1)), nf_nobj=(num_frames, num_objs))
            # pred_masks = self.grounding_encoder.inject_language_embd(sam_states, language_embeddings, nf_nobj=(num_frames, num_objs))
            # gt_masks_small = [F.interpolate(gt_mask.unsqueeze(0), size=pred_masks_tensor[0].shape[-2:], mode='nearest').squeeze(0) for gt_mask in gt_masks_video]
            # gt_masks_small = torch.cat(gt_masks_small, dim=0)
            gt_masks_huge = [F.interpolate(gt_mask.unsqueeze(0).float(), size=pred_masks[0].shape[-2:], mode='bilinear', align_corners=False).squeeze(0) for gt_mask in gt_masks_video]
            gt_masks = torch.cat(gt_masks_huge, dim=0)
            # print(gt_masks.shape) # torch.Size([40, 256, 256])
            pred_masks = pred_masks.flatten(0, 1)
            # pred_masks_ = torch.cat(pred_masks, dim=0)
            # print(pred_masks.shape)

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
            # # === 在这里组装成标准格式 ===
            # pred_masks_list = []
            # pred_probs_list = []
            # for i in range(len(preds["pred_masks"])):
            #     preds_masks_by_num_obj = preds["pred_masks"][i].cpu()
            #     preds_probs_by_num_obj = preds["pred_probs"][i].cpu()
            #     pred_masks_list.append(preds_masks_by_num_obj)
            #     pred_probs_list.append(preds_probs_by_num_obj)

            # results = [dict(
            #         pred_masks= pred_masks_list, 
            #         pred_probs=pred_probs_list,
            #         # data_sample=data_samples[i] if data_samples is not None else None
            #     )]
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
        # print(logits)
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
        推理函数（不计算 loss，不还原原图尺寸）
        返回与训练时 mask 相同分辨率的预测结果。
        Args:
            data: dict，需要字段：
                - g_pixel_values
                - prompt_g_pixel_values
                - prompt_masks
                - frames_per_batch
        Returns:
            dict:
                pred_probs: torch.Tensor [N, H, W] 概率图
                pred_masks: torch.Tensor [N, H, W] 二值图(>0.5)
        """
        g_pixel_values = data.pop('g_pixel_values', None)
        prompt_g_pixel_values = data.pop('prompt_g_pixel_values', None)
        gt_masks = data.pop('masks', None)
        prompt_masks = data.pop('prompt_masks', None)
        frames_per_batch = data.pop('frames_per_batch', None)

        # print(len(g_pixel_values))
        # print(g_pixel_values[0].shape)

        assert frames_per_batch, "Video require frames_per_batch !!!"

        gt_masks_video = self.process_video_gt_masks(gt_masks, frames_per_batch)
        
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

        # === 提取特征 ===
        sam_states, prompt_vision_features, vision_features = \
            self.grounding_encoder.get_sam2_embeddings(
                g_pixel_values, prompt_g_pixel_values, expand_size=num_objs
            )
        # print(prompt_vision_features.shape)
        # print(prompt_masks[0].shape)
        # === prompt 区域采样 ===
        prompt_vp_embeds = self.region_sampler(
            prompt_vision_features, prompt_masks,
            original_dtype=prompt_vision_features.dtype,
            return_dtype=prompt_vision_features.dtype
        )
        prompt_vp_embeds = torch.cat(prompt_vp_embeds, dim=0)
        prompt_masks_tensor = torch.cat(prompt_masks, dim=0)
        
        # === dtype 对齐 ===
        if prompt_vp_embeds.dtype != next(self.matcher.parameters()).dtype:
            prompt_vp_embeds = prompt_vp_embeds.to(next(self.matcher.parameters()).dtype)

        # === 预测与注入 ===
        # predict_vp_embeds = self.matcher(prompt_vp_embeds, vision_features)
        # inject_feat = self.constr_prompt_fcs(predict_vp_embeds)

        predict_vp_embeds, pred_masks_tensor = self.matcher(prompt_vp_embeds, prompt_masks_tensor, vision_features) # 用来生成我们需要的特征，可以采用特征筛选的方式
        # pred_masks_tensor_05 = (pred_masks_tensor.sigmoid() > 0.5).float()
        pred_masks_tensor_list = [m.squeeze(0) for m in pred_masks_tensor.split(1, dim=0)]

        pred_mask_vp_embeds = self.region_sampler(vision_features, pred_masks_tensor_list,
                original_dtype=vision_features.dtype,
                return_dtype=vision_features.dtype)
        pred_mask_vp_embeds = torch.cat(pred_mask_vp_embeds, dim=0)
        inject_feat = self.constr_prompt_fcs(torch.cat([predict_vp_embeds, pred_mask_vp_embeds], dim=-1))
        
        pred_masks = self.grounding_encoder.inject_language_embd(
            sam_states, inject_feat, nf_nobj=(num_frames, num_objs)
        )
        pred_mask_ = [F.interpolate(pred_mask.unsqueeze(0), size=gt_masks[0].shape[-2:], mode='bilinear', align_corners=False).squeeze(0) for pred_mask in pred_masks]
        pred_masks_tensor_1 = [F.interpolate(pred_masks_tensor_.unsqueeze(0), size=gt_masks[0].shape[-2:], mode='bilinear', align_corners=False).squeeze(0) for pred_masks_tensor_ in pred_masks_tensor]
        gt_masks_ = [F.interpolate(gt_mask.unsqueeze(0), size=pred_mask_[0].shape[-2:], mode='bilinear', align_corners=False).squeeze(0) for gt_mask in gt_masks_video]

        pred_masks = torch.cat(pred_mask_, dim=0)
        # print(pred_masks.shape)
        # print(gt_masks.shape) # torch.Size([40, 256, 256])
        # pred_masks = pred_masks.flatten(0, 1)  # [N, 1, H, W]

        # === sigmoid + 二值化 ===
        pred_probs = pred_masks.sigmoid()
        pred_masks = (pred_probs > 0.5).to(torch.uint8)
        # print(pred_masks.shape)
        # pred_masks = self.refine_masks(pred_masks, min_area=120, kernel_size=5)
        # pred_masks = self.keep_largest_component_masks(pred_masks, binarized=True, min_area=0)
        pred_masks = self.fill_holes_in_masks(pred_masks)
        # print(len(pred_mask_))
        # print(len(gt_masks))
        # print(pred_mask_[0].shape)
        # print(gt_masks[0].shape)
        
        # if not hasattr(V2SAM, "_vis_step"):
        #     V2SAM._vis_step = 0
        # V2SAM._vis_step += 1
        
        # if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        #     try:
        #         save_dir = "vis_results_testing"
        #         os.makedirs(save_dir, exist_ok=True)
    
        #         # 取第一个输入图像
        #         img_t = g_pixel_values[0]
        #         if img_t.ndim == 4:
        #             img_t = img_t[0]
        #         img_t = img_t.detach().float().cpu()
        #         img_t = (img_t - img_t.min()) / (img_t.max() - img_t.min() + 1e-5)
        #         base_img = TF.to_pil_image(img_t)
    
        #         # 取第一个输入的提示图像
        #         prompt_img_t = prompt_g_pixel_values[0]
        #         if prompt_img_t.ndim == 4:
        #             prompt_img_t = prompt_img_t[0]
        #         prompt_img_t = prompt_img_t.detach().float().cpu()
        #         prompt_img_t = (prompt_img_t - prompt_img_t.min()) / (prompt_img_t.max() - prompt_img_t.min() + 1e-5)
        #         prompt_base_img = TF.to_pil_image(prompt_img_t)
                
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
    
        #         # === GT mask 另一个视角可视化 ===
        #         prompt_m_gt = prompt_masks[0]
        #         if prompt_m_gt.ndim == 3:
        #             prompt_m_gt = prompt_m_gt[0]
        #         prompt_m_gt = (prompt_m_gt > 0.5).to(torch.uint8).cpu().numpy() * 255
        #         prompt_mask_gt = Image.fromarray(prompt_m_gt, mode="L").resize(prompt_base_img.size, resample=Image.NEAREST)
    
        #         prompt_overlay_gt = prompt_base_img.copy().convert("RGBA")
        #         prompt_color_gt = Image.new("RGBA", prompt_overlay_gt.size, (0, 255, 0, 0))
        #         prompt_alpha_gt = prompt_mask_gt.point(lambda p: 100 if p > 0 else 0)
        #         prompt_color_gt.putalpha(prompt_alpha_gt)
        #         prompt_overlay_gt.alpha_composite(prompt_color_gt)
                
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
        #         # combined.paste(overlay_gt.convert("RGB"), (0, 0))
        #         # combined.paste(prompt_overlay_gt.convert("RGB"), (w, 0))
    
        #         save_path = os.path.join(save_dir, f"step_{V2SAM._vis_step:06d}.png")
        #         combined.save(save_path)
        #         print(f"[Vis] saved: {save_path}")
        #     except Exception as e:
        #         print(f"[Vis] failed: {e}")
            
        # 当前nf_no的所有目标都存在这里了
        results = [dict(
                    pred_masks=pred_masks, 
                    pred_probs=pred_probs,
                    # data_sample=data_samples[i] if data_samples is not None else None
                )]
        
        return results
        
    # ========================
    # 🧩 内部辅助函数：类型归一化
    # ========================
    def _normalize_mask(self, mask: torch.Tensor, ref_dtype: torch.dtype) -> torch.Tensor:
        """
        自动将 mask 转为浮点类型，并匹配模型 dtype。
        支持 bool, uint8, int, float 等类型。
        """
        if mask.dtype == torch.bool:
            mask = mask.to(dtype=ref_dtype)
        elif mask.dtype in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
            mask = (mask > 0).to(dtype=ref_dtype)
        else:
            mask = mask.to(dtype=ref_dtype)
        return mask.clamp(0, 1)
            
    def keep_largest_component_masks(self,
        pred_masks: torch.Tensor,
        binarized: bool = True,   # 传进来是否已是0/1(或0/255)的二值mask
        min_area: int = 0,        # 可选：若最大的连通域也小于该阈值，则整通道清零
    ) -> torch.Tensor:
        """
        对 [C, H, W] 的多通道mask进行后处理：
        - 每个通道仅保留“最大连通域”，其它全部置零
        - 可选：若最大连通域面积 < min_area，则该通道清零
        - 返回与输入相同的 dtype / device / shape
    
        参数：
        pred_masks: torch.uint8 或 bool，形状 [C, H, W]。若非二值，需先阈值化。
        binarized:  若为 False，则会以 >0 自动二值化；True 则直接当作二值使用。
        min_area:   面积阈值（像素数），用于过滤太小的最大连通域。
        """
        assert pred_masks.ndim == 3, f"Expected [C, H, W], got {pred_masks.shape}"
        device = pred_masks.device
        out_dtype = pred_masks.dtype
        C, H, W = pred_masks.shape
    
        # 转 numpy，确保是0/255
        m = pred_masks.detach().cpu().numpy()
        if not binarized:
            m = (m > 0).astype(np.uint8)
        else:
            # 若原本是0/1，转为0/1；若已是0/255也没关系
            m = (m > 0).astype(np.uint8)
        m = m * 255
    
        result = np.zeros_like(m, dtype=np.uint8)
    
        for c in range(C):
            mask = m[c]  # (H, W), uint8 in {0,255}
    
            # 连通域分析
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
            if num_labels <= 1:
                # 没有前景（只有背景），该通道保持全零
                continue
    
            # 找到除背景(label=0)外面积最大的label
            areas = stats[1:, cv2.CC_STAT_AREA]  # 跳过背景
            max_idx = np.argmax(areas) + 1       # 补回真实label索引
            max_area = int(areas[max_idx - 1])
    
            if max_area >= min_area:
                result[c][labels == max_idx] = 255
            # 否则保持全零（等于过滤掉这个通道的全部前景）
    
        # 还原到原来的类型与取值（0/1 if 原来是0/1；保持uint8则0/1或0/255都可）
        out = torch.from_numpy((result > 0).astype(np.uint8)).to(device=device)
        if out_dtype == torch.bool:
            out = out.to(torch.bool)
        else:
            out = out.to(torch.uint8)  # 多数分割后处理都用uint8的0/1
    
        return out  # 形状 [C, H, W]



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

    def refine_masks(self,
        pred_masks: torch.Tensor,
        min_area: int = 100,
        kernel_size: int = 5,
        smooth: bool = True
    ) -> torch.Tensor:
        """
        对 SAM2 输出的多个 mask (形状 [C, H, W]) 进行形态学后处理。
        
        每个通道单独处理：
          1. 闭运算连接边缘裂缝
          2. 开运算去除离散点
          3. 小连通域过滤
          4. 可选平滑抗锯齿
    
        返回与输入形状、dtype、device 一致。
        """
    
        assert pred_masks.ndim == 3, f"Expected [C, H, W], got {pred_masks.shape}"
    
        device = pred_masks.device
        dtype = torch.uint8
        C, H, W = pred_masks.shape
    
        # 转 numpy 并二值化
        masks_np = (pred_masks.detach().cpu().numpy() > 0).astype(np.uint8) * 255
    
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        refined_channels = []
    
        for c in range(C):
            mask = masks_np[c]
    
            # 1️⃣ 闭运算：填补边缘裂缝
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
            # 2️⃣ 开运算：去除离散小点
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
            # 3️⃣ 删除小连通区域
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
            cleaned = np.zeros_like(mask)
            for i in range(1, num_labels):  # 跳过背景
                if stats[i, cv2.CC_STAT_AREA] >= min_area:
                    cleaned[labels == i] = 255
    
            # 4️⃣ 可选平滑抗锯齿
            if smooth:
                mask = cv2.GaussianBlur(cleaned.astype(np.float32), (3, 3), sigmaX=0.5)
                mask = (mask > 127).astype(np.uint8)
            else:
                mask = (cleaned > 127).astype(np.uint8)
    
            refined_channels.append(mask)
    
        refined_np = np.stack(refined_channels, axis=0)
        refined_masks = torch.from_numpy(refined_np).to(device=device, dtype=dtype)
        return refined_masks

    def refine_mask_connect_fill(self,
        pred_masks: torch.Tensor,
        kernel_size: int = 5,
        min_area: int = 0,
        smooth: bool = True
    ) -> torch.Tensor:
        """
        综合后处理:
          - 对每个通道进行闭运算(连接断裂)
          - 填充内部空洞
          - 在所有通道中保留全局最大连通域
          - 可选平滑边缘
        输入: [C, H, W]
        输出: 与输入同shape/dtype/device
        """
    
        assert pred_masks.ndim == 3, f"Expected [C, H, W], got {pred_masks.shape}"
        device = pred_masks.device
        dtype = pred_masks.dtype
        C, H, W = pred_masks.shape
    
        # → numpy uint8 0/255
        masks_np = (pred_masks.detach().cpu().numpy() > 0).astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
        # --- Step 1: 每个通道先修复边缘并填洞 ---
        processed = np.zeros_like(masks_np, dtype=np.uint8)
        for c in range(C):
            m = masks_np[c]
    
            # 闭运算：修复狭长断裂
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)
    
            # 洞填充：内部空白点
            contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            filled = np.zeros_like(m)
            cv2.drawContours(filled, contours, -1, color=255, thickness=-1)
    
            # 可选轻微平滑，减少锯齿
            if smooth:
                filled = cv2.GaussianBlur(filled, (3, 3), sigmaX=0.5)
                filled = (filled > 127).astype(np.uint8) * 255
    
            processed[c] = filled
    
        # --- Step 2: 找出全图最大连通域 ---
        best_area = 0
        best_c = -1
        best_label = -1
        labels_cache = {}
    
        for c in range(C):
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(processed[c], connectivity=8)
            labels_cache[c] = labels
            if num_labels <= 1:
                continue
            areas = stats[1:, cv2.CC_STAT_AREA]
            max_idx = np.argmax(areas) + 1
            max_area = int(areas[max_idx - 1])
            if max_area > best_area:
                best_area = max_area
                best_c = c
                best_label = max_idx
    
        result = np.zeros_like(processed)
        if best_c != -1 and best_area >= min_area:
            result[best_c][labels_cache[best_c] == best_label] = 255
    
        # --- Step 3: 转回 torch 张量 ---
        out = torch.from_numpy((result > 0).astype(np.uint8)).to(device=device)
        out = out.to(dtype)
    
        return out

    