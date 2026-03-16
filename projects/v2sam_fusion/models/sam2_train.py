import os.path

import torch

from hydra import compose
from hydra.utils import instantiate
from omegaconf import OmegaConf

from mmengine.model import BaseModule

from vlm.utils import load_checkpoint_with_prefix, load_state_dict_to_model


class SAM2TrainRunner(BaseModule):
    def __init__(
            self,
            cfg_path: str = "sam2_hiera_l.yaml",
            ckpt_path: str = "sam2_hiera_large.pt",
            hydra_overrides_extra=None,
            apply_postprocessing=True,
            base_dir: str = "weights/sam2",
    ):
        super().__init__(init_cfg=None)

        import third_parts.sam2 # noqa: F401

        if hydra_overrides_extra is None:
            hydra_overrides_extra = []
        hydra_overrides = [
            ## Extension: LLM prompt
            "++model._target_=projects.v2sam.models.extension.SAM2Base",
        ]

        if apply_postprocessing:
            hydra_overrides_extra = hydra_overrides_extra.copy()
            hydra_overrides_extra += [
                # dynamically fall back to multi-mask if the single mask is not stable
                # "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",
                # "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05",
                # "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98",
                # the sigmoid mask logits on interacted frames with clicks in the memory encoder so that the encoded masks are exactly as what users see from clicking
                # "++model.binarize_mask_from_pts_for_mem_enc=true",
                # fill small holes in the low-res masks up to `fill_hole_area` (before resizing them to the original video resolution)
                # "++model.fill_hole_area=8",
            ]
        hydra_overrides.extend(hydra_overrides_extra)

        # Read config and init model
        cfg = compose(config_name=cfg_path, overrides=hydra_overrides)
        OmegaConf.resolve(cfg)
        sam2_model = instantiate(cfg.model, _recursive_=True)
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
        resolved_base_dir = base_dir
        if not os.path.isabs(resolved_base_dir):
            resolved_base_dir = os.path.abspath(os.path.join(project_root, resolved_base_dir))

        state_dict = load_checkpoint_with_prefix(os.path.join(resolved_base_dir, ckpt_path))
        load_state_dict_to_model(sam2_model, state_dict)

        self.sam2_model = sam2_model

        self.hidden_dim = self.sam2_model.hidden_dim
        self.img_mean = (0.485, 0.456, 0.406)
        self.img_std = (0.229, 0.224, 0.225)

    def preprocess_image(self, image: torch.Tensor) -> torch.Tensor:
        image = image / 255.
        img_mean = torch.tensor(self.img_mean, dtype=image.dtype, device=image.device)[:, None, None]
        img_std = torch.tensor(self.img_std, dtype=image.dtype, device=image.device)[:, None, None]
        image -= img_mean
        image /= img_std
        return image

    def inject_language_embd(self, sam_states, language_embd, sparse_points_dict, nf_nobj=None):
        high_res_features = [
            x.permute(1, 2, 0).view(x.size(1), x.size(2), *s)
            for x, s in zip(sam_states['current_vision_feats'][:-1], sam_states['feat_sizes'][:-1])
        ]

        B = sam_states['current_vision_feats'][-1].size(1)  # batch size on this frame
        C = self.hidden_dim
        H, W = sam_states['feat_sizes'][-1]

        if self.sam2_model.directly_add_no_mem_embed:
            # directly add no-mem embedding (instead of using the transformer encoder)
            pix_feat_with_mem = sam_states['current_vision_feats'][-1] + self.sam2_model.no_mem_embed
            pix_feat_with_mem = pix_feat_with_mem.permute(1, 2, 0).view(B, C, H, W)
        else:
            raise NotImplementedError("directly add no memory embedding is not implemented")
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _, _, _, low_res_masks, high_res_masks, obj_ptr, _, = self.sam2_model._forward_sam_heads(
                backbone_features=pix_feat_with_mem,
                point_inputs=sparse_points_dict, # wrz: 传入 sparse points dict
                mask_inputs=None,
                high_res_features=high_res_features,
                multimask_output=self.sam2_model._use_multimask(is_init_cond_frame=True, point_inputs=sparse_points_dict),
                # Debug
                # Inject language Embed if possible
                language_embd=language_embd, # debug: 不想使用的直接设置为None
            )

        if nf_nobj is not None:
            pred_masks = low_res_masks.squeeze(1)
            pred_masks = pred_masks.unflatten(0, nf_nobj)
        else:
            pred_masks = low_res_masks
        return pred_masks

    # wrz: 双decoder的功能函数
    def predict_with_custom_decoder(self, sam_states, sparse_points_dict, custom_decoder, nf_nobj=None):
        """
        使用自定义decoder(如原始SAM2 decoder)基于sparse points进行预测
        
        Args:
            sam_states: backbone特征状态
            sparse_points_dict: sparse correspondence点坐标
            custom_decoder: 自定义的mask decoder
            nf_nobj: (num_frames, num_objs) tuple
        
        Returns:
            pred_masks: 预测的mask
        """
        high_res_features = [
            x.permute(1, 2, 0).view(x.size(1), x.size(2), *s)
            for x, s in zip(sam_states['current_vision_feats'][:-1], sam_states['feat_sizes'][:-1])
        ]

        B = sam_states['current_vision_feats'][-1].size(1)
        C = self.hidden_dim
        H, W = sam_states['feat_sizes'][-1]

        if self.sam2_model.directly_add_no_mem_embed:
            pix_feat_with_mem = sam_states['current_vision_feats'][-1] + self.sam2_model.no_mem_embed
            pix_feat_with_mem = pix_feat_with_mem.permute(1, 2, 0).view(B, C, H, W)
        else:
            raise NotImplementedError("directly add no memory embedding is not implemented")
        
        # 使用自定义decoder进行预测
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            # 调用SAM2的_forward_sam_heads但使用自定义decoder
            _, _, _, low_res_masks, _, _, _ = self.sam2_model._forward_sam_heads(
                backbone_features=pix_feat_with_mem,
                point_inputs=sparse_points_dict,
                mask_inputs=None,
                high_res_features=high_res_features,
                multimask_output=self.sam2_model._use_multimask(is_init_cond_frame=True, point_inputs=sparse_points_dict),
                language_embd=None,  # 不使用language embedding
            )

        if nf_nobj is not None:
            pred_masks = low_res_masks.squeeze(1)
            pred_masks = pred_masks.unflatten(0, nf_nobj)
        else:
            pred_masks = low_res_masks
        return pred_masks

    def get_sam2_embeddings(self, images, prompt_images, expand_size=1):
        # Step 1: inference the backbone with the images
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            feats = self.sam2_model.forward_image(images)
            prompt_feats = self.sam2_model.forward_image(prompt_images)
            prompt_vision_features = prompt_feats['vision_features'].flatten(2).permute(0, 2, 1)
            vision_features = feats['vision_features'].flatten(2).permute(0, 2, 1)

        if expand_size > 1:
            # print("prompt_feats shape is:", prompt_feats.shape)
            # feats['vision_features'] = feats['vision_features'][:, None].expand(-1, expand_size, -1, -1, -1).flatten(0, 1) # torch.Size([B*expand_size, C, H, W]) = torch.Size([40, 256, 64, 64])
            # print("vision_features shape is:", feats['vision_features'].shape)
            for i, feat in enumerate(feats["backbone_fpn"]):
                feats["backbone_fpn"][i] = feat[:, None].expand(-1, expand_size, -1, -1, -1).flatten(0, 1) # torch.Size([B*expand_size, C, H, W]) = torch.Size([40, 256, 64, 64])
            for i, pos in enumerate(feats["vision_pos_enc"]):
                pos = pos[:, None].expand(-1, expand_size, -1, -1, -1).flatten(0, 1)
                feats["vision_pos_enc"][i] = pos # torch.Size([B*expand_size, C, H, W]) = torch.Size([40, 256, 64, 64])

        # Step 2: Process the features to output
        _, current_vision_feats, current_vision_pos_embeds, feat_sizes = self.sam2_model._prepare_backbone_features(feats)
        # print("feat_sizes:", feat_sizes)
        # print("len(current_vision_feats):", len(current_vision_feats))
        # print("current_vision_feats shape is:", current_vision_feats[0].shape)
        # print("current_vision_pos_embeds shape is:", current_vision_pos_embeds[0].shape)
        return {
            "current_vision_feats": current_vision_feats,
            "current_vision_pos_embeds": current_vision_pos_embeds,
            "feat_sizes": feat_sizes,
        }, prompt_vision_features, vision_features

    def forward(self, batch):
        raise NotImplementedError
