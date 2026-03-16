import torch
import torch.nn as nn
import torch.nn.functional as F


class VPFeatureMatcher(nn.Module):
    """
    输入:
        prompt_vp_embeds: [B*num_obj, 1, D]
        prompt_masks_tensor: [B*num_obj, H, W]
        vision_features: [B, H*W, D]
    输出:
        predict_vp_embeds: [B*num_obj, 1, D]
        pred_masks_tensor: [B, num_obj, 256, 256]
    """

    def __init__(self, dim=256, num_layers=2, use_softmax=True, out_size=256):
        super().__init__()
        self.dim = dim
        self.num_layers = num_layers
        self.use_softmax = use_softmax
        self.out_size = out_size
        self.base_res = out_size // 8  # 与 ConvTranspose2d ×8 对齐，默认 256->32 起始分辨率

        # === 1) 几何编码（与原始一致）===
        self.mask_encoder = nn.Sequential(
            nn.Conv2d(1, dim // 4, 3, padding=1, stride=2),  # H/2
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // 4, dim // 2, 3, padding=1, stride=2),  # H/4
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // 2, dim, 3, padding=1),
            nn.AdaptiveAvgPool2d(1)
        )

        # === 2) QKV（与原始一致）===
        self.query_proj = nn.Linear(dim, dim)
        self.key_proj = nn.Linear(dim, dim)
        self.value_proj = nn.Linear(dim, dim)

        # === 3) spatial gate（与原始一致）===
        self.spatial_gate = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(dim // 2, 1),
            nn.Sigmoid()
        )

        # === 4) cross attention refinement（与原始一致）===
        self.cross_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=8,
                dim_feedforward=dim * 4,
                activation="gelu",
                batch_first=True
            )
            for _ in range(num_layers)
        ])

        # === 5) residual MLP（与原始一致）===
        self.residual_mlp = nn.Sequential(
            nn.Linear(dim * 2, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim)
        )

        # === 6) 新增：mask prior encoder（mask→mask 主干）===
        # 将输入 mask 下采样到 base_res，并编码到 dim 通道，作为 decoder 的主要输入
        self.mask_prior_encoder = nn.Sequential(
            nn.Conv2d(1, dim // 4, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // 4, dim // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // 2, dim, 3, padding=1)
        )

        # === 7) 新增：FiLM 条件注入（由 prompt_vp_embeds 产生）===
        # 产生与通道数 dim 对齐的 (gamma, beta)，调制 mask prior 特征，注入视角/语义引导
        self.prompt_cond = nn.Linear(dim, dim * 2)

        # === 8) 上采样 decoder（保持结构，输入通道维持 dim）===
        self.mask_decoder = nn.Sequential(
            nn.Conv2d(dim, dim // 2, 3, padding=1),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(dim // 2, dim // 4, 4, stride=2, padding=1),  # ×2
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(dim // 4, dim // 8, 4, stride=2, padding=1),  # ×4
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(dim // 8, 1, 4, stride=2, padding=1)          # ×8 -> base_res*8 = out_size
        )

    def _normalize_mask(self, mask: torch.Tensor, ref_dtype: torch.dtype) -> torch.Tensor:
        if mask.dtype == torch.bool:
            mask = mask.to(dtype=ref_dtype)
        elif mask.dtype in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
            mask = (mask > 0).to(dtype=ref_dtype)
        else:
            mask = mask.to(dtype=ref_dtype)
        return mask.clamp(0, 1)

    def forward(self, prompt_vp_embeds, prompt_masks_tensor, vision_features):
        B = vision_features.size(0)
        D = vision_features.size(-1)
        H_W = vision_features.size(1)
        H = W = int(H_W ** 0.5)
        num_obj = prompt_vp_embeds.size(0) // B

        # === (1) 归一化 mask ===
        prompt_masks_tensor = self._normalize_mask(prompt_masks_tensor, vision_features.dtype)

        # === (2) 几何特征（与原始一致）===
        mask_feat = self.mask_encoder(prompt_masks_tensor.unsqueeze(1)).view(B * num_obj, 1, D)

        # === (3) 融合语义与几何（与原始一致）===
        fused_prompt = prompt_vp_embeds + mask_feat  # predict_vp_embeds 分支保持不变

        # ---------------------------------------------------------------------
        # A) predict_vp_embeds 分支 —— 保持原理与实现不变
        # ---------------------------------------------------------------------
        vision_features_view = vision_features.view(B, H * W, D)
        k = vision_features_view.unsqueeze(1).repeat(1, num_obj, 1, 1)
        v = k.clone()
        q = fused_prompt.view(B, num_obj, 1, D)

        q = self.query_proj(q)
        k = self.key_proj(k)
        v = self.value_proj(v)

        attn = torch.matmul(q, k.transpose(-1, -2)) / (D ** 0.5)
        attn = F.softmax(attn, dim=-1) if self.use_softmax else torch.sigmoid(attn)

        spatial_weights = self.spatial_gate(k).squeeze(-1).unsqueeze(2)
        attn = attn * spatial_weights
        attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-6)

        weighted_sum = torch.matmul(attn, v)
        x = weighted_sum.view(B * num_obj, 1, D)

        vision_flat = vision_features_view.repeat_interleave(num_obj, dim=0)
        for layer in self.cross_layers:
            x = layer(torch.cat([x, vision_flat.mean(1, keepdim=True)], dim=1))[:, 0:1, :]

        predict_vp_embeds = self.residual_mlp(torch.cat([x, fused_prompt], dim=-1))  # [B*num_obj, 1, D]

        # ---------------------------------------------------------------------
        # B) mask→mask 分支 —— 以 mask 为主体，注入语义 FiLM，完全不依赖 vision_features
        # ---------------------------------------------------------------------
        # 将输入 mask 直接下采样到 decoder 的起始分辨率（与 ×8 上采样对齐）
        mask_coarse = F.interpolate(
            prompt_masks_tensor.unsqueeze(1),
            size=(self.base_res, self.base_res),
            mode='bilinear',
            align_corners=False
        )  # [B*num_obj, 1, base_res, base_res]

        # 编码成 dim 通道的空间特征
        prior_feat = self.mask_prior_encoder(mask_coarse)  # [B*num_obj, dim, base_res, base_res]

        # 用 prompt_vp_embeds 产生 FiLM 参数 (gamma, beta) 注入视角提示
        gb = self.prompt_cond(prompt_vp_embeds.squeeze(1))  # [B*num_obj, 2*dim]
        gamma, beta = gb.chunk(2, dim=-1)                   # [B*num_obj, dim], [B*num_obj, dim]
        gamma = torch.tanh(gamma).unsqueeze(-1).unsqueeze(-1)  # 稳定训练
        beta = beta.unsqueeze(-1).unsqueeze(-1)

        # 额外将全局 mask_feat 注入（作为全局先验），提升几何稳定性
        global_mask_bias = mask_feat.squeeze(1).unsqueeze(-1).unsqueeze(-1)  # [B*num_obj, dim, 1, 1]

        # FiLM 调制 + 全局几何偏置（不引入 vision_features）
        fused_mask_feat = prior_feat * (1.0 + gamma) + beta + global_mask_bias

        # 上采样解码 -> 目标视角 mask
        pred_masks_tensor = torch.sigmoid(self.mask_decoder(fused_mask_feat))
        pred_masks_tensor = F.interpolate(
            pred_masks_tensor, size=(self.out_size, self.out_size),
            mode='bilinear', align_corners=False
        ).squeeze(1).reshape(B, num_obj, self.out_size, self.out_size)  # [B, num_obj, 256, 256]

        
        return predict_vp_embeds, pred_masks_tensor