import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from types import SimpleNamespace
from typing import List, Optional, Union, Tuple
from PIL import Image
import torch
import torch.nn as nn
from model.internvl3.internvl3_embedder_cut3r import InternVL3Embedder
from model.action_head.flow_matching import FlowmatchingActionHead
import logging

class CrossAttentionFusion(nn.Module):
    def __init__(self, d_clip, d_spatial_encoder, d_attn, num_heads):
        super(CrossAttentionFusion, self).__init__()
        
        # pre-norm
        self.clip_norm = nn.LayerNorm(d_clip)
        self.spatial_encoder_norm = nn.LayerNorm(d_spatial_encoder)
        # projection
        self.clip_query_proj = nn.Linear(d_clip, d_attn)
        self.spatial_encoder_key_proj = nn.Linear(d_spatial_encoder, d_attn)
        self.spatial_encoder_value_proj = nn.Linear(d_spatial_encoder, d_attn)
        # cross attention
        self.cross_attention = nn.MultiheadAttention(embed_dim=d_attn, num_heads=num_heads, batch_first=True)
        # post-norm
        self.out_norm = nn.LayerNorm(d_attn)
        # projection
        self.out_proj = nn.Linear(d_attn, d_clip)
        # dropout
        self.dropout = nn.Dropout(0.1)
        # 在 CrossAttentionFusion.__init__ 最后加这两行
        nn.init.constant_(self.out_proj.weight, 0.0)
        nn.init.constant_(self.out_proj.bias, 0.0)
    def forward(self, clip_features, spatial_encoder_features):
        """
        Args:
            clip_features: [B, N, D_clip]
            spatial_encoder_features: [B, N, D_spatial_encoder]
        Returns:
            fused_features: [B, N, D_clip]
        """
        # pre-norm
        clip_features_norm = self.clip_norm(clip_features)  # [B, N, D_clip]
        spatial_encoder_features_norm = self.spatial_encoder_norm(spatial_encoder_features)  # [B, N, D_spatial_encoder]
        # projection to D_attn dimension
        clip_query_proj = self.clip_query_proj(clip_features_norm)  # [B, N, D_attn]
        spatial_encoder_key_proj = self.spatial_encoder_key_proj(spatial_encoder_features_norm)  # [B, N, D_attn]
        spatial_encoder_value_proj = self.spatial_encoder_value_proj(spatial_encoder_features_norm)  # [B, N, D_attn]
        # cross attention
        fused_features, attn_weights = self.cross_attention(
            query=clip_query_proj,
            key=spatial_encoder_key_proj,
            value=spatial_encoder_value_proj
        )
        # projection to D_clip dimension
        fused_features = self.out_proj(fused_features)   # [B, N_clip, D_clip]
        # residual connection and dropout
        fused_features = self.out_norm(fused_features)
        fused_features = fused_features + clip_features  # [B, N_clip, D_clip]
        # print(f'status_of_fused_features: max:{fused_features.max():.2f}, min:{fused_features.min():.2f}, mean:{fused_features.mean():.2f}, std:{fused_features.std():.2f}')
        # print(f'status_of_clip_features: max:{clip_features.max():.2f}, min:{clip_features.min():.2f}, mean:{clip_features.mean():.2f}, std:{clip_features.std():.2f}')
        fused_features = self.dropout(fused_features)
        
        return fused_features, attn_weights

class EVO1(nn.Module):
    def __init__(self, config: dict):
        super().__init__() 
        self.config = config
        self._device = config.get("device", "cuda")
        self.return_cls_only = config.get("return_cls_only", False)
        vlm_name = config.get("vlm_name", "OpenGVLab/InternVL3-1B")
        self.use_cut3r = config.get("use_cut3r", False)
        is_training = config.get("training", True)
        if self.use_cut3r and not is_training:
            from scripts.cut3r_spatial_encoder import (
                Cut3rSpatialTower, Cut3rSpatialConfig
            )
            
            cut3r_config = Cut3rSpatialConfig(
                weights_path="/opt/liblibai-models/user-workspace2/users/lyh/lyh_openpi_train/src/openpi/models_pytorch/spatial_encoder_checkpoint/cut3r_512_dpt_4_64.pth",
                spatial_tower_select_feature="all",
                spatial_tower_select_layer=-1,
                export_point_cloud=False
            )
            
            self.spatial_tower = Cut3rSpatialTower(
                spatial_tower='cut3r',
                spatial_tower_cfg=cut3r_config,
                delay_load=False
            )
            self.spatial_tower.to(device=self._device, dtype=torch.float16)
            self.spatial_tower.reset_state()
            print(f"✅ Initialized CUT3R Spatial Tower")
            
        d_vit = 896  # InternVL3-1B的VIT维度
        d_spatial = 768  # CUT3R输出维度
        
        self.fusion_block = CrossAttentionFusion(
            d_clip=d_vit,
            d_spatial_encoder=d_spatial,
            d_attn=d_vit,
            num_heads=16
        )
        self.fusion_block.to(device=self._device, dtype=torch.bfloat16)
        print(f"✅ Initialized CrossAttentionFusion")
        self.embedder = InternVL3Embedder(model_name=vlm_name, device=self._device,fusion_block=self.fusion_block)

        action_head_type = config.get("action_head", "flowmatching").lower()
        
        if action_head_type == "flowmatching":
           
            horizon = config.get("action_horizon", config.get("horizon", 16))
            per_action_dim = config.get("per_action_dim", 7)
            action_dim = horizon * per_action_dim
            
            config["horizon"] = horizon
            config["per_action_dim"] = per_action_dim
            config["action_dim"] = action_dim
            
            if action_dim != horizon * per_action_dim:
                raise ValueError(f"action_dim ({action_dim}) ≠ horizon ({horizon}) × per_action_dim ({per_action_dim})")
            
            self.horizon = horizon
            self.per_action_dim = per_action_dim
            
            self.action_head = FlowmatchingActionHead(config=SimpleNamespace(
                embed_dim=config.get("embed_dim", 896),    
                hidden_dim=config.get("hidden_dim", 1024),
                action_dim=action_dim,
                horizon=horizon,
                per_action_dim=per_action_dim,
                state_dim=config.get("state_dim", 7),
                state_hidden_dim=config.get("state_hidden_dim", 1024),
                num_heads=config.get("num_heads", 8),
                num_layers=config.get("num_layers", 8),
                dropout=config.get("dropout", 0.0),
                num_inference_timesteps=config.get("num_inference_timesteps", 50),
                num_categories=config.get("num_categories", 1)
            )).to(self._device)
        else:
            raise NotImplementedError(f"Unknown action_head: {action_head_type}")

        
    def _extract_spatial_features(self, images):
        if self.spatial_tower is None:
            raise RuntimeError("spatial_tower not initialized. Set training=False in config for inference.")
        
        processed = []
        for img in images:
            if isinstance(img, Image.Image):
                img = T.ToTensor()(img)
            if img.dtype != torch.uint8:
                img = (img * 255).clamp(0, 255).to(torch.uint8)
            img = img.to(self._device, dtype=torch.float16) / 127.5 - 1.0
            processed.append(img)
        
        batch = torch.stack(processed).unsqueeze(0)  # [1, N, C, H, W]
        
        with torch.no_grad():
            camera_tokens, patch_tokens = self.spatial_tower(batch)
            # camera_tokens: [N=3, 1, 768]
            # patch_tokens: [N=3,729, 768]
           
            # 🔥 拼接 camera 和 patch
            spatial_tokens = torch.cat([camera_tokens, patch_tokens], dim=1)  # [N, 730, 768]
        
        return spatial_tokens.to(dtype=torch.bfloat16)  
        
    def get_vl_embeddings(
        self,
        images: List[Image.Image],
        image_mask: torch.Tensor,  
        prompt: str = "",
        return_cls_only: Union[bool, None] = None,
        spatial_tokens: Optional[torch.Tensor] = None
    ) -> torch.Tensor:

        if return_cls_only is None:
            return_cls_only = self.return_cls_only
        
        # ========== 智能获取 spatial_tokens ==========
        if self.use_cut3r:
            if spatial_tokens is None:
                # 外界没有传入，实时提取
                spatial_tokens = self._extract_spatial_features(images)
            # else: 使用外界传入的 spatial_tokens
            else:
                spatial_tokens = spatial_tokens.to(dtype=torch.bfloat16)
        else:
            spatial_tokens = None  # 不使用 CUT3R

        if images is None or len(images) == 0:
            raise ValueError("Must provide at least one image (PIL.Image). Got `images=None` or empty list.")
        return self.embedder.get_fused_image_text_embedding_from_tensor_images(
            image_tensors=images,
            image_mask=image_mask,
            text_prompt=prompt,
            return_cls_only=return_cls_only,
            spatial_tokens=spatial_tokens,
        )

    def prepare_state(self, state_input: Union[list, torch.Tensor]) -> torch.Tensor:

        if isinstance(state_input, list):
            state_tensor = torch.tensor(state_input)
        elif isinstance(state_input, torch.Tensor):
            state_tensor = state_input
        else:
            raise TypeError("Unsupported state input type")

        if state_tensor.ndim == 1:
            state_tensor = state_tensor.unsqueeze(0)

        return state_tensor.to(self._device)

    
    def predict_action(
        self,
        fused_tokens: torch.Tensor,
        state: torch.Tensor,
        actions_gt: torch.Tensor = None,
        action_mask: torch.Tensor = None,
        embodiment_ids: torch.Tensor = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        
        if actions_gt is None:
            return self.action_head.get_action(fused_tokens, state=state, action_mask=action_mask, embodiment_id=embodiment_ids)
        else:
            return self.action_head(fused_tokens, state=state, actions_gt=actions_gt, action_mask=action_mask, embodiment_id=embodiment_ids)


    @torch.no_grad()
    def run_inference(
        self,
        images: List[Union[Image.Image, torch.Tensor]],
        image_mask: torch.Tensor,
        prompt: str,
        state_input: Union[list, torch.Tensor],
        return_cls_only: Union[bool, None] = None,
        action_mask: Union[torch.Tensor, None] = None
    ) -> torch.Tensor:

        fused_tokens = self.get_vl_embeddings(
                        images=images,
                        image_mask=image_mask,
                        prompt=prompt,
                        return_cls_only=return_cls_only
                        
                    )

        state_tensor = self.prepare_state(state_input)  
        
        return self.predict_action(fused_tokens, state_tensor, action_mask=action_mask)
    

    def forward(self, fused_tokens, state=None, actions_gt=None, action_mask=None, embodiment_ids=None):
   

        return self.predict_action(fused_tokens, state, actions_gt, action_mask, embodiment_ids)

    def _freeze_module(self, module: nn.Module, name: str):
        print(f"Freezing {name} parameters...")
        for p in module.parameters():
            p.requires_grad = False

    def set_finetune_flags(self):
        config = self.config  
        if not config.get("finetune_vlm", False):
            self._freeze_module(self.embedder, "VLM (InternVL3)")
            print("Freezing VLM (InternVL3)...")
        else:
            print("Finetuning VLM (InternVL3)...")

        if not config.get("finetune_action_head", False):
            self._freeze_module(self.action_head, "Action Head")
            print("Freezing Action Head...")
        else:
            print("Finetuning Action Head...")

        
        if not config.get("finetune_fusion_block", False):
            self._freeze_module(self.fusion_block, "Fusion Block")
            print("Freezing Fusion Block...")
        else:
            print("Finetuning Fusion Block...")
