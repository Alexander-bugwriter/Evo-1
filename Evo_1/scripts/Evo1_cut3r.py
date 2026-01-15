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
from scripts.cut3r_encoder_lyh import prepare_input

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
        #nn.init.constant_(self.out_proj.weight, 0.0)
        #nn.init.constant_(self.out_proj.bias, 0.0)
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
            from scripts.cut3r_encoder_lyh import CUT3REncoder
            
            cut3r_weights = "/opt/liblibai-models/user-workspace2/users/lyh/Evo-1/Evo_1/spatial_encoder_checkpoint/cut3r_512_dpt_4_64.pth"
            
            self.cut3r_encoder = CUT3REncoder(
                model_path=cut3r_weights,
                device=self._device
            )
            self.cut3r_encoder.model.eval()
            print(f"✅ Initialized new CUT3R Encoder from cut3r_encoder_lyh.py")
            
        d_vit=896
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
        #self.embedder = InternVL3Embedder(model_name=vlm_name, device=self._device,fusion_block=None)

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
        """
    使用新的 CUT3R Encoder 提取空间特征
    
    Args:
        images: List[PIL.Image] or List[torch.Tensor], 长度为3
                对应 [agentview, wrist, dummy_proc]
    
    Returns:
        spatial_tokens: [3, 730, 768]
                        camera_token (1) + patch_tokens (729) = 730
    """
        if self.cut3r_encoder is None:
            raise RuntimeError("CUT3R encoder not initialized. Set training=False in config.")
        #print(f"\n[DEBUG] Input images info:")
        #for idx, img in enumerate(images):
            #if isinstance(img, Image.Image):
                #print(f"  Image {idx}: PIL.Image, size={img.size}, mode={img.mode}")
            #elif isinstance(img, torch.Tensor):
                #print(f"  Image {idx}: Tensor, shape={img.shape}, dtype={img.dtype}, "
                    #f"range=[{img.min().item():.3f}, {img.max().item():.3f}]")
            #else:
                #print(f"  Image {idx}: {type(img)}")
        # ========== 1. 图像预处理 ==========
        processed = []
        for img in images:
        # 归一化到 [-1, 1]
            img = img * 2.0 - 1.0
            img = img.to(self._device, dtype=torch.bfloat16)
            processed.append(img)
        # ========== 2. 组织成 [F, B, C, H, W] 格式 ==========
        # F=1 (单帧), B=3 (3个视角)
        pixel_values = torch.stack(processed, dim=0).unsqueeze(0)  # [1, 3, C, H, W]
        #print(f"[_extract_spatial_features] Input shape: {pixel_values.shape}")
        # ========== 3. Prepare input for CUT3R ==========
        views = prepare_input(
            pixel_values=pixel_values,
            device=self._device,
            target_size=432  # CUT3R 的目标尺寸
        )
        
        # ========== 4. Forward through CUT3R ==========
        with torch.no_grad():
            results, camera_tokens, patch_tokens = self.cut3r_encoder.forward(views)
        
        # camera_tokens: [F*B, 1, 768] = [1*3, 1, 768] = [3, 1, 768]
        # patch_tokens: [F*B, 729, 768] = [3, 729, 768]
        
        #print(f"[_extract_spatial_features] camera_tokens: {camera_tokens.shape}")
        #print(f"[_extract_spatial_features] patch_tokens: {patch_tokens.shape}")
        
        # ========== 5. 拼接 camera 和 patch tokens ==========
        # 拼接后: [3, 730, 768]
        spatial_tokens = torch.cat([camera_tokens, patch_tokens], dim=1)
        
        #print(f"[_extract_spatial_features] Output spatial_tokens: {spatial_tokens.shape}")
        
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
        action_mask: Union[torch.Tensor, None] = None,
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
        #for param in self.embedder.model.vision_model.parameters():
        #    param.requires_grad = False
        #print("Frozen VIT (vision_model)") 
        #if not config.get("finetune_vlm", False):
        #    self._freeze_module(self.embedder, "VLM (InternVL3)")
        #else:
        #    print("Finetuning VLM (InternVL3)...")
        
         if not config.get("finetune_vit", False):
            self._freeze_module(self.embedder.model.vision_model, "VIT (vision_model)")
        else:
            print("Finetuning VIT (vision_model)...")
            
        if not config.get("finetune_llm_backbone", False):
            self._freeze_module(self.embedder.model.language_model, "LLM Backbone")
        else:
            print("Finetuning LLM Backbone...")


        if not config.get("finetune_action_head", False):
            self._freeze_module(self.action_head, "Action Head")
        else:
            print("Finetuning Action Head...")

        
        if not config.get("finetune_fusion_block", False):
            try:
                self._freeze_module(self.fusion_block, "Fusion Block")
            except:
                print("No_fusion_block")
        else:
            print("Finetuning Fusion Block...")
