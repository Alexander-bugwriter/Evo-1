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
from scripts.cut3r_spatial_encoder import (
    Cut3rSpatialTower, Cut3rSpatialConfig
    )
import os

weights_path = "/opt/liblibai-models/user-workspace2/users/lyh/lyh_openpi_train/src/openpi/models_pytorch/spatial_encoder_checkpoint/cut3r_512_dpt_4_64.pth"

print(f"🔍 Checking weights file:")
print(f"   Path: {weights_path}")
print(f"   os.path.exists: {os.path.exists(weights_path)}")
print(f"   os.path.isfile: {os.path.isfile(weights_path)}")
print(f"   os.path.abspath: {os.path.abspath(weights_path)}")

cut3r_config = Cut3rSpatialConfig(
    weights_path="/opt/liblibai-models/user-workspace2/users/lyh/lyh_openpi_train/src/openpi/models_pytorch/spatial_encoder_checkpoint/cut3r_512_dpt_4_64.pth",
    spatial_tower_select_feature="all",
    spatial_tower_select_layer=-1,
    export_point_cloud=False
)

spatial_tower = Cut3rSpatialTower(
    spatial_tower='cut3r',
    spatial_tower_cfg=cut3r_config,
    delay_load=False
    )
spatial_tower.to(device="cuda:4", dtype=torch.float16)
spatial_tower.reset_state()
print(f"✅ Initialized CUT3R Spatial Tower")
dummy_image1 = torch.empty(1, 3, 3, 224, 224, dtype=torch.float16).uniform_(-1, 1)
dummy_camera_token1,dummy_patch_token1=spatial_tower(dummy_image1)
print(f"\n--- First Forward Pass ---")
print(f"dummy_image1 shape: {dummy_image1.shape}")
print(f"dummy_camera_token1 shape: {dummy_camera_token1.shape}")
print(f"dummy_patch_token1 shape: {dummy_patch_token1.shape}")
dummy_image2 = torch.empty(1, 3, 3, 224, 224, dtype=torch.float16).uniform_(-1, 1)
dummy_camera_token2,dummy_patch_token2=spatial_tower(dummy_image2)
print(f"\n--- Second Forward Pass ---")
print(f"dummy_image2 shape: {dummy_image2.shape}")
print(f"dummy_camera_token2 shape: {dummy_camera_token2.shape}")
print(f"dummy_patch_token2 shape: {dummy_patch_token2.shape}")

spatial_tower.reset_state()
print("Test end. Successfully reset.")

