# dataset/lerobot_dataset_cut3r.py
# dataset/lerobot_dataset_cut3r.py
"""
Libero Parquet Dataset for CUT3R
单一数据集，从parquet直接读取PNG bytes图片和flatten的spatial tokens
"""

import os
import torch
import random
import json
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from tqdm.auto import tqdm
from typing import Union
from torch.utils.data import Dataset
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
import multiprocessing as mp
import logging
import pickle

logging.basicConfig(level=logging.INFO)


def decode_image_from_bytes(image_data):
    """从PNG字节流解码图像"""
    if isinstance(image_data, dict) and 'bytes' in image_data:
        import io
        img = Image.open(io.BytesIO(image_data['bytes']))
        return np.array(img)
    elif isinstance(image_data, np.ndarray):
        return image_data
    else:
        raise ValueError(f"Unsupported image format: {type(image_data)}")


def _process_parquet_file_worker(args):
    """多进程worker：处理单个parquet文件"""
    parquet_path, action_horizon, max_samples_per_file, cache_dir,task_mapping = args
    
    try:
        df = pd.read_parquet(parquet_path)
        
        # 末尾padding
        last_row = df.iloc[-1:]
        padding_rows = pd.concat([last_row] * action_horizon, ignore_index=True)
        df = pd.concat([df, padding_rows], ignore_index=True)

        if max_samples_per_file is not None:
            df = df.head(max_samples_per_file)

        episode_files = []
        for i in range(len(df) - action_horizon + 1):
            start_idx = i
            end_idx = i + action_horizon
            
            cache_subdir = cache_dir / parquet_path.parent.name / parquet_path.stem
            cache_filename = f"{start_idx}_{end_idx}.pkl"
            cache_filepath = cache_subdir / cache_filename
            
            if cache_filepath.exists():
                episode_files.append(str(cache_filepath))
                continue
            
            sub_df = df.iloc[i: i + action_horizon]

             # 🔥 修复：从 task_index 查找 prompt
            task_index = sub_df.iloc[0].get("task_index", None)
            if task_index is not None and task_index in task_mapping:
                prompt = task_mapping[task_index]
            else:
                prompt = ""
                if task_index is not None:
                    logging.warning(f"Task index {task_index} not found in task mapping")
            
            episode = {
                "prompt": prompt,
                "state": sub_df.iloc[0].get("state", None),
                "action": [row["actions"] for _, row in sub_df.iterrows()],
                "image": sub_df.iloc[0].get("image", None),
                "wrist_image": sub_df.iloc[0].get("wrist_image", None),
                "episode_index": int(sub_df.iloc[0].get("episode_index", 0)),
                "frame_index": int(sub_df.iloc[0].get("frame_index", 0)),
                # Flatten的spatial tokens
                "base_camera_tokens_flat": sub_df.iloc[0].get("base_camera_tokens_flat", None),
                "base_patch_tokens_flat": sub_df.iloc[0].get("base_patch_tokens_flat", None),
                "wrist_camera_tokens_flat": sub_df.iloc[0].get("wrist_camera_tokens_flat", None),
                "wrist_patch_tokens_flat": sub_df.iloc[0].get("wrist_patch_tokens_flat", None),
            }
            
            cache_subdir.mkdir(parents=True, exist_ok=True)
            with open(cache_filepath, 'wb') as f:
                pickle.dump(episode, f)
            
            episode_files.append(str(cache_filepath))
            
        return episode_files, None
        
    except Exception as e:
        error_msg = f"Error processing file {parquet_path}: {str(e)}"
        logging.error(error_msg)
        return [], error_msg


class LeRobotDatasetCut3r(Dataset):
    """
    Libero Parquet Dataset for CUT3R
    单一数据集，从parquet直接读取PNG bytes图片和flatten的spatial tokens
    """
    
    def __init__(
        self,
        image_size: int = 448,
        max_samples_per_file: Union[int, None] = None,
        action_horizon: int = 50,
        cache_dir: Union[str, None] = None,
        use_augmentation: bool = False,
        binarize_gripper: bool = False,  # 保持和原版一致，虽然不用
    ):
        # 硬编码数据集路径
        self.dataset_root = Path("/opt/liblibai-models/user-workspace2/dataset/libero_dataset")
        
        # 硬编码max维度（和原始lerobot_dataset_pretrain_mp保持一致）
        self.max_action_dim = 24
        self.max_state_dim = 24
        self.max_views = 3
        
        self.image_size = image_size
        self.max_samples_per_file = max_samples_per_file
        self.use_augmentation = use_augmentation
        self.action_horizon = action_horizon
        self.binarize_gripper = binarize_gripper

        if cache_dir is None:
            self.cache_dir = self.dataset_root / ".cache"
        else:
            self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.data = []

        self._load_task_mapping()

        # 加载归一化统计
        self._load_norm_stats()
        self.norm_stats = {
            "state": {
                "min": self.state_min.tolist(), 
                "max": self.state_max.tolist()
            },
            "actions": {
                "min": self.action_min.tolist(), 
                "max": self.action_max.tolist()
            }
        }
        
        # 加载数据
        self._load_trajectories()

        # 图像预处理
        self.basic_transform = T.Compose([
            T.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor()
        ])

        self.aug_transform = T.Compose([
            T.RandomResizedCrop(448, scale=(0.95, 1.0), interpolation=InterpolationMode.BICUBIC),
            T.RandomRotation(degrees=(-5, 5), interpolation=InterpolationMode.BICUBIC),
            T.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5, hue=0.08),
            T.ToTensor()
        ])
    def _load_task_mapping(self):
        """加载任务描述映射"""
        tasks_path = self.dataset_root / "meta" / "tasks.jsonl"
        
        if not tasks_path.exists():
            print(f"⚠️  Tasks file not found: {tasks_path}")
            print(f"   Prompts will be empty")
            self.task_mapping = {}
            return
        
        print(f"📖 Loading task mapping from: {tasks_path}")
        
        dataset_tasks = pd.read_json(tasks_path, lines=True).to_dict("records")
        self.task_mapping = {
            task_obj["task_index"]: task_obj["task"]
            for task_obj in dataset_tasks
            if "task_index" in task_obj and "task" in task_obj
        }
        
        print(f"✅ Loaded {len(self.task_mapping)} task descriptions")
    def _load_norm_stats(self):
        """加载归一化统计"""
        stats_path = self.dataset_root / "meta" / "stats.json"
        
        if not stats_path.exists():
            raise FileNotFoundError(f"Stats file not found: {stats_path}")
        
        print(f"Loading stats from: {stats_path}")
        with open(stats_path, "r") as f:
            stats = json.load(f)
        
        self.state_min = torch.tensor(stats["state"]["min"], dtype=torch.float32)
        self.state_max = torch.tensor(stats["state"]["max"], dtype=torch.float32)
        self.action_min = torch.tensor(stats["actions"]["min"], dtype=torch.float32)
        self.action_max = torch.tensor(stats["actions"]["max"], dtype=torch.float32)

    def _load_trajectories(self):
        """多进程加载parquet文件"""
        data_dir = self.dataset_root / "data"
        parquet_files = list(data_dir.glob("*/*.parquet"))
        
        print(f"Found {len(parquet_files)} parquet files")
        
        parquet_process_units = [
            (pf, self.action_horizon, self.max_samples_per_file, self.cache_dir)
            for pf in parquet_files
        ]

        # 🔥 修复：传递 task_mapping
        parquet_process_units = [
            (pf, self.action_horizon, self.max_samples_per_file, self.cache_dir, self.task_mapping)
            for pf in parquet_files
        ]
        
        num_processes = min(16, len(parquet_process_units))
        print(f"Using {num_processes} processes")
        
        with mp.Pool(processes=num_processes) as pool:
            total_episodes = 0
            with tqdm(total=len(parquet_process_units), desc="Processing Parquet files") as pbar:
                for episode_files, error in pool.imap_unordered(_process_parquet_file_worker, parquet_process_units):
                    if error:
                        logging.error(error)
                    else:
                        self.data.extend(episode_files)
                        total_episodes += len(episode_files)
                    
                    pbar.set_postfix({'total': total_episodes})
                    pbar.update(1)
        
        print(f"Data loading completed: {len(self.data)} samples")

    def _pad_tensor(self, source_tensor: torch.Tensor, max_dim: int) -> (torch.Tensor, torch.Tensor):
        """Padding tensor到最大维度（和原始lerobot_dataset_pretrain_mp一致）"""
        source_dim = source_tensor.shape[-1]
        
        if source_tensor.dim() > 1:
            padded_shape = (*source_tensor.shape[:-1], max_dim)
        else:
            padded_shape = (max_dim,)

        padded_tensor = torch.zeros(padded_shape, dtype=source_tensor.dtype, device=source_tensor.device)
        mask = torch.zeros(padded_shape, dtype=torch.bool, device=source_tensor.device)

        data_slice = (..., slice(0, source_dim))
        
        padded_tensor[data_slice] = source_tensor
        mask[data_slice] = True
            
        return padded_tensor, mask

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        cache_filepath = self.data[idx]
        
        try:
            with open(cache_filepath, 'rb') as f:
                item = pickle.load(f)
        except Exception as e:
            logging.info(f"Cannot load cache {cache_filepath}: {str(e)}")
            return self[random.randint(0, len(self.data)-1)]

        # ========== 从parquet读取PNG bytes图像 ==========
        try:
            base_img = decode_image_from_bytes(item["image"])
            wrist_img = decode_image_from_bytes(item["wrist_image"])
            
            base_img = Image.fromarray(base_img)
            wrist_img = Image.fromarray(wrist_img)
            
            if self.use_augmentation:
                images = [
                    self.aug_transform(base_img) if random.random() < 0.5 else self.basic_transform(base_img),
                    self.aug_transform(wrist_img) if random.random() < 0.5 else self.basic_transform(wrist_img)
                ]
            else:
                images = [self.basic_transform(base_img), self.basic_transform(wrist_img)]
            
        except Exception as e:
            logging.info(f"Failed to decode images: {e}")
            return self[random.randint(0, len(self.data)-1)]

        # Padding到max_views（和原始lerobot_dataset_pretrain_mp一致）
        num_real_views = len(images)
        image_mask = torch.zeros(self.max_views, dtype=torch.bool)
        image_mask[:num_real_views] = True

        while len(images) < self.max_views:
            if len(images) == 0:
                dummy_image = torch.zeros(3, 448, 448)
                logging.info("Warning: Image list is empty, using zero tensor for padding")
            else:
                dummy_image = torch.zeros_like(images[0]) 
            # dummy_image = torch.zeros_like(images[0]) if len(images) > 0 else torch.zeros(3, 448, 448)
            images.append(dummy_image)

        images = torch.stack(images)  # [3, 3, 448, 448]

        # ========== 读取flatten的spatial tokens并reshape ==========
        # camera tokens: (768,) -> (1, 768)
        # patch tokens: (559872,) -> (729, 768)
        
        base_camera_flat = item["base_camera_tokens_flat"]
        base_patch_flat = item["base_patch_tokens_flat"]
        wrist_camera_flat = item["wrist_camera_tokens_flat"]
        wrist_patch_flat = item["wrist_patch_tokens_flat"]
        
        # Reshape
        base_camera_tokens = torch.from_numpy(base_camera_flat).reshape(1, 768)    # [1, 768]
        base_patch_tokens = torch.from_numpy(base_patch_flat).reshape(729, 768)    # [729, 768]
        wrist_camera_tokens = torch.from_numpy(wrist_camera_flat).reshape(1, 768)  # [1, 768]
        wrist_patch_tokens = torch.from_numpy(wrist_patch_flat).reshape(729, 768)  # [729, 768]
        
        # 对每个相机：[camera_token, patch_token] concat
        base_spatial = torch.cat([base_camera_tokens, base_patch_tokens], dim=0)     # [730, 768]
        wrist_spatial = torch.cat([wrist_camera_tokens, wrist_patch_tokens], dim=0)  # [730, 768]
        
        # 创建spatial tokens列表
        spatial_tokens_list = [base_spatial, wrist_spatial]  # 2个相机
        
        # Padding到max_views（和图像一样）
        while len(spatial_tokens_list) < self.max_views:
            if len(images) == 0:
                dummy_spatial_token = torch.zeros(3, 730, 768)
                logging.info("Warning: Spatial token list is empty, using zero tensor for padding")
            else:
                dummy_spatial_token = torch.zeros_like(spatial_tokens_list[0])
            spatial_tokens_list.append(dummy_spatial_token)
        
        spatial_tokens = torch.stack(spatial_tokens_list)  # [3, 730, 768]

        # ========== State和Action归一化（和原始lerobot_dataset_pretrain_mp一致）==========
        if item["state"] is None:
            raise ValueError("Missing observation.state")
        
        state = torch.tensor(item["state"], dtype=torch.float32)
        state_min = self.state_min.to(device=state.device, dtype=state.dtype)
        state_max = self.state_max.to(device=state.device, dtype=state.dtype)
        state = 2 * (state - state_min) / (state_max - state_min + 1e-8) - 1
        state = torch.clamp(state, -1.0, 1.0)

        state_padded, state_mask = self._pad_tensor(state, self.max_state_dim)

        if item["action"] is None:
            raise ValueError("Missing action")

        action = torch.from_numpy(np.stack(item["action"])).float()
        action_min = self.action_min.to(device=action.device, dtype=action.dtype)
        action_max = self.action_max.to(device=action.device, dtype=action.dtype)
        action = 2 * (action - action_min.unsqueeze(0)) / (action_max.unsqueeze(0) - action_min.unsqueeze(0) + 1e-8) - 1
        action = torch.clamp(action, -1.0, 1.0)

        action_padded, action_mask = self._pad_tensor(action, self.max_action_dim)

        prompt = item["prompt"] if item["prompt"] else ""
        
        return {
            "images": images,  # [3, 3, 448, 448]
            "image_mask": image_mask,  # [3]
            "prompt": prompt,
            "state": state_padded.to(dtype=torch.bfloat16),  # [24]
            "state_mask": state_mask,  # [24]
            "action": action_padded.to(dtype=torch.bfloat16),  # [horizon, 24]
            "action_mask": action_mask,  # [horizon, 24]
            "embodiment_id": torch.tensor(0, dtype=torch.long),  # 单一数据集，固定为0
            "spatial_tokens": spatial_tokens  # [3, 730, 768]
        }
