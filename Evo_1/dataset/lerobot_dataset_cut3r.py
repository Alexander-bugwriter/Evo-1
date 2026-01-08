# LeRobot Dataset with CUT3R Spatial Tokens and Pointcloud Support
# Based on lerobot_dataset_pretrain_mp.py with CUT3R extensions

import os
import torch
import random
import json
import numpy as np
import pandas as pd
import h5py
from PIL import Image
from pathlib import Path
from tqdm.auto import tqdm  
from typing import List, Union, Dict, Any
from torch.utils.data import Dataset
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
import multiprocessing as mp
import logging
import pickle

try:
    import open3d as o3d
    _OPEN3D_AVAILABLE = True
except ImportError:
    logging.warning("⚠️ open3d not available, pointcloud loading will fail")
    _OPEN3D_AVAILABLE = False

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def compute_lerobot_normalization_stats_from_minmax(jsonl_path):
    state_mins, state_maxs = [], []
    action_mins, action_maxs = [], []

    with open(jsonl_path, "r") as f:
        for line in tqdm(f, desc="Extracting min/max"):
            obj = json.loads(line)
            stats = obj.get("stats", {})
            try:
                state_mins.append(stats["observation.state"]["min"])
                state_maxs.append(stats["observation.state"]["max"])
                action_mins.append(stats["action"]["min"])
                action_maxs.append(stats["action"]["max"])
            except Exception as e:
                print(f"skipping abnormal line: {e}")

    state_min_global = np.min(np.array(state_mins), axis=0).tolist()
    state_max_global = np.max(np.array(state_maxs), axis=0).tolist()
    action_min_global = np.min(np.array(action_mins), axis=0).tolist()
    action_max_global = np.max(np.array(action_maxs), axis=0).tolist()

    return {
        "observation.state": {"min": state_min_global, "max": state_max_global},
        "action": {"min": action_min_global, "max": action_max_global}
    }

def merge_lerobot_stats(stats_list: List[Dict[str, Dict[str, List[float]]]]) -> Dict:
    state_mins = [np.array(d["observation.state"]["min"]) for d in stats_list]
    state_maxs = [np.array(d["observation.state"]["max"]) for d in stats_list]
    action_mins = [np.array(d["action"]["min"]) for d in stats_list]
    action_maxs = [np.array(d["action"]["max"]) for d in stats_list]
    state_min_global = np.min(np.stack(state_mins), axis=0).tolist()
    state_max_global = np.max(np.stack(state_maxs), axis=0).tolist()
    action_min_global = np.min(np.stack(action_mins), axis=0).tolist()
    action_max_global = np.max(np.stack(action_maxs), axis=0).tolist()

    return {
        "observation.state": {"min": state_min_global, "max": state_max_global},
        "action": {"min": action_min_global, "max": action_max_global}
    }


def _process_parquet_file_worker(args):
    """
    Worker function for processing parquet files
    🔥 Modified: Also records H5 file path and total episode length
    """
    parquet_path, arm_name, dataset_name, dataset_config, dataset_path, task_mapping, action_horizon, max_samples_per_file, cache_dir = args
    
    try:
        view_map = dataset_config.get('view_map', None)
        if not view_map:
            logging.info(f"did not find view_map for '{arm_name}-{dataset_name}', use default mapping")
            default_keys = ["image_1", "image_2", "image_3"]
            view_map = {key: f"observation.images.{key}" for key in default_keys}

        df = pd.read_parquet(parquet_path)

        # 🔥 记录原始 episode 长度（用于检查点云边界）
        original_episode_length = len(df)

        last_row = df.iloc[-1:]  
        padding_rows = pd.concat([last_row] * action_horizon, ignore_index=True)
        df = pd.concat([df, padding_rows], ignore_index=True)

        if max_samples_per_file is not None:
            df = df.head(max_samples_per_file)

        # 🔥 构建 CUT3R H5 文件路径
        chunk_name = parquet_path.parent.name
        episode_stem = parquet_path.stem
        h5_filename = f"{chunk_name}_{episode_stem}.h5"
        cut3r_h5_path = dataset_path / "cut3r_feature" / h5_filename
        
        # 检查 H5 文件是否存在
        has_cut3r = cut3r_h5_path.exists()
        if not has_cut3r:
            logging.warning(f"CUT3R H5 file not found: {cut3r_h5_path}")

        episode_files = []
        for i in range(len(df) - action_horizon + 1): 
            start_idx = i
            end_idx = i + action_horizon
            
            cache_subdir = cache_dir / arm_name / dataset_name / parquet_path.parent.name / parquet_path.stem
            cache_filename = f"{start_idx}_{end_idx}.pkl"
            cache_filepath = cache_subdir / cache_filename
            
            if cache_filepath.exists():
                episode_files.append(str(cache_filepath))
                continue
            
            logging.info(f"build {cache_filename}")
            sub_df = df.iloc[i: i + action_horizon]
            video_paths = {}
            base_video_path = dataset_path / "videos" / parquet_path.parent.name

            for view_key, view_folder in view_map.items():
                full_path = base_video_path / view_folder / f"{parquet_path.stem}.mp4"
                if full_path.exists():
                    video_paths[view_key] = str(full_path)
                else:
                    logging.warning(f"missing video file: {full_path}")
            
            task_index = sub_df.iloc[0].get("task_index", None)
            if task_index is not None and task_index in task_mapping:
                prompt = task_mapping[task_index]
            else:
                logging.info(f"cannot find task description from task_index={task_index}")
                prompt = ""

            episode = {
                "arm_key": arm_name,
                "dataset_key": dataset_name,
                "prompt": prompt,
                "state": sub_df.iloc[0].get("observation.state", None),
                "action": [row["action"] for _, row in sub_df.iterrows()],
                "video_paths": video_paths,
                "timestamp": sub_df.iloc[0].get("timestamp", None),
                "frame_index": int(sub_df.iloc[0].get("frame_index", 0)),
                # 🔥 新增字段
                "cut3r_h5_path": str(cut3r_h5_path) if has_cut3r else None,
                "episode_length": original_episode_length,  # 原始 episode 长度
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


class LeRobotDatasetCUT3R(Dataset):
    """
    LeRobot Dataset with CUT3R Spatial Tokens and Future Scene Support
    
    Loads:
    - Spatial tokens: [max_views, 730, 768] - Image features
    - Future scenes: [future_horizon, H, W, (3+3+1)] - 3D structure + RGB + Confidence
    
    The dense scene representation enables:
    - 3D scene reconstruction
    - 4D (3D + time) scene understanding
    - Novel view synthesis
    - Spatial reasoning for robotic tasks
    """
    def __init__(
        self,
        config: Dict[str, Any],
        image_size: int = 448,
        max_samples_per_file: Union[int, None] = None,
        video_backend: str = "av",
        action_horizon: int = 50,
        future_horizon: int = 20,  # 🔥 未来场景序列长度
        video_backend_kwargs: Dict[str, Any] = None,
        binarize_gripper: bool = False,
        cache_dir: Union[str, Path] = None,  
        use_augmentation: bool = False,
    ):
        self.config = config

        sorted_datasets = sorted(self.config['data_groups'].keys())
        self.arm_to_embodiment_id = {key: i for i, key in enumerate(sorted_datasets)}

        self.max_action_dim = config['max_action_dim']
        self.max_state_dim = config['max_state_dim']
        self.max_views = config['max_views']

        self.image_size = image_size
        self.max_samples_per_file = max_samples_per_file
        self.binarize_gripper = binarize_gripper
        self.use_augmentation = use_augmentation

        if cache_dir is None:
            self.cache_dir = Path("/opt/liblibai-models/user-workspace2/users/lyh/Evo-1/Evo_1/training_cut3r_data_cache")
        else:
            self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.data = []  
        self.arm2stats_dict = {}
        self.action_horizon = action_horizon
        self.video_backend = video_backend
        self.video_backend_kwargs = video_backend_kwargs or {}  
        self.future_horizon = future_horizon  # 🔥 保存未来场景序列长度

        if self.video_backend == "decord" and not self.video_backend_kwargs:
            self.video_backend_kwargs = {"ctx": "cpu"}  

        self._load_metadata()
        self._load_trajectories()

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

    def _load_metadata(self):
        """加载元数据（与原版相同）"""
        self.episodes = []
        self.tasks = {}
        norm_stats_list = []

        for arm_name, arm_config in self.config['data_groups'].items():
            print(f"  -- Processing arm group: '{arm_name}'")

            norm_arm_list = []
            self.tasks[arm_name] = {}
            for dataset_name, dataset_config in arm_config.items():
                print(f"    -- Processing dataset: '{dataset_name}'")
                print(f"    -- Dataset config: {dataset_config}")
                dataset_tasks = []
                path_str = dataset_config['path']
                dataset_path = Path(path_str)
                tasks_path = dataset_path / "meta" / "tasks.jsonl"
                if tasks_path.exists():
                    dataset_tasks = pd.read_json(tasks_path, lines=True).to_dict("records")
                    task_index_to_task = {
                        task_obj["task_index"]: task_obj["task"]
                        for task_obj in dataset_tasks
                        if "task_index" in task_obj and "task" in task_obj
                    }
                    self.tasks[arm_name][dataset_name] = task_index_to_task
                else:
                    raise FileNotFoundError(f"tasks file not found: {tasks_path}")
                
                episodes_path = dataset_path / "meta" / "episodes.jsonl"
                if episodes_path.exists():
                    self.episodes += pd.read_json(episodes_path, lines=True).to_dict("records")

                stats_path = dataset_path / "meta" / "episodes_stats.jsonl"
                stats_path_after_compute = dataset_path / "meta" / "stats.json"
                if stats_path_after_compute.exists():
                    print(f"already have stats file: {stats_path_after_compute}")
                    with open(stats_path_after_compute, "r") as f:
                        stats = json.load(f)
                    norm_arm_list.append(stats)
                elif stats_path.exists():
                    stats = compute_lerobot_normalization_stats_from_minmax(stats_path)
                    with open(stats_path_after_compute, "w") as f:
                        json.dump(stats, f, indent=4)
                    print(f"computed stats and saved to: {stats_path_after_compute}")
                    norm_arm_list.append(stats)
                else:
                    raise FileNotFoundError(f"normalization stats file not found: {stats_path}")
            
            self.arm2stats_dict[arm_name] = merge_lerobot_stats(norm_arm_list)

    def _load_trajectories(self):
        """加载轨迹数据（与原版相同，但会记录 H5 路径）"""
        parquet_process_units = []
        for arm_name, arm_config in self.config['data_groups'].items():
            for dataset_name, dataset_config in arm_config.items():
                dataset_path = dataset_config.get('path', None)
                if dataset_path is None:
                    raise ValueError(f"Dataset path for '{arm_name}-{dataset_name}' is not configured")
                dataset_path = Path(dataset_path)
                parquet_files = list(dataset_path.glob("data/*/*.parquet"))
                
                task_mapping = self.tasks[arm_name][dataset_name]
                
                for parquet_path in parquet_files:
                    parquet_process_units.append((
                        parquet_path, 
                        arm_name, 
                        dataset_name, 
                        dataset_config, 
                        dataset_path,
                        task_mapping,  
                        self.action_horizon,
                        self.max_samples_per_file,
                        self.cache_dir  
                    ))

        print(f"total {len(parquet_process_units)} parquet files to process")
        
        num_processes = min(16, len(parquet_process_units))
        print(f"Using {num_processes} processes for concurrent processing")
        
        with mp.Pool(processes=num_processes) as pool:
            total_episodes = 0
            with tqdm(total=len(parquet_process_units), desc="Processing Parquet files to cache") as pbar:
                for episode_files, error in pool.imap_unordered(_process_parquet_file_worker, parquet_process_units):
                    if error:
                        logging.error(error)
                    else:
                        self.data.extend(episode_files)  
                        total_episodes += len(episode_files)
                    
                    pbar.set_postfix({
                        'episodes_this_file': len(episode_files),
                        'total_episodes': total_episodes
                    })
                    pbar.update(1)
        
        print(f"Data processing completed, total {len(self.data)} files generated")

    def _pad_tensor(
        self, 
        source_tensor: torch.Tensor, 
        max_dim: int
    ) -> (torch.Tensor, torch.Tensor):
        """填充 tensor 并创建掩码（与原版相同）"""
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

    def _load_video_frame(self, video_paths: dict, timestamp: float, frame_index: int) -> List[Image.Image]:
        """加载视频帧（与原版相同）"""
        frames = []
        for view, path in video_paths.items():
            if not os.path.exists(path):
                raise FileNotFoundError(f"video file not found: {path}")
            
            if self.video_backend == "decord":
                import decord
                try:
                    ctx = self.video_backend_kwargs.get("ctx", "cpu")
                    if ctx == "cpu":
                        ctx = decord.cpu(0)
                    elif ctx == "gpu":
                        ctx = decord.gpu(0)
                    vr = decord.VideoReader(path, ctx=ctx)
                    fps = vr.get_avg_fps()
                    if fps is None or np.isnan(fps):
                        raise ValueError(f"Unable to read FPS: {path}")
                    frame_idx = int(timestamp * fps)
                    if frame_idx >= len(vr):
                        frame_idx = len(vr) - 1
                    frame = vr[frame_idx].asnumpy()
                    frames.append(Image.fromarray(frame))
                except Exception as e:
                    logging.error(f"Failed to read video: {path}, {e}")
                    raise

            elif self.video_backend == "av":
                import av
                try:
                    with av.open(path) as container:
                        video_stream = container.streams.video[0]
                        fps = float(video_stream.average_rate) if video_stream.average_rate else 20.0
                        timestamp_seconds = frame_index / fps
                        target_pts = int(timestamp_seconds / float(video_stream.time_base))
                        container.seek(offset=target_pts, stream=video_stream)
                        frame = next(container.decode(video=0))
                        frames.append(Image.fromarray(frame.to_ndarray(format='rgb24')))
                except Exception as e:
                    logging.error(f"Failed to read video: {path}, {e}")
                    raise
            else:
                raise NotImplementedError(f"Video backend {self.video_backend} not implemented")
        
        if not frames:
            raise ValueError(f"No frames loaded from video!")
        
        return frames

    def _load_spatial_tokens_from_h5(
        self, 
        h5_path: str, 
        frame_index: int, 
        view_keys: List[str]
    ) -> torch.Tensor:
        """
        从 H5 文件加载 spatial tokens
        
        Args:
            h5_path: H5 文件路径
            frame_index: 帧索引
            view_keys: 视角列表
        
        Returns:
            spatial_tokens: [max_views, 730, 768] float32
        """
        spatial_tokens = []
        
        # 🔥 硬编码映射：view 索引 -> H5 key
        view_key_mapping = {
            0: 'image',          # 第一个 view
            1: 'wrist_image',    # 第二个 view
        }
        
        
        with h5py.File(h5_path, 'r') as h5f:
            frame_key = f"frame_{frame_index:06d}"
            
            if frame_key not in h5f:
                raise KeyError(f"Frame {frame_key} not found in {h5_path}")
            
            frame_grp = h5f[frame_key]
            
            # 🔥 只用一个循环！
            for i, view_key in enumerate(view_keys):
                # 使用索引映射
                h5_view_key = view_key_mapping.get(i, view_key)
                h5_key = f"observation/images/{h5_view_key}_spatial_token"
                token = frame_grp[h5_key][:]
                # 验证形状
                if token.shape != (1, 730, 768):
                    raise ValueError(
                        f"Expected spatial token shape (1, 730, 768), got {token.shape} "
                        f"for {h5_key} in {h5_path}"
                    )
                # 转换为 [730, 768] float32
                token = torch.from_numpy(token).squeeze(0).to(torch.float32)
                spatial_tokens.append(token)
        # 填充到 max_views
        while len(spatial_tokens) < self.max_views:
            dummy_token = torch.zeros(730, 768, dtype=torch.float32)
            spatial_tokens.append(dummy_token)
        
        # 只取前 max_views 个
        spatial_tokens = torch.stack(spatial_tokens[:self.max_views])
        
        return spatial_tokens

    def _load_future_scene_sequence(
        self, 
        h5_path: str, 
        start_frame_index: int,
        episode_length: int
    ) -> (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):
        """
        加载未来场景序列（3D 结构 + RGB + 置信度）
        
        这些密集信息可以用于：
        - 重建 3D/4D 场景
        - 空间推理和规划
        - 任意视角渲染
        - 场景理解
        
        Args:
            h5_path: H5 文件路径
            start_frame_index: 起始帧索引
            episode_length: episode 总长度
        
        Returns:
            pts3d: [N, H, W, 3] 3D坐标（相机坐标系），场景几何结构
            rgb: [N, H, W, 3] RGB颜色 [0, 1]，场景外观
            conf: [N, H, W] 置信度 [0, 1]，估计可靠性
            mask: [N] bool，标记有效帧
        """
        pts3d_list = []
        rgb_list = []
        conf_list = []
        
        # 用于填充边界外的帧（与 action_horizon 相同的处理方式）
        last_valid_pts3d = None
        last_valid_rgb = None
        last_valid_conf = None
        
        try:
            with h5py.File(h5_path, 'r') as h5f:
                for i in range(self.future_horizon):
                    frame_idx = start_frame_index + i
                    
                    # 在 episode 边界内
                    if frame_idx < episode_length:
                        frame_key = f"frame_{frame_idx:06d}"
                        
                        if frame_key not in h5f:
                            if last_valid_pts3d is None:
                                raise KeyError(f"First frame {frame_key} not found in {h5_path}")
                            pts3d_list.append(last_valid_pts3d.copy())
                            rgb_list.append(last_valid_rgb.copy())
                            conf_list.append(last_valid_conf.copy())
                            continue
                        
                        frame_grp = h5f[frame_key]
                        
                        # 🔥 加载完整场景信息
                        pts3d = frame_grp["observation/images/image_pts3d"][:]  # [H, W, 3]
                        rgb = frame_grp["observation/images/image_rgb"][:]      # [H, W, 3]
                        conf = frame_grp["observation/images/image_conf"][:]    # [H, W]
                        
                        pts3d = pts3d.astype(np.float32)
                        rgb = rgb.astype(np.float32)
                        conf = conf.astype(np.float32)
                        
                        pts3d_list.append(pts3d)
                        rgb_list.append(rgb)
                        conf_list.append(conf)
                        
                        last_valid_pts3d = pts3d.copy()
                        last_valid_rgb = rgb.copy()
                        last_valid_conf = conf.copy()
                    
                    else:
                        # 超出边界，用最后有效帧填充
                        if last_valid_pts3d is None:
                            raise ValueError(f"No valid frames found in {h5_path}")
                        
                        pts3d_list.append(last_valid_pts3d.copy())
                        rgb_list.append(last_valid_rgb.copy())
                        conf_list.append(last_valid_conf.copy())
        
        except Exception as e:
            raise RuntimeError(f"Failed to load future scene sequence from {h5_path}: {e}")
        
        # Stack 成序列
        pts3d_tensor = torch.from_numpy(np.stack(pts3d_list))  # [N, H, W, 3]
        rgb_tensor = torch.from_numpy(np.stack(rgb_list))      # [N, H, W, 3]
        conf_tensor = torch.from_numpy(np.stack(conf_list))    # [N, H, W]
        
        # 创建 mask
        valid_length = min(self.future_horizon, episode_length - start_frame_index)
        scene_mask = torch.zeros(self.future_horizon, dtype=torch.bool)
        scene_mask[:valid_length] = True
        
        return pts3d_tensor, rgb_tensor, conf_tensor, scene_mask

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        cache_filepath = self.data[idx]
        
        try:
            with open(cache_filepath, 'rb') as f:
                item = pickle.load(f)
        except Exception as e:
            logging.warning(f"cannot load cache file {cache_filepath}: {str(e)}")
            return self[random.randint(0, len(self.data)-1)]

        arm_key = item["arm_key"]
        dataset_key = item["dataset_key"]
        embodiment_id = self.arm_to_embodiment_id[arm_key]

        # ========== 1. 加载图像 ==========
        try:
            frames = self._load_video_frame(
                item["video_paths"], 
                item["timestamp"], 
                item["frame_index"]
            )
        except Exception as e:
            logging.warning(f"skipping sample that cannot decode video: {e}")
            return self[random.randint(0, len(self.data)-1)]

        images = frames

        if self.use_augmentation:
            images = [
                self.aug_transform(img) if random.random() < 0.5 else self.basic_transform(img)
                for img in images
            ]
        else:
            images = [self.basic_transform(img) for img in images]

        num_real_views = len(images)
        image_mask = torch.zeros(self.max_views, dtype=torch.bool)
        image_mask[:num_real_views] = True 

        # 填充到 max_views（像原版一样）
        while len(images) < self.max_views:
            if len(images) == 0:
                dummy_image = torch.zeros(3, 448, 448)
            else:
                dummy_image = torch.zeros_like(images[0]) 
            images.append(dummy_image)

        images = torch.stack(images)

        # ========== 2. 加载 Spatial Tokens ==========
        view_keys = list(item["video_paths"].keys())
        spatial_tokens = self._load_spatial_tokens_from_h5(
            item["cut3r_h5_path"],
            item["frame_index"],
            view_keys
        )  # [max_views, 730, 768] float32

        # ========== 3. 加载 State ==========
        if item["state"] is None:
            raise ValueError("missing observation.state")
        
        try:
            norm_stats = self.arm2stats_dict[arm_key]
        except KeyError:
            raise KeyError(f"Normalization stats not found for arm_key={arm_key}")

        state = torch.tensor(item["state"], dtype=torch.float32)
        device = state.device
        state_min = torch.tensor(norm_stats["observation.state"]["min"], dtype=torch.float32, device=device)
        state_max = torch.tensor(norm_stats["observation.state"]["max"], dtype=torch.float32, device=device)
        
        state = 2 * (state - state_min) / (state_max - state_min + 1e-8) - 1
        state = torch.clamp(state, -1.0, 1.0)  

        state_padded, state_mask = self._pad_tensor(state, self.max_state_dim)

        # ========== 4. 加载 Action ==========
        if item["action"] is None:
            raise ValueError("missing action")

        action = torch.from_numpy(np.stack(item["action"])).float()
        device = action.device
        action_min = torch.tensor(norm_stats["action"]["min"], dtype=torch.float32, device=device)
        action_max = torch.tensor(norm_stats["action"]["max"], dtype=torch.float32, device=device)
        action = 2 * (action - action_min.unsqueeze(0)) / (action_max.unsqueeze(0) - action_min.unsqueeze(0) + 1e-8) - 1
        action = torch.clamp(action, -1.0, 1.0)

        action_padded, action_mask = self._pad_tensor(action, self.max_action_dim)

        # ========== 5. 加载点云序列 ========== 目前不用 先注释掉用来提速
        #future_pts3d, future_rgb, future_conf, future_mask = self._load_future_scene_sequence(item["cut3r_h5_path"],item["frame_index"],item["episode_length"])
        
        future_pts3d = torch.zeros(self.future_horizon, 1, 1, 3, dtype=torch.float32)
        future_rgb = torch.zeros(self.future_horizon, 1, 1, 3, dtype=torch.float32)
        future_conf = torch.zeros(self.future_horizon, 1, 1, dtype=torch.float32)
        future_mask = torch.zeros(self.future_horizon, dtype=torch.bool)

        # ========== 6. 构建返回字典 ==========
        prompt = item["prompt"]
        
        return {
            # 原有字段
            "images": images,
            "image_mask": image_mask,
            "prompt": prompt,
            "state": state_padded.to(dtype=torch.bfloat16),
            "state_mask": state_mask,
            "action": action_padded.to(dtype=torch.bfloat16),
            "action_mask": action_mask,
            "embodiment_id": torch.tensor(embodiment_id, dtype=torch.long),
            
            # CUT3R 字段
            "spatial_tokens": spatial_tokens,      # [max_views, 730, 768] 图像特征
            "future_pts3d": future_pts3d,          # [future_horizon, H, W, 3] 3D结构
            "future_rgb": future_rgb,              # [future_horizon, H, W, 3] 颜色
            "future_conf": future_conf,            # [future_horizon, H, W] 置信度
            "future_mask": future_mask,            # [future_horizon] 有效帧标记
        }
