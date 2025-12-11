"""
从 LeRobot 数据集提取 CUT3R 特征并保存到 H5

用法:
    # 正常处理（检测已有文件，跳过）
    python extract_cut3r_features.py --config dataset/config.yaml --cut3r_weights ./cut3r.pth --device cuda:0
    
    # Resume 模式（跳过已处理的文件）
    python extract_cut3r_features.py --config dataset/config.yaml --cut3r_weights ./cut3r.pth --resume
    
    # Overwrite 模式（重新处理所有文件）
    python extract_cut3r_features.py --config dataset/config.yaml --cut3r_weights ./cut3r.pth --overwrite
    
    # Debug 模式（只处理 2 个 episode 并渲染视频）
    python extract_cut3r_features.py --config dataset/config.yaml --cut3r_weights ./cut3r.pth --debug_episodes 2
"""
import os
import sys
import argparse
import yaml
import h5py
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import logging
import cv2

# 添加当前目录到 path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 导入 CUT3R encoder 和渲染函数
from cut3r_encoder_lyh import (
    CUT3REncoder, 
    prepare_input,
)

try:
    import open3d as o3d
    _OPEN3D_AVAILABLE = True
except ImportError:
    print("⚠️ Warning: open3d not available")
    _OPEN3D_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_video_frame_av(video_path: str, frame_index: int):
    """使用 av 后端加载视频帧"""
    import av
    
    try:
        with av.open(video_path) as container:
            video_stream = container.streams.video[0]
            fps = float(video_stream.average_rate) if video_stream.average_rate else 20.0
            timestamp_seconds = frame_index / fps
            target_pts = int(timestamp_seconds / float(video_stream.time_base))
            container.seek(offset=target_pts, stream=video_stream)
            frame = next(container.decode(video=0))
            return np.array(frame.to_ndarray(format='rgb24'))
    except Exception as e:
        logger.error(f"Failed to read video {video_path} frame {frame_index}: {e}")
        raise


# def process_episode(
#     parquet_path: Path,
#     dataset_path: Path,
#     view_map: dict,
#     encoder: CUT3REncoder,
#     device: str,
#     output_dir: Path,
#     episode_name: str,
#     voxel_size: float = 0.02,
#     debug_mode: bool = False
# ):
#     """
#     处理单个 episode，提取 CUT3R 特征并保存到 H5
    
#     Returns:
#         success: bool
#         debug_data: dict (仅 debug 模式返回)
#     """
#     # 读取 parquet
#     df = pd.read_parquet(parquet_path)
#     num_frames = len(df)
    
#     logger.info(f"  📊 Episode: {episode_name}, {num_frames} frames")
    
#     # 构建 video 路径
#     chunk_name = parquet_path.parent.name
#     base_video_path = dataset_path / "videos" / chunk_name
    
#     # 获取视图对应的 video 路径
#     video_paths = {}
#     view_keys = sorted(view_map.keys())
    
#     for view_key, view_folder in view_map.items():
#         video_file = base_video_path / view_folder / f"{parquet_path.stem}.mp4"
#         if video_file.exists():
#             video_paths[view_key] = str(video_file)
#         else:
#             logger.warning(f"  ⚠️ Video not found: {video_file}")
    
#     if len(video_paths) < 2:
#         logger.error(f"  ❌ Need at least 2 views, found {len(video_paths)}")
#         return False, None
    
#     # 使用前两个视图（base + wrist）
#     view_keys = sorted(video_paths.keys())[:2]
#     base_video = video_paths[view_keys[0]]
#     wrist_video = video_paths[view_keys[1]]
    
#   /opt/liblibai-models/user-workspace2/dataset/IPEC-COMMUNITY/libero_spatial_no_noops_1.0.0_lerobot  logger.info(f"  Base: {Path(base_video).name}")
#     logger.info(f"  Wrist: {Path(wrist_video).name}")
    
#     # 创建临时 H5 文件
#     h5_path = output_dir / f"{episode_name}.h5"
#     h5_tmp_path = output_dir / f"{episode_name}.h5.tmp"
    
#     # Debug 数据和目录
#     debug_data = None
#     debug_pc_dir = None
#     if debug_mode:
#         debug_data = {
#             'base_images': [],
#             'wrist_images': [],
#             'base_pointclouds': [],
#             'wrist_pointclouds': []
#         }
#         # Debug模式：创建PLY文件保存目录
#         debug_pc_dir = Path("/opt/liblibai-models/user-workspace2/users/lyh/Evo-1/Evo_1/scripts/debug_videos/debug_pointcloud") / f"debug_episode_{episode_name}"
#         debug_pc_dir.mkdir(parents=True, exist_ok=True)
#         logger.info(f"  🐛 Debug pointcloud dir: {debug_pc_dir}")
#     encoder.reset_state()
#     logger.info(f"  🔄 Reset CUT3R state for new episode") 
#     try:
#         with h5py.File(h5_tmp_path, 'w') as h5f:
#             # 逐帧处理
#             for frame_idx in tqdm(range(num_frames), desc=f"    Processing {episode_name}", leave=False):
#                 # 加载图像
#                 base_img = load_video_frame_av(base_video, frame_idx)
#                 wrist_img = load_video_frame_av(wrist_video, frame_idx)
                
#                 # Debug: 保存原图
#                 if debug_mode:
#                     debug_data['base_images'].append(base_img.copy())
#                     debug_data['wrist_images'].append(wrist_img.copy())
                
#                 # 转换为 tensor
#                 base_tensor = torch.from_numpy(base_img).permute(2, 0, 1)
#                 base_norm = base_tensor.to(torch.float32).to(device) / 127.5 - 1.0
                
#                 wrist_tensor = torch.from_numpy(wrist_img).permute(2, 0, 1)
#                 wrist_norm = wrist_tensor.to(torch.float32).to(device) / 127.5 - 1.0
                
#                 # 拼接成 (1, 2, C, H, W)
#                 pixel_values = torch.stack([base_norm, wrist_norm], dim=0).unsqueeze(0)
                
#                 # Prepare input（会 resize 到 432x432）
#                 views = prepare_input(pixel_values, device, target_size=432)
                
#                 # 🔥 第一帧重置状态
#                 if frame_idx == 0:
#                     for view in views:
#                         view["reset"] = torch.tensor([True], device=device).expand(view["img"].shape[0])
                
#                 # Forward
#                 results, camera_tokens, patch_tokens = encoder.forward(views)
                
#                 # 切分 batch: [2, N, 768] -> [1, N, 768] + [1, N, 768]
#                 base_camera = camera_tokens[0:1]   # [1, 1, 768]
#                 wrist_camera = camera_tokens[1:2]  # [1, 1, 768]
#                 base_patch = patch_tokens[0:1]     # [1, 729, 768]
#                 wrist_patch = patch_tokens[1:2]    # [1, 729, 768]
                
#                 # 拼接成 spatial_token [1, 730, 768]
#                 base_spatial = torch.cat([base_camera, base_patch], dim=1)
#                 wrist_spatial = torch.cat([wrist_camera, wrist_patch], dim=1)
                
#                 # 处理点云数据
#                 base_pointcloud_data = None
#                 wrist_pointcloud_data = None
                
#                 for b in range(2):  # base=0, wrist=1
#                     pts3d = results[0]["pts3d_in_other_view"][b].cpu().numpy()
#                     img = views[0]["img"][b]
#                     colors = (img.permute(1, 2, 0) * 0.5 + 0.5).clamp(0, 1).cpu().numpy()
                    
#                     pts3d_flat = pts3d.reshape(-1, 3)
#                     colors_flat = colors.reshape(-1, 3)
                    
#                     # 过滤无效点
#                     valid_mask = np.isfinite(pts3d_flat).all(axis=1)
#                     pts3d_valid = pts3d_flat[valid_mask]
#                     colors_valid = colors_flat[valid_mask]
                    
#                     # 可选：体素下采样
#                     if len(pts3d_valid) > 0 and voxel_size > 0 and _OPEN3D_AVAILABLE:
#                         pcd = o3d.geometry.PointCloud()
#                         pcd.points = o3d.utility.Vector3dVector(pts3d_valid)
#                         pcd.colors = o3d.utility.Vector3dVector(colors_valid)
#                         pcd = pcd.voxel_down_sample(voxel_size)
#                         pts3d_valid = np.asarray(pcd.points)
#                         colors_valid = np.asarray(pcd.colors)
                        
#                         # Debug模式：保存PLY文件
#                         if debug_mode and debug_pc_dir is not None:
#                             view_name = "base" if b == 0 else "wrist"
#                             debug_pc_path = debug_pc_dir / f"frame_{frame_idx:06d}_{view_name}.ply"
#                             o3d.io.write_point_cloud(str(debug_pc_path), pcd)
                    
#                     # 拼接成 [N, 6] (xyz + rgb)，用于存入H5
#                     if len(pts3d_valid) > 0:
#                         pointcloud_array = np.column_stack([pts3d_valid, colors_valid])
#                     else:
#                         pointcloud_array = np.zeros((0, 6), dtype=np.float32)
                    
#                     if b == 0:
#                         base_pointcloud_data = pointcloud_array
#                     else:
#                         wrist_pointcloud_data = pointcloud_array
                    
#                     # Debug: 记录点云数据
#                     if debug_mode and debug_pc_path is not None:
#                         if b == 0:
#                             debug_data['base_pointclouds'].append(str(debug_pc_path))
#                         else:
#                             debug_data['wrist_pointclouds'].append(str(debug_pc_path))
                
#                 # 保存到 H5
#                 frame_grp = h5f.create_group(f"frame_{frame_idx:06d}")
                
#                 # 保存 spatial tokens
#                 frame_grp.create_dataset(
#                     "observation/images/image_spatial_token",
#                     data=base_spatial.cpu().detach().to(torch.float32).numpy(),
#                     compression="gzip"
#                 )
#                 frame_grp.create_dataset(
#                     "observation/images/wrist_image_spatial_token",
#                     data=wrist_spatial.cpu().detach().to(torch.float32).numpy(),
#                     compression="gzip"
#                 )
                
#                 # 保存点云数据 [N, 6] (xyz + rgb)
#                 frame_grp.create_dataset(
#                     "observation/images/image_pointcloud",
#                     data=base_pointcloud_data,
#                     compression="gzip",
#                     compression_opts=9
#                 )
#                 frame_grp.create_dataset(
#                     "observation/images/wrist_image_pointcloud",
#                     data=wrist_pointcloud_data,
#                     compression="gzip",
#                     compression_opts=9
#                 )
        
#         # 🔥 处理完成，重命名临时文件
#         if h5_tmp_path.exists():
#             h5_tmp_path.rename(h5_path)
#             logger.info(f"  ✅ Saved: {h5_path.name}")
        
#         return True, debug_data
        
#     except Exception as e:
#         logger.error(f"  ❌ Failed to process episode: {e}")
#         import traceback
#         logger.error(traceback.format_exc())
        
#         # 清理临时文件
#         if h5_tmp_path.exists():
#             h5_tmp_path.unlink()
        
#         raise

def process_episode(
    parquet_path: Path,
    dataset_path: Path,
    view_map: dict,
    encoder: CUT3REncoder,
    device: str,
    output_dir: Path,
    episode_name: str,
    voxel_size: float = 0.02,
    debug_mode: bool = False
):
    """
    处理单个 episode，提取 CUT3R 特征并保存到 H5
    
    Returns:
        success: bool
        debug_data: dict (仅 debug 模式返回)
    """
    # 读取 parquet
    df = pd.read_parquet(parquet_path)
    num_frames = len(df)
    
    logger.info(f"  📊 Episode: {episode_name}, {num_frames} frames")
    
    # 构建 video 路径
    chunk_name = parquet_path.parent.name
    base_video_path = dataset_path / "videos" / chunk_name
    
    # 获取视图对应的 video 路径
    video_paths = {}
    view_keys = sorted(view_map.keys())
    
    for view_key, view_folder in view_map.items():
        video_file = base_video_path / view_folder / f"{parquet_path.stem}.mp4"
        if video_file.exists():
            video_paths[view_key] = str(video_file)
        else:
            logger.warning(f"  ⚠️ Video not found: {video_file}")
    
    if len(video_paths) < 2:
        logger.error(f"  ❌ Need at least 2 views, found {len(video_paths)}")
        return False, None
    
    # 使用前两个视图（base + wrist）
    view_keys = sorted(video_paths.keys())[:2]
    base_video = video_paths[view_keys[0]]
    wrist_video = video_paths[view_keys[1]]
    
    logger.info(f"  Base: {Path(base_video).name}")
    logger.info(f"  Wrist: {Path(wrist_video).name}")
    
    # 创建临时 H5 文件
    h5_path = output_dir / f"{episode_name}.h5"
    h5_tmp_path = output_dir / f"{episode_name}.h5.tmp"
    
    # Debug 数据
    debug_data = None
    if debug_mode:
        debug_data = {
            'base_images': [],
            'wrist_images': [],
            'base_depth': [],
            'wrist_depth': []
        }
    
    encoder.reset_state()
    logger.info(f"  🔄 Reset CUT3R state for new episode") 
    
    try:
        with h5py.File(h5_tmp_path, 'w') as h5f:
            # 逐帧处理
            for frame_idx in tqdm(range(num_frames), desc=f"    Processing {episode_name}", leave=False):
                # 加载图像
                base_img = load_video_frame_av(base_video, frame_idx)
                wrist_img = load_video_frame_av(wrist_video, frame_idx)
                
                # Debug: 保存原图
                if debug_mode:
                    debug_data['base_images'].append(base_img.copy())
                    debug_data['wrist_images'].append(wrist_img.copy())
                
                # 转换为 tensor
                base_tensor = torch.from_numpy(base_img).permute(2, 0, 1)
                base_norm = base_tensor.to(torch.float32).to(device) / 127.5 - 1.0
                
                wrist_tensor = torch.from_numpy(wrist_img).permute(2, 0, 1)
                wrist_norm = wrist_tensor.to(torch.float32).to(device) / 127.5 - 1.0
                
                # 拼接成 (1, 2, C, H, W)
                pixel_values = torch.stack([base_norm, wrist_norm], dim=0).unsqueeze(0)
                
                # Prepare input（会 resize 到 432x432）
                views = prepare_input(pixel_values, device, target_size=432)
                
                # 🔥 第一帧重置状态
                if frame_idx == 0:
                    for view in views:
                        view["reset"] = torch.tensor([True], device=device).expand(view["img"].shape[0])
                
                # Forward
                results, camera_tokens, patch_tokens = encoder.forward(views)
                
                # 切分 batch: [2, N, 768] -> [1, N, 768] + [1, N, 768]
                base_camera = camera_tokens[0:1]   # [1, 1, 768]
                wrist_camera = camera_tokens[1:2]  # [1, 1, 768]
                base_patch = patch_tokens[0:1]     # [1, 729, 768]
                wrist_patch = patch_tokens[1:2]    # [1, 729, 768]
                
                # 拼接成 spatial_token [1, 730, 768]
                base_spatial = torch.cat([base_camera, base_patch], dim=1)
                wrist_spatial = torch.cat([wrist_camera, wrist_patch], dim=1)
                
                # 🔥 处理密集深度数据（RGB + Depth + Conf）
                base_pts3d = None
                base_rgb = None
                base_conf = None
                wrist_pts3d = None
                wrist_rgb = None
                wrist_conf = None
                
                for b in range(2):  # base=0, wrist=1
                    pts3d = results[0]["pts3d_in_other_view"][b].cpu().numpy()  # [H, W, 3]
                    conf = results[0]["conf"][b].cpu().numpy()  # [H, W]
                    img = views[0]["img"][b]
                    rgb = (img.permute(1, 2, 0) * 0.5 + 0.5).clamp(0, 1).cpu().numpy()  # [H, W, 3]
                    
                    if b == 0:
                        base_pts3d = pts3d.astype(np.float32)
                        base_rgb = rgb.astype(np.float32)
                        base_conf = conf.astype(np.float32)
                        
                        # Debug: 保存深度数据
                        if debug_mode:
                            debug_data['base_depth'].append({
                                'pts3d': pts3d.copy(),
                                'rgb': rgb.copy(),
                                'conf': conf.copy()
                            })
                    else:
                        wrist_pts3d = pts3d.astype(np.float32)
                        wrist_rgb = rgb.astype(np.float32)
                        wrist_conf = conf.astype(np.float32)
                        
                        # Debug: 保存深度数据
                        if debug_mode:
                            debug_data['wrist_depth'].append({
                                'pts3d': pts3d.copy(),
                                'rgb': rgb.copy(),
                                'conf': conf.copy()
                            })
                
                # 保存到 H5
                frame_grp = h5f.create_group(f"frame_{frame_idx:06d}")
                
                # 保存 spatial tokens
                frame_grp.create_dataset(
                    "observation/images/image_spatial_token",
                    data=base_spatial.cpu().detach().to(torch.float32).numpy(),
                    compression="gzip"
                )
                frame_grp.create_dataset(
                    "observation/images/wrist_image_spatial_token",
                    data=wrist_spatial.cpu().detach().to(torch.float32).numpy(),
                    compression="gzip"
                )
                
                # 🔥 保存密集 RGB + Depth + Conf
                frame_grp.create_dataset(
                    "observation/images/image_pts3d",
                    data=base_pts3d,
                    compression="gzip"
                )
                frame_grp.create_dataset(
                    "observation/images/image_rgb",
                    data=base_rgb,
                    compression="gzip"
                )
                frame_grp.create_dataset(
                    "observation/images/image_conf",
                    data=base_conf,
                    compression="gzip"
                )
                frame_grp.create_dataset(
                    "observation/images/wrist_image_pts3d",
                    data=wrist_pts3d,
                    compression="gzip"
                )
                frame_grp.create_dataset(
                    "observation/images/wrist_image_rgb",
                    data=wrist_rgb,
                    compression="gzip"
                )
                frame_grp.create_dataset(
                    "observation/images/wrist_image_conf",
                    data=wrist_conf,
                    compression="gzip"
                )
        
        # 🔥 处理完成，重命名临时文件
        if h5_tmp_path.exists():
            h5_tmp_path.rename(h5_path)
            logger.info(f"  ✅ Saved: {h5_path.name}")
        
        return True, debug_data
        
    except Exception as e:
        logger.error(f"  ❌ Failed to process episode: {e}")
        import traceback
        logger.error(traceback.format_exc())
        
        # 清理临时文件
        if h5_tmp_path.exists():
            h5_tmp_path.unlink()
        
        raise

def reconstruct_pointcloud_from_depth(pts3d, rgb, conf=None, conf_threshold=0.5):
    """
    从密集深度重建点云
    
    Args:
        pts3d: [H, W, 3] 3D坐标
        rgb: [H, W, 3] RGB颜色
        conf: [H, W] 置信度（可选）
        conf_threshold: 置信度阈值
    
    Returns:
        pcd: open3d.geometry.PointCloud
    """
    
    # Flatten
    pts3d_flat = pts3d.reshape(-1, 3)
    rgb_flat = rgb.reshape(-1, 3)
    
    # 过滤无效点
    valid_mask = np.isfinite(pts3d_flat).all(axis=1)
    
    # 如果有置信度，额外过滤低置信度点
    if conf is not None:
        conf_flat = conf.reshape(-1)
        valid_mask = valid_mask & (conf_flat > conf_threshold)
    
    pts3d_valid = pts3d_flat[valid_mask]
    rgb_valid = rgb_flat[valid_mask]
    
    # 创建点云
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts3d_valid)
    pcd.colors = o3d.utility.Vector3dVector(rgb_valid)
    
    return pcd


def render_pointcloud_to_image(pcd, image_size=(432, 432)):
    """
    将点云渲染为图像（第一视角）
    
    Args:
        pcd: open3d.geometry.PointCloud
        image_size: (height, width)
    
    Returns:
        rendered_img: [H, W, 3] numpy array
    """
    if len(pcd.points) == 0:
        return np.zeros((image_size[0], image_size[1], 3), dtype=np.uint8)
    
    try:
        # 离屏渲染
        render = o3d.visualization.rendering.OffscreenRenderer(image_size[1], image_size[0])
        
        # 设置材质
        mat = o3d.visualization.rendering.MaterialRecord()
        mat.shader = 'defaultUnlit'
        
        # 添加点云
        render.scene.add_geometry("pointcloud", pcd, mat)
        
        # 设置相机
        bounds = pcd.get_axis_aligned_bounding_box()
        center = bounds.get_center()
        extent = bounds.get_extent()
        
        # 计算相机位置（正前方）
        distance = np.linalg.norm(extent) * 1.5
        camera_pos = center + np.array([0, 0, distance])
        
        render.setup_camera(60, center, camera_pos, [0, -1, 0])
        
        # 渲染
        image = render.render_to_image()
        image_np = np.asarray(image)
        
        return image_np
    
    except Exception as e:
        logger.warning(f"  ⚠️ 渲染点云失败: {e}")
        return np.zeros((image_size[0], image_size[1], 3), dtype=np.uint8)


def render_debug_videos(debug_data_list, output_dir, fps=10):
    """
    渲染 debug 视频（原图 vs 点云渲染）
    同时保存点云 PLY 文件
    """
    if not _OPEN3D_AVAILABLE:
        logger.warning("⚠️ open3d not available, skipping video rendering")
        return
    
    logger.info("\n🎥 渲染 Debug 对比视频...")
    
    video_dir = Path(output_dir) / "debug_videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建点云保存目录
    pointcloud_dir = video_dir / "pointclouds"
    pointcloud_dir.mkdir(parents=True, exist_ok=True)
    
    for ep_idx, debug_data in enumerate(debug_data_list):
        # Base 视频和点云
        base_video = str(video_dir / f"episode_{ep_idx}_base.mp4")
        base_pc_dir = pointcloud_dir / f"episode_{ep_idx}_base"
        base_pc_dir.mkdir(parents=True, exist_ok=True)
        
        create_pointcloud_comparison_video(
            debug_data['base_images'],
            debug_data['base_depth'],
            base_video,
            base_pc_dir,
            fps=fps
        )
        
        # Wrist 视频和点云
        wrist_video = str(video_dir / f"episode_{ep_idx}_wrist.mp4")
        wrist_pc_dir = pointcloud_dir / f"episode_{ep_idx}_wrist"
        wrist_pc_dir.mkdir(parents=True, exist_ok=True)
        
        create_pointcloud_comparison_video(
            debug_data['wrist_images'],
            debug_data['wrist_depth'],
            wrist_video,
            wrist_pc_dir,
            fps=fps
        )
    
    logger.info(f"✅ Debug videos saved to: {video_dir}")
    logger.info(f"✅ Pointclouds saved to: {pointcloud_dir}")


def create_pointcloud_comparison_video(
    original_images, 
    depth_data_list, 
    output_path, 
    pointcloud_dir,
    fps=10
):
    """
    创建原始图像和点云渲染的对比视频
    同时保存每帧的点云 PLY 文件
    
    Args:
        original_images: list of [H, W, 3] numpy arrays
        depth_data_list: list of dicts with keys ['pts3d', 'rgb', 'conf']
        output_path: 输出视频路径
        pointcloud_dir: 点云保存目录
        fps: 帧率
    """
    logger.info(f"  📹 创建视频: {output_path}")
    
    if len(original_images) == 0:
        logger.warning("  ⚠️ 没有图像数据")
        return
    
    # 获取图像尺寸
    img_h, img_w = original_images[0].shape[:2]
    
    # 视频尺寸：原图 | 点云渲染（左右对比）
    video_w = img_w * 2
    video_h = img_h
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (video_w, video_h))
    
    for i, (orig_img, depth_data) in enumerate(tqdm(
        zip(original_images, depth_data_list),
        total=len(original_images),
        desc="      处理帧",
        leave=False
    )):
        # 🔥 从密集深度重建点云
        pcd = reconstruct_pointcloud_from_depth(
            depth_data['pts3d'],
            depth_data['rgb'],
            depth_data['conf'],
            conf_threshold=0.5
        )
        
        # 🔥 保存点云 PLY 文件
        ply_path = pointcloud_dir / f"frame_{i:06d}.ply"
        o3d.io.write_point_cloud(str(ply_path), pcd)
        
        # 🔥 渲染点云为图像
        pc_rendered = render_pointcloud_to_image(pcd, image_size=(img_h, img_w))
        
        # 转换为 BGR（OpenCV 格式）
        orig_bgr = cv2.cvtColor(orig_img.astype(np.uint8), cv2.COLOR_RGB2BGR)
        pc_bgr = cv2.cvtColor(pc_rendered.astype(np.uint8), cv2.COLOR_RGB2BGR)
        
        # 添加标签
        cv2.putText(orig_bgr, "Original", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(pc_bgr, "Pointcloud", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # 添加帧号和点数
        cv2.putText(orig_bgr, f"Frame {i}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(pc_bgr, f"Frame {i} | {len(pcd.points)} pts", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # 左右拼接：原图 | 点云渲染
        combined = np.hstack([orig_bgr, pc_bgr])
        
        out.write(combined)
    
    out.release()
    logger.info(f"  ✅ 视频保存成功")
    logger.info(f"  ✅ 点云保存到: {pointcloud_dir}")


def main():
    parser = argparse.ArgumentParser(description="从 LeRobot 数据集提取 CUT3R 特征")
    parser.add_argument("--config", type=str, required=True, help="数据集配置文件")
    parser.add_argument("--cut3r_weights", type=str, required=True, help="CUT3R 权重路径")
    parser.add_argument("--device", type=str, default="cuda:0", help="设备")
    parser.add_argument("--voxel_size", type=float, default=0.02, help="voxel downsampling")
    parser.add_argument("--resume", action="store_true", help="跳过已处理的文件")
    parser.add_argument("--overwrite", action="store_true", help="重新处理所有文件")
    parser.add_argument("--debug_episodes", type=int, default=None, help="Debug: 只处理 N 个 episodes + 渲染视频")
    args = parser.parse_args()
    
    # Resume 和 Overwrite 互斥
    if args.resume and args.overwrite:
        logger.error("❌ --resume 和 --overwrite 不能同时使用")
        return
    
    logger.info("=" * 80)
    logger.info("🚀 CUT3R 特征提取")
    logger.info("=" * 80)
    logger.info(f"📝 Config: {args.config}")
    logger.info(f"🔧 Weights: {args.cut3r_weights}")
    logger.info(f"💻 Device: {args.device}")
    logger.info(f"📊 Voxel: {args.voxel_size}")
    
    if args.resume:
        logger.info(f"♻️  Resume: 跳过已处理的文件")
    elif args.overwrite:
        logger.info(f"🔄 Overwrite: 重新处理所有文件")
    
    if args.debug_episodes:
        logger.info(f"🐛 DEBUG: 处理 {args.debug_episodes} episodes + 渲染视频")
    
    # 加载配置
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # 初始化 CUT3R
    logger.info("\n🔧 初始化 CUT3R...")
    encoder = CUT3REncoder(args.cut3r_weights, args.device)
    logger.info("✅ CUT3R 初始化完成")
    
    # 遍历所有数据集
    total_episodes = 0
    processed_episodes = 0
    skipped_episodes = 0
    debug_data_list = []
    
    for arm_key, arm_config in config['data_groups'].items():
        logger.info(f"\n{'='*80}")
        logger.info(f"🤖 处理 arm: {arm_key}")
        logger.info(f"{'='*80}")
        
        for dataset_key, dataset_config in arm_config.items():
            logger.info(f"\n📦 Dataset: {dataset_key}")
            
            dataset_path = Path(dataset_config['path'])
            view_map = dataset_config.get('view_map', {})
            
            # 创建 cut3r_feature 目录
            cut3r_feature_dir = dataset_path / "cut3r_feature"
            cut3r_feature_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"  📂 Output: {cut3r_feature_dir}")
            
            # 查找所有 parquet 文件
            parquet_files = sorted(dataset_path.glob("data/*/*.parquet"))
            logger.info(f"  📊 Found {len(parquet_files)} episodes")
            
            for parquet_path in parquet_files:
                if args.debug_episodes and processed_episodes >= args.debug_episodes:
                    logger.info(f"\n🐛 Debug limit reached ({args.debug_episodes} episodes)")
                    break
                
                total_episodes += 1
                
                # 构建 episode 名称
                chunk_name = parquet_path.parent.name
                episode_name = f"{chunk_name}_{parquet_path.stem}"
                
                # 检查文件是否已存在
                h5_path = cut3r_feature_dir / f"{episode_name}.h5"
                if h5_path.exists() and not args.overwrite:
                    if args.resume:
                        logger.info(f"  ⏭️  Skipped (exists): {episode_name}")
                        skipped_episodes += 1
                        continue
                    else:
                        # 默认行为：文件存在则跳过
                        logger.info(f"  ⏭️  Skipped (exists): {episode_name}")
                        skipped_episodes += 1
                        continue
                
                
                # 处理 episode
                success, debug_data = process_episode(
                    parquet_path,
                    dataset_path,
                    view_map,
                    encoder,
                    args.device,
                    cut3r_feature_dir,
                    episode_name,
                    args.voxel_size,
                    debug_mode=(args.debug_episodes is not None)
                )
                
                if success:
                    processed_episodes += 1
                    if debug_data is not None:
                        debug_data_list.append(debug_data)
                    logger.info(f"  ✅ Processed: {episode_name}")
                    
                
            
            if args.debug_episodes and processed_episodes >= args.debug_episodes:
                break
        
        if args.debug_episodes and processed_episodes >= args.debug_episodes:
            break
    
    # 渲染 debug 视频
    if args.debug_episodes and debug_data_list:
        script_dir = Path(__file__).parent
        render_debug_videos(debug_data_list, script_dir, fps=10)
    
    # 总结
    logger.info("\n" + "=" * 80)
    logger.info("📊 处理总结")
    logger.info("=" * 80)
    logger.info(f"✅ 成功: {processed_episodes}")
    logger.info(f"⏭️  跳过: {skipped_episodes}")
    logger.info(f"❌ 失败: {total_episodes - processed_episodes - skipped_episodes}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
