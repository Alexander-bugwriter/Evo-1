import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
"""
从配置文件读取H5，快速渲染点云视频 - 渲染器复用版

关键优化：渲染器只初始化一次，避免前14帧初始化失败

用法:
    python render_reuse.py --num_frames 50
"""

import argparse
import yaml
import h5py
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
import random
import open3d as o3d

# 只导入点云重建函数
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


# def render_pointcloud_to_image(pcd, image_size=(432, 432)):
#     """
#     将点云渲染为图像（第一视角）
    
#     Args:
#         pcd: open3d.geometry.PointCloud
#         image_size: (height, width)
    
#     Returns:
#         rendered_img: [H, W, 3] numpy array
#     """
#     if len(pcd.points) == 0:
#         return np.zeros((image_size[0], image_size[1], 3), dtype=np.uint8)
    
#     try:
#         # 创建visualizer
#         vis = o3d.visualization.Visualizer()
#         vis.create_window(visible=False, width=image_size[1], height=image_size[0])
#         vis.add_geometry(pcd)
        
#         # 设置视角
#         ctr = vis.get_view_control()
#         ctr.set_zoom(1.0)
#         ctr.set_front([0, 0, -1])
#         ctr.set_lookat([0, 0, 1])
#         ctr.set_up([0, -1, 0])
        
#         # 渲染
#         vis.poll_events()
#         vis.update_renderer()
        
#         # 截图
#         image = vis.capture_screen_float_buffer(do_render=True)
#         vis.destroy_window()
        
#         # 转换为numpy数组
#         image_np = np.asarray(image)
#         image_np = (image_np * 255).astype(np.uint8)
        
#         return image_np
    
#     except Exception as e:
#         print(f"  ⚠️ 渲染点云失败: {e}")
#         return np.zeros((image_size[0], image_size[1], 3), dtype=np.uint8)

def render_pointcloud_to_image_with_depth(pcd, image_size=(432, 432), zoom=0.8):
    """
    将点云渲染为RGB图和深度图
    
    Args:
        pcd: open3d.geometry.PointCloud
        image_size: (height, width)
        zoom: 缩放级别
    
    Returns:
        rgb_img: [H, W, 3] numpy array (RGB渲染)
        depth_img: [H, W, 3] numpy array (深度图可视化，伪彩色)
    """
    if len(pcd.points) == 0:
        black = np.zeros((image_size[0], image_size[1], 3), dtype=np.uint8)
        return black, black
    
    try:
        # 创建visualizer
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False, width=image_size[1], height=image_size[0])
        vis.add_geometry(pcd)
        
        # 设置视角
        ctr = vis.get_view_control()
        ctr.set_zoom(zoom)
        ctr.set_front([0, 0, -1])
        ctr.set_lookat([0, 0, 0])
        ctr.set_up([0, -1, 0])
        
        # 渲染
        vis.poll_events()
        vis.update_renderer()
        
        # 🔥 截取RGB图
        rgb_buffer = vis.capture_screen_float_buffer(do_render=True)
        rgb_np = np.asarray(rgb_buffer)
        rgb_np = (rgb_np * 255).astype(np.uint8)
        
        # 🔥 截取深度图
        depth_buffer = vis.capture_depth_float_buffer(do_render=False)
        depth_np = np.asarray(depth_buffer)
        
        vis.destroy_window()
        
        # 🔥 深度图可视化（转换为伪彩色）
        # 归一化到 0-1
        
        valid_mask = (depth_np > 0) & np.isfinite(depth_np)
        depth_valid = depth_np[valid_mask]
        
        # 只对有效区域归一化
        if len(depth_valid) > 0:
            depth_min, depth_max = depth_valid.min(), depth_valid.max()
            if depth_max > depth_min:
                depth_normalized = np.zeros_like(depth_np)
                depth_normalized[valid_mask] = \
                    (depth_np[valid_mask] - depth_min) / (depth_max - depth_min)
            else:
                depth_normalized = np.zeros_like(depth_np)
        else:
            depth_normalized = np.zeros_like(depth_np)
        # 转换为8位
        depth_uint8 = (depth_normalized * 255).astype(np.uint8)
        
        # 应用伪彩色（TURBO colormap）
        depth_colored = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_JET)
        depth_colored = cv2.cvtColor(depth_colored, cv2.COLOR_BGR2RGB)
       
        depth_colored[~valid_mask] = [0, 0, 0]

        return rgb_np, depth_colored
        
    except Exception as e:
        print(f"  ⚠️ 渲染点云失败: {e}")
        black = np.zeros((image_size[0], image_size[1], 3), dtype=np.uint8)
        return black, black


def find_all_h5_files(config):
    h5_files = []
    for arm_key, arm_config in config['data_groups'].items():
        for dataset_key, dataset_config in arm_config.items():
            dataset_path = Path(dataset_config['path'])
            cut3r_feature_dir = dataset_path / "cut3r_feature"
            if cut3r_feature_dir.exists():
                h5_list = sorted(cut3r_feature_dir.glob("*.h5"))
                h5_files.extend(h5_list)
                print(f"  找到 {len(h5_list)} 个H5: {arm_key}/{dataset_key}")
    return h5_files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="dataset/config.yaml")
    parser.add_argument("--episode_name", type=str, default=None)
    parser.add_argument("--num_frames", type=int, default=50)
    parser.add_argument("--output", type=str, default="pointcloud_render.mp4")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--conf_threshold", type=float, default=0)
    args = parser.parse_args()
    
    print(f"📄 读取配置: {args.config}")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"🔍 查找H5文件...")
    h5_files = find_all_h5_files(config)
    if len(h5_files) == 0:
        print("❌ 没找到H5文件")
        return
    print(f"✅ 总共 {len(h5_files)} 个H5")
    
    # 选择H5
    if args.episode_name:
        h5_path = None
        for h5 in h5_files:
            if h5.stem == args.episode_name:
                h5_path = h5
                break
        if h5_path is None:
            print(f"❌ 未找到: {args.episode_name}")
            return
    else:
        h5_path = random.choice(h5_files)
        print(f"🎲 随机选择: {h5_path.name}")
    
    # 检查帧数
    with h5py.File(h5_path, 'r') as h5f:
        num_frames_in_h5 = len([k for k in h5f.keys() if k.startswith("frame_")])
    print(f"📊 该episode有 {num_frames_in_h5} 帧")
    actual_num_frames = min(args.num_frames, num_frames_in_h5)
    
    print(f"\n📹 开始渲染: {h5_path.name}")
    print(f"   帧数: {actual_num_frames}")
    print(f"   置信度阈值: {args.conf_threshold}")
    
    # 🔥 完全照抄extract脚本的流程
    original_images = []
    depth_data_list = []
    
    print(f"\n📥 加载数据...")
    for i in tqdm(range(actual_num_frames), desc="加载"):
        with h5py.File(h5_path, 'r') as h5f:
            frame_key = f"frame_{i:06d}"
            if frame_key not in h5f:
                continue
            
            frame_grp = h5f[frame_key]
            pts3d = frame_grp["observation/images/image_pts3d"][:]
            rgb = frame_grp["observation/images/image_rgb"][:]
            conf = frame_grp["observation/images/image_conf"][:]
            
            # 保存原始图像（用于对比）
            # 注意：rgb已经是[0,1]范围的float32
            original_images.append((rgb * 255).astype(np.uint8))
            
            # 保存深度数据
            depth_data_list.append({
                'pts3d': pts3d,
                'rgb': rgb,
                'conf': conf
            })
    
    if len(depth_data_list) == 0:
        print("❌ 没有加载到任何数据")
        return
    
    print(f"✅ 加载了 {len(depth_data_list)} 帧")
    
    # 获取图像尺寸
    img_h, img_w = original_images[0].shape[:2]
    print(f"📐 图像尺寸: {img_h}x{img_w}")
    
    # 准备视频
    # video_w = img_w * 2  # 原图 | 点云渲染
    # video_h = img_h
    
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # out = cv2.VideoWriter(args.output, fourcc, args.fps, (video_w, video_h))
    video_w = img_w * 3  # 🔥 三列：原图 | 点云RGB | 点云深度
    video_h = img_h

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, args.fps, (video_w, video_h))
    
    # print(f"\n🎨 渲染点云...")
    # success_count = 0
    
    # for i, (orig_img, depth_data) in enumerate(tqdm(
    #     zip(original_images, depth_data_list),
    #     total=len(original_images),
    #     desc="渲染"
    # )):
    #     try:
    #         # 🔥 完全照抄：从密集深度重建点云
    #         pcd = reconstruct_pointcloud_from_depth(
    #             depth_data['pts3d'],
    #             depth_data['rgb'],
    #             depth_data['conf'],
    #             conf_threshold=args.conf_threshold
    #         )
            
    #         if len(pcd.points) == 0:
    #             print(f"\n⚠️ Frame {i}: 点云为空，跳过")
    #             continue
            
    #         # 🔥 完全照抄：渲染点云为图像
    #         pc_rendered = render_pointcloud_to_image(pcd, image_size=(img_h, img_w))
            
    #         # 🔥 完全照抄：转换为BGR
    #         orig_bgr = cv2.cvtColor(orig_img.astype(np.uint8), cv2.COLOR_RGB2BGR)
    #         pc_bgr = cv2.cvtColor(pc_rendered.astype(np.uint8), cv2.COLOR_RGB2BGR)
            
    #         # 🔥 完全照抄：添加标签
    #         cv2.putText(orig_bgr, "Original", (10, 30),
    #                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    #         cv2.putText(pc_bgr, f"Pointcloud - {len(pcd.points)} pts", (10, 30),
    #                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
    #         cv2.putText(orig_bgr, f"Frame {i}", (10, 60),
    #                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    #         cv2.putText(pc_bgr, f"Frame {i}", (10, 60),
    #                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
    #         # 🔥 完全照抄：左右拼接
    #         combined = np.hstack([orig_bgr, pc_bgr])
            
    #         out.write(combined)
    #         success_count += 1
            
    #     except Exception as e:
    #         print(f"\n⚠️ Frame {i} 失败: {e}")
    #         import traceback
    #         traceback.print_exc()
    #         continue
    print(f"\n🎨 渲染点云...")
    success_count = 0

    for i, (orig_img, depth_data) in enumerate(tqdm(
        zip(original_images, depth_data_list),
        total=len(original_images),
        desc="渲染"
    )):
        try:
            # 从密集深度重建点云
            pcd = reconstruct_pointcloud_from_depth(
                depth_data['pts3d'],
                depth_data['rgb'],
                depth_data['conf'],
                conf_threshold=args.conf_threshold
            )
            
            if len(pcd.points) == 0:
                print(f"\n⚠️ Frame {i}: 点云为空，跳过")
                continue
            
            # 🔥 渲染RGB和深度图
            pc_rgb, pc_depth = render_pointcloud_to_image_with_depth(
                pcd, image_size=(img_h, img_w), zoom=0.8
            )
            
            # 转换为BGR
            orig_bgr = cv2.cvtColor(orig_img.astype(np.uint8), cv2.COLOR_RGB2BGR)
            pc_rgb_bgr = cv2.cvtColor(pc_rgb.astype(np.uint8), cv2.COLOR_RGB2BGR)
            pc_depth_bgr = cv2.cvtColor(pc_depth.astype(np.uint8), cv2.COLOR_RGB2BGR)
            
            # 添加标签
            cv2.putText(orig_bgr, "Original RGB", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(pc_rgb_bgr, f"PC RGB - {len(pcd.points)} pts", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(pc_depth_bgr, "PC Depth", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            cv2.putText(orig_bgr, f"Frame {i}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(pc_rgb_bgr, f"Frame {i}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(pc_depth_bgr, f"Frame {i}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            # 🔥 三列拼接：原图 | 点云RGB | 点云深度
            combined = np.hstack([orig_bgr, pc_rgb_bgr, pc_depth_bgr])
            
            out.write(combined)
            success_count += 1
            
        except Exception as e:
            print(f"\n⚠️ Frame {i} 失败: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    out.release()
    
    print(f"\n{'='*60}")
    print(f"✅ 完成!")
    print(f"   成功: {success_count}/{len(original_images)} 帧")
    print(f"   输出: {args.output}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
