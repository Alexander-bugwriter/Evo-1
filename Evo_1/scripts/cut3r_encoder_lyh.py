"""
极简版 CUT3R Encoder - 直接使用官方方法
"""
import torch
import torch.nn as nn
import numpy as np
import sys
import os
from PIL import Image
from pathlib import Path
import cv2
from tqdm import tqdm

# 添加 CUT3R 路径
script_dir = os.path.dirname(os.path.abspath(__file__))
cut3r_dir = os.path.join(os.path.dirname(script_dir), 'CUT3R')
sys.path.insert(0, os.path.join(cut3r_dir, 'src'))

# from dust3r.model import ARCroco3DStereo
from CUT3R.src.dust3r.model import ARCroco3DStereo

try:
    import open3d as o3d

    _OPEN3D_AVAILABLE = True
except ImportError:
    print("Warning: open3d not available")
    _OPEN3D_AVAILABLE = False


def prepare_input(pixel_values, device, target_size=512):
    """
    准备输入 views

    Args:
        pixel_values: [F, B, C, H, W] 范围 [-1, 1]
        device: 设备
        target_size: resize 目标尺寸

    Returns:
        list of view dicts
    """
    F, B, C, H_orig, W_orig = pixel_values.shape

    # # 计算 resize 尺寸（保持长宽比，确保是 16 的倍数）
    # aspect_ratio = W_orig / H_orig
    # if aspect_ratio > 1:
    #     new_w = target_size
    #     new_h = int(target_size / aspect_ratio)
    # else:
    #     new_h = target_size
    #     new_w = int(target_size * aspect_ratio)

    new_h = target_size
    new_w = target_size

    # Resize
    pixel_values_flat = pixel_values.reshape(F * B, C, H_orig, W_orig)
    pixel_values_resized = torch.nn.functional.interpolate(
        pixel_values_flat,
        size=(new_h, new_w),
        mode='bilinear',
        align_corners=False
    )
    pixel_values_resized = pixel_values_resized.reshape(F, B, C, new_h, new_w)

    #print(f"[prepare_input] Resized: {H_orig}x{W_orig} -> {new_h}x{new_w}, F={F}, B={B}")

    # 构建 views
    views = []
    for i in range(F):
        current_frame_batch = pixel_values_resized[i]  # [B, C, H, W]

        view = {
            "img": current_frame_batch,
            "ray_map": torch.full((B, 6, new_h, new_w), torch.nan, device=device),
            "true_shape": torch.tensor([[new_h, new_w]], device=device, dtype=torch.int64).expand(B, -1),
            "idx": i,
            "instance": [str(j) for j in range(B)],
            "camera_pose": torch.eye(4, device=device).unsqueeze(0).expand(B, -1, -1),
            "img_mask": torch.tensor([True], device=device).expand(B),
            "ray_mask": torch.tensor([False], device=device).expand(B),
            "update": torch.tensor([True], device=device).expand(B),
            "reset": torch.tensor([False], device=device).expand(B),
        }
        views.append(view)

    return views


class CUT3REncoder:
    """极简 CUT3R Encoder"""

    def __init__(self, model_path, device='cuda'):
        """
        Args:
            model_path: CUT3R 权重路径
            device: 设备
        """
        self.device = device

        # 加载模型
        print(f"Loading CUT3R from {model_path}")
        self.model = ARCroco3DStereo.from_pretrained(model_path).to(device)
        self.model.eval()
        print("✅ Model loaded")

         # ========== 🔥 状态缓存 ==========
        self.cached_state_feat = None
        self.cached_state_pos = None
        self.cached_mem = None
        self.cached_init_state_feat = None
        self.cached_init_mem = None
        print("✅ State cache initialized")

    def reset_state(self):
        """🔥 重置缓存状态"""
        self.cached_state_feat = None
        self.cached_state_pos = None
        self.cached_mem = None
        self.cached_init_state_feat = None
        self.cached_init_mem = None
        print("🔄 [CUT3REncoder] State cache cleared")

    def forward(self, views):
        """
        前向传播

        Args:
            views: prepare_input 返回的 views

        Returns:
            results: list of dicts，每个包含 pts3d_in_other_view, conf 等
            camera_tokens: [F*B, 1, 768]
            patch_tokens: [F*B, 729, 768]
        """
        # with torch.no_grad():
        #     # 使用官方 _forward_impl
        #     results, _ = self.model._forward_impl(views, ret_state=False)

        # 提取 camera 和 patch tokens（需要重新前向获取 dec）
        # 由于官方方法不直接返回 dec，我们需要手动提取
        with torch.no_grad():
            camera_tokens_list = []
            patch_tokens_list = []
            ress = []  # 官方的结果列表

            # 重新做一次前向来获取 tokens
            shape, feat_ls, pos = self.model._encode_views(views)
            feat = feat_ls[-1]
            # state_feat, state_pos = self.model._init_state(feat[0], pos[0])
            # mem = self.model.pose_retriever.mem.expand(feat[0].shape[0], -1, -1)
            # init_state_feat = state_feat.clone()
            # init_mem = mem.clone()
            # ========== 🔥 检查是否有缓存状态 ==========
            if self.cached_state_feat is not None:
                # 复用上一次的状态
                state_feat = self.cached_state_feat
                state_pos = self.cached_state_pos
                mem = self.cached_mem
                init_state_feat = self.cached_init_state_feat
                init_mem = self.cached_init_mem
                # print("🔄 Reusing cached state")
            else:
                # 初始化新状态
                state_feat, state_pos = self.model._init_state(feat[0], pos[0])
                mem = self.model.pose_retriever.mem.expand(feat[0].shape[0], -1, -1)
                init_state_feat = state_feat.clone()
                init_mem = mem.clone()

            for i in range(len(views)):
                feat_i = feat[i]
                pos_i = pos[i]

                if self.model.pose_head_flag:
                    global_img_feat_i = self.model._get_img_level_feat(feat_i)
                    if i == 0:
                        pose_feat_i = self.model.pose_token.expand(feat_i.shape[0], -1, -1)
                    else:
                        pose_feat_i = self.model.pose_retriever.inquire(global_img_feat_i, mem)
                    pose_pos_i = -torch.ones(
                        feat_i.shape[0], 1, 2, device=feat_i.device, dtype=pos_i.dtype
                    )
                else:
                    pose_feat_i = None
                    pose_pos_i = None

                new_state_feat, dec = self.model._recurrent_rollout(
                    state_feat, state_pos, feat_i, pos_i,
                    pose_feat_i, pose_pos_i, init_state_feat,
                    img_mask=views[i]["img_mask"],
                    reset_mask=views[i]["reset"],
                    update=views[i].get("update", None),
                )

                # 提取 tokens
                camera_tokens_list.append(dec[-1][:, :1])  # [B, 1, 768]
                patch_tokens_list.append(dec[-1][:, 1:])  # [B, 729, 768]
                
                out_pose_feat_i = dec[-1][:, 0:1]
                new_mem = self.model.pose_retriever.update_mem(
                    mem, global_img_feat_i, out_pose_feat_i
                )
                # 🔥 关键：从 dec 提取多层特征（完全按照官方）
                assert len(dec) == self.model.dec_depth + 1
                head_input = [
                    dec[0].float(),                                    # 最浅层
                    dec[self.model.dec_depth * 2 // 4][:, 1:].float(),  # 中层（只要patch tokens）
                    dec[self.model.dec_depth * 3 // 4][:, 1:].float(),  # 深层（只要patch tokens）
                    dec[self.model.dec_depth].float(),                 # 最深层（dec[-1]）
                ]
                
                # 🔥 调用官方的 downstream head（包含所有后处理）
                res = self.model._downstream_head(head_input, shape[i], pos=pos_i)
                ress.append(res)

                # 更新状态（与官方完全一致）
                img_mask = views[i]["img_mask"]
                update = views[i].get("update", None)
                if update is not None:
                    update_mask = (img_mask & update)
                else:
                    update_mask = img_mask
                update_mask = update_mask[:, None, None].float()

                state_feat = new_state_feat * update_mask + state_feat * (1 - update_mask)
                mem = new_mem * update_mask + mem * (1 - update_mask)
                
                reset_mask = views[i]["reset"]
                if reset_mask is not None:
                    reset_mask = reset_mask[:, None, None].float()
                    state_feat = init_state_feat * reset_mask + state_feat * (1 - reset_mask)
                    mem = init_mem * reset_mask + mem * (1 - reset_mask)

                # 更新状态
                # if self.model.pose_head_flag:
                #     out_pose_feat_i = dec[-1][:, 0:1]
                #     new_mem = self.model.pose_retriever.update_mem(mem, global_img_feat_i, out_pose_feat_i)
                # else:
                #     new_mem = mem

                # img_mask = views[i]["img_mask"]
                # update = views[i].get("update", None)
                # update_mask = (img_mask & update) if update is not None else img_mask
                # update_mask = update_mask[:, None, None].float()

                # state_feat = new_state_feat * update_mask + state_feat * (1 - update_mask)
                # mem = new_mem * update_mask + mem * (1 - update_mask)

                # reset_mask = views[i]["reset"]
                # if reset_mask is not None and reset_mask.any():
                #     reset_mask = reset_mask[:, None, None].float()
                #     state_feat = init_state_feat * reset_mask + state_feat * (1 - reset_mask)
                #     mem = init_mem * reset_mask + mem * (1 - reset_mask)

            # 拼接所有帧的 tokens
            camera_tokens = torch.cat(camera_tokens_list, dim=0)  # [F*B, 1, 768]
            patch_tokens = torch.cat(patch_tokens_list, dim=0)  # [F*B, 729, 768]

            self.cached_state_feat = state_feat.detach()
            self.cached_state_pos = state_pos.detach()
            self.cached_mem = mem.detach()
            self.cached_init_state_feat = init_state_feat.detach()
            self.cached_init_mem = init_mem.detach()

            return ress, camera_tokens, patch_tokens

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
    import open3d as o3d
    
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
    import open3d as o3d
    
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
        print(f"⚠️ 渲染点云失败: {e}")
        return np.zeros((image_size[0], image_size[1], 3), dtype=np.uint8)


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
        pointcloud_dir: 点云保存目录（Path对象）
        fps: 帧率
    """
    
    
    print(f"🎥 创建对比视频: {output_path}")
    
    if len(original_images) == 0:
        print("⚠️ 没有图像数据")
        return
    
    img_h, img_w = original_images[0].shape[:2]
    video_w = img_w * 2  # 原图 | 点云渲染
    video_h = img_h
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (video_w, video_h))
    
    for i, (orig_img, depth_data) in enumerate(tqdm(
        zip(original_images, depth_data_list),
        total=len(original_images),
        desc="    处理帧"
    )):
        # 🔥 从密集深度重建点云
        pcd = reconstruct_pointcloud_from_depth(
            depth_data['pts3d'],
            depth_data['rgb'],
            depth_data.get('conf', None),
            conf_threshold=0.5
        )
        
        # 🔥 保存点云 PLY 文件
        ply_path = pointcloud_dir / f"frame_{i:06d}.ply"
        o3d.io.write_point_cloud(str(ply_path), pcd)
        
        # 🔥 渲染点云为图像
        pc_rendered = render_pointcloud_to_image(pcd, image_size=(img_h, img_w))
        
        # 转换为 BGR
        orig_bgr = cv2.cvtColor(orig_img.astype(np.uint8), cv2.COLOR_RGB2BGR)
        pc_bgr = cv2.cvtColor(pc_rendered.astype(np.uint8), cv2.COLOR_RGB2BGR)
        
        # 添加标签
        cv2.putText(orig_bgr, "Original", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(pc_bgr, "Pointcloud", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.putText(orig_bgr, f"Frame {i}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(pc_bgr, f"Frame {i} | {len(pcd.points)} pts", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # 左右拼接
        combined = np.hstack([orig_bgr, pc_bgr])
        out.write(combined)
    
    out.release()
    print(f"✅ 视频已保存: {output_path}")
    print(f"✅ 点云已保存: {pointcloud_dir}")

def test():
    """测试 - 顺序处理10帧，每帧从两个目录拼接成(1,2,C,H,W)，测试状态迭代"""
    model_path = "/opt/liblibai-models/user-workspace2/users/lyh/Evo-1/Evo_1/spatial_encoder_checkpoint/cut3r_512_dpt_4_64.pth"
    dir_001 = Path("/opt/liblibai-models/user-workspace2/users/lyh/Evo-1/Evo_1/scripts/CUT3R/examples/001")
    dir_002 = Path("/opt/liblibai-models/user-workspace2/users/lyh/Evo-1/Evo_1/scripts/CUT3R/examples/002")
    device = "cuda:0"
    num_frames = 10

    print("\n" + "=" * 80)
    print("🎬 顺序处理10帧，每帧从两个目录拼接 (1,2,C,H,W) - 测试状态迭代")
    print("=" * 80)

    # 初始化
    encoder = CUT3REncoder(model_path, device)

    # 用于视频渲染
    original_images_001 = []
    original_images_002 = []
    base_depth_data_list = []
    wrist_depth_data_list = []

    # 创建输出目录
    output_dir = Path("/opt/liblibai-models/user-workspace2/users/lyh/Evo-1/Evo_1/cut3r_viz")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    video_dir = output_dir / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    
    pointcloud_dir = output_dir / "pointclouds"
    base_pc_dir = pointcloud_dir / "base"
    wrist_pc_dir = pointcloud_dir / "wrist"
    base_pc_dir.mkdir(parents=True, exist_ok=True)
    wrist_pc_dir.mkdir(parents=True, exist_ok=True)

    # 🔥 顺序处理每一帧
    for frame_idx in range(1, num_frames + 1):
        print(f"\n{'=' * 80}")
        print(f"📸 处理第 {frame_idx} 帧")
        print(f"{'=' * 80}")

        # 读取两个目录的同一帧
        img_path_001 = dir_001 / f"frame_{frame_idx:04d}.jpg"
        img_path_002 = dir_002 / f"frame_{frame_idx:04d}.jpg"

        if not img_path_001.exists() or not img_path_002.exists():
            print(f"⚠️ 文件不存在，跳过")
            continue

        # 加载图像
        img_001 = Image.open(img_path_001).convert('RGB')
        img_001_np = np.array(img_001)
        original_images_001.append(img_001_np.copy())

        img_002 = Image.open(img_path_002).convert('RGB')
        img_002_np = np.array(img_002)
        original_images_002.append(img_002_np.copy())

        print(f"  ✓ 001: {img_path_001.name}")
        print(f"  ✓ 002: {img_path_002.name}")

        # 转换为 tensor
        img_001_tensor = torch.from_numpy(img_001_np).permute(2, 0, 1)
        img_001_norm = img_001_tensor.to(torch.float32).to(device) / 127.5 - 1.0

        img_002_tensor = torch.from_numpy(img_002_np).permute(2, 0, 1)
        img_002_norm = img_002_tensor.to(torch.float32).to(device) / 127.5 - 1.0

        # 🔥 拼接成 (1, 2, C, H, W)
        pixel_values = torch.stack([img_001_norm, img_002_norm], dim=0).unsqueeze(0)
        print(f"  Batch shape: {pixel_values.shape}")

        # Prepare input
        views = prepare_input(pixel_values, device, target_size=432)

        # 🔥 Forward (状态会自动缓存和迭代)
        results, camera_tokens, patch_tokens = encoder.forward(views)

        print(f"  Output: camera_tokens={camera_tokens.shape}, patch_tokens={patch_tokens.shape}")

        # 🔥 保存深度数据
        for b in range(2):
            pts3d = results[0]["pts3d_in_other_view"][b].cpu().numpy()  # [H, W, 3]
            conf = results[0]["conf"][b].cpu().numpy()  # [H, W]
            img = views[0]["img"][b]
            rgb = (img.permute(1, 2, 0) * 0.5 + 0.5).clamp(0, 1).cpu().numpy()  # [H, W, 3]
            
            depth_data = {
                'pts3d': pts3d,
                'rgb': rgb,
                'conf': conf
            }
            
            if b == 0:
                base_depth_data_list.append(depth_data)
                print(f"  💾 View 0 (base): pts3d shape {pts3d.shape}, {np.isfinite(pts3d).all(axis=-1).sum()} valid points")
            else:
                wrist_depth_data_list.append(depth_data)
                print(f"  💾 View 1 (wrist): pts3d shape {pts3d.shape}, {np.isfinite(pts3d).all(axis=-1).sum()} valid points")

    # 🔥 渲染对比视频（原图 vs 点云渲染）
    print(f"\n{'=' * 80}")
    print(f"🎥 渲染对比视频...")
    print(f"{'=' * 80}")

    video_001 = str(video_dir / "sequential_001.mp4")
    create_pointcloud_comparison_video(
        original_images_001, 
        base_depth_data_list, 
        video_001,
        base_pc_dir,
        fps=5
    )

    video_002 = str(video_dir / "sequential_002.mp4")
    create_pointcloud_comparison_video(
        original_images_002, 
        wrist_depth_data_list, 
        video_002,
        wrist_pc_dir,
        fps=5
    )

    print("\n" + "=" * 80)
    print("✅ 全部完成!")
    print(f"  📹 视频:")
    print(f"    - Base: {video_001}")
    print(f"    - Wrist: {video_002}")
    print(f"  ☁️  点云:")
    print(f"    - Base: {base_pc_dir}/")
    print(f"    - Wrist: {wrist_pc_dir}/")
    print("=" * 80)
    print("\n💡 说明:")
    print("  - 每次forward处理 (1,2,C,H,W) batch")
    print("  - 保存密集 RGB + Depth + Conf 到内存")
    print("  - 从密集深度重建点云并保存 PLY 文件")
    print("  - 视频展示：原图 vs 点云渲染（验证深度质量）")
    print("  - 状态在10帧之间自动缓存和迭代")
    print("=" * 80)

if __name__ == "__main__":
    test()
