import torch
import os
import shutil
import yaml
from pathlib import Path
from torchvision.transforms import ToPILImage
import matplotlib.pyplot as plt

# 假设你的类定义在 lerobot_dataset_cut3r.py 中
try:
    from lerobot_dataset_cut3r_multiframe import LeRobotDatasetCUT3R
except ImportError:
    print("❌ 错误: 找不到 lerobot_dataset_cut3r.py，请确保文件在当前目录下")
    exit()

def verify_dataset():
    # ================= 配置区域 (请根据你的实际路径修改) =================
    # 你的数据集配置文件路径 (.yaml)
    DATASET_CONFIG_PATH = "./config_libero_spatial.yaml" 
    
    # 模拟传入的 config 字典
    # 注意：这里需要根据你实际 yaml 的结构构造，或者直接读取 yaml
    
    with open(DATASET_CONFIG_PATH, 'r') as f:
        dataset_config = yaml.safe_load(f)
   

    # ================= 初始化数据集 =================
    print("\n🚀 初始化数据集...")
    dataset = LeRobotDatasetCUT3R(
        config=dataset_config,
        image_size=448,
        #max_samples_per_file=10, # 限制样本数，加快初始化速度
        action_horizon=50,
        #num_history_frames=3,    # 历史帧数 (总长=4)
        binarize_gripper=False,
        use_augmentation=False,   # 关闭增强以验证原始像素
        #cache_dir="./debug_cache" # 临时缓存目录
    )

    print(f"数据集加载完成，共有 {len(dataset)} 个样本")
    if len(dataset) == 0:
        print("❌ 数据集为空，请检查路径配置")
        return

    # ================= 获取样本 =================
    sample_idx = len(dataset) // 2 # 取中间的一个样本，避免取到开头全是padding的情况
    print(f"🔍 正在读取样本 idx={sample_idx} ...")
    
    item = dataset[sample_idx]
    
    images = item['images']       # expect [4, 3, 3, 448, 448]
    image_mask = item['image_mask'] # expect [4, 3]
    prompt = item['prompt']

    # ================= 验证形状 =================
    print("\n📊 Tensor 形状检查:")
    print(f"  Shape expected: [4, 3, 3, 448, 448]")
    print(f"  Shape actual:   {list(images.shape)}")
    # 新增：打印张量范围统计
    print(f"\n📈 Tensor 数值范围:")
    print(f"  Dtype: {images.dtype}")
    print(f"  Min:   {images.min().item():.4f}")
    print(f"  Max:   {images.max().item():.4f}")
    print(f"  Mean:  {images.mean().item():.4f}")
    
    # 检查归一化范围（常见的像素值范围）
    if images.max() <= 1.0 and images.min() >= 0.0:
        print(f"  Range: [0, 1] (已归一化)")
    elif images.max() <= 255.0 and images.min() >= 0.0:
        print(f"  Range: [0, 255] (原始像素值)")
    else:
        print(f"  Range: 非标准范围，建议检查预处理逻辑")
    
    if list(images.shape) == [4, 3, 3, 448, 448]:
        print("\n✅ 形状验证通过!")
    else:
        print("\n❌ 形状验证失败!") 
    if list(images.shape) == [4, 3, 3, 448, 448]:
        print("形状验证通过!")
    else:
        print("形状验证失败!")

    print(f"\n🎭 Image Mask (Valid Views):")
    print(image_mask)

    # ================= 保存图片进行可视化 =================
    output_dir = Path("verify_output")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir()

    print(f"\n正在保存图片到 {output_dir}/ ...")
    
    to_pil = ToPILImage()
    
    # 遍历时间步 T (0-3)
    for t in range(images.shape[0]):
        # 遍历视角 V (0-2)
        for v in range(images.shape[1]):
            # 检查 mask，如果是 padding 的全黑帧可以跳过或者标记
            is_valid = image_mask[t, v].item()
            status = "valid" if is_valid else "padding"
            
            img_tensor = images[t, v] # [3, H, W]
            
            # 保存文件: T0_V0_valid.jpg
            # T0, T1, T2 是历史，T3 是当前帧
            time_label = "History" if t < 3 else "CURRENT"
            filename = f"T{t}_{time_label}_View{v}_{status}.jpg"
            
            save_path = output_dir / filename
            
            # 还原并保存
            try:
                img_pil = to_pil(img_tensor)
                img_pil.save(save_path)
                print(f"  -> Saved {filename}")
            except Exception as e:
                print(f"  ❌ Save failed for {filename}: {e}")

    print("\n✅ 验证结束！请打开 verify_output 文件夹查看图片时序。")
    print("   应该看到 T0 -> T1 -> T2 -> T3(CURRENT) 的动作是连贯的。")

if __name__ == "__main__":
    verify_dataset()
