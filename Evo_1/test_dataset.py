
import sys
sys.path.append('.')

from dataset.lerobot_dataset_cut3r import LeRobotDatasetCut3r
from dataset.lerobot_dataset_pretrain_mp import LeRobotDataset 
import yaml

print("=" * 80)
print("初始化数据集...")
print("=" * 80)

# 加载原始Evo-1数据集
with open("dataset/config.yaml", 'r') as f:
    dataset_config = yaml.safe_load(f)

dataset_evo1 = LeRobotDataset(
    config=dataset_config,
    image_size=448,
    action_horizon=50,
    binarize_gripper=False,
    use_augmentation=True  # 先关闭增强方便对比
)

# 加载CUT3R数据集
dataset_cut3r = LeRobotDatasetCut3r(
    image_size=448,
    action_horizon=50,
    use_augmentation=True
)

print(f"✅ 原始Evo-1: {len(dataset_evo1)} samples")
print(f"✅ CUT3R版本: {len(dataset_cut3r)} samples")

# 对比同一个索引的样本
print("\n" + "=" * 80)
print("对比 dataset[0]")
print("=" * 80)

sample_evo1 = dataset_evo1[0]
sample_cut3r = dataset_cut3r[0]

# 打印对比
print("\n📋 返回的keys:")
print(f"   Evo-1:  {list(sample_evo1.keys())}")
print(f"   CUT3R:  {list(sample_cut3r.keys())}")

print("\n📊 Images:")
print(f"   Evo-1:  shape={sample_evo1['images'].shape}, dtype={sample_evo1['images'].dtype}, range=[{sample_evo1['images'].min():.3f}, {sample_evo1['images'].max():.3f}]")
print(f"   CUT3R:  shape={sample_cut3r['images'].shape}, dtype={sample_cut3r['images'].dtype}, range=[{sample_cut3r['images'].min():.3f}, {sample_cut3r['images'].max():.3f}]")
print(f"   相同: {sample_evo1['images'].shape == sample_cut3r['images'].shape}")

print("\n📝 Prompt:")
print(f"   Evo-1:  '{sample_evo1['prompt']}'")
print(f"   CUT3R:  '{sample_cut3r['prompt']}'")
print(f"   相同: {sample_evo1['prompt'] == sample_cut3r['prompt']}")

print("\n📊 State:")
print(f"   Evo-1:  shape={sample_evo1['state'].shape}, dtype={sample_evo1['state'].dtype}, range=[{sample_evo1['state'].min():.3f}, {sample_evo1['state'].max():.3f}]")
print(f"   CUT3R:  shape={sample_cut3r['state'].shape}, dtype={sample_cut3r['state'].dtype}, range=[{sample_cut3r['state'].min():.3f}, {sample_cut3r['state'].max():.3f}]")

print("\n🎯 Action:")
print(f"   Evo-1:  shape={sample_evo1['action'].shape}, dtype={sample_evo1['action'].dtype}, range=[{sample_evo1['action'].min():.3f}, {sample_evo1['action'].max():.3f}]")
print(f"   CUT3R:  shape={sample_cut3r['action'].shape}, dtype={sample_cut3r['action'].dtype}, range=[{sample_cut3r['action'].min():.3f}, {sample_cut3r['action'].max():.3f}]")
print(f"   Horizon相同: {sample_evo1['action'].shape[0] == sample_cut3r['action'].shape[0] == 50}")

print("\n🎭 Masks:")
print(f"   image_mask  - Evo-1: {sample_evo1['image_mask'].tolist()}")
print(f"   image_mask  - CUT3R: {sample_cut3r['image_mask'].tolist()}")
print(f"   state_mask  - Evo-1: sum={sample_evo1['state_mask'].sum().item()}/24")
print(f"   state_mask  - CUT3R: sum={sample_cut3r['state_mask'].sum().item()}/24")
print(f"   action_mask - Evo-1: sum={sample_evo1['action_mask'].sum().item()}/{sample_evo1['action_mask'].numel()}")
print(f"   action_mask - CUT3R: sum={sample_cut3r['action_mask'].sum().item()}/{sample_cut3r['action_mask'].numel()}")

print("\n🤖 Embodiment ID:")
print(f"   Evo-1:  {sample_evo1['embodiment_id'].item()}")
print(f"   CUT3R:  {sample_cut3r['embodiment_id'].item()}")

print("\n🔮 Spatial Tokens (CUT3R独有):")
if 'spatial_tokens' in sample_cut3r:
    print(f"   shape={sample_cut3r['spatial_tokens'].shape}, dtype={sample_cut3r['spatial_tokens'].dtype}")
    print(f"   range=[{sample_cut3r['spatial_tokens'].min():.3f}, {sample_cut3r['spatial_tokens'].max():.3f}]")
    print(f"   View0 non-zero: {(sample_cut3r['spatial_tokens'][0] != 0).any()}")
    print(f"   View1 non-zero: {(sample_cut3r['spatial_tokens'][1] != 0).any()}")
else:
    print(f"   ❌ 未找到spatial_tokens")

# 快速对比几个索引
print("\n" + "=" * 80)
print("快速对比多个索引")
print("=" * 80)

for idx in [0, 10, 100]:
    if idx >= len(dataset_evo1) or idx >= len(dataset_cut3r):
        continue
    s1 = dataset_evo1[idx]
    s2 = dataset_cut3r[idx]
    
    imgs_same = (s1['images'].shape == s2['images'].shape)
    prompt_same = (s1['prompt'] == s2['prompt'])
    action_same = (s1['action'].shape == s2['action'].shape)
    
    status = "✅" if (imgs_same and prompt_same and action_same) else "❌"
    print(f"[{idx}] {status} - images:{imgs_same}, prompt:{prompt_same}, action:{action_same}")

print("\n" + "=" * 80)
print("数据集配置:")
print("=" * 80)
print(f"   图像大小: 448x448")
print(f"   Action Horizon: 50")
print(f"   图像增强: False (测试时关闭)")
print(f"   图像范围: [0, 1] (ToTensor自动转换)")
print(f"   归一化: [-1, 1] (min-max归一化)")

print("\n✅ 测试完成")
