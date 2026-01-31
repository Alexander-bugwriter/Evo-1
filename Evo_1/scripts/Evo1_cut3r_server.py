# evo1_server_json.py

import sys
import os
import asyncio
import websockets
import numpy as np
import cv2
import json
import torch
from PIL import Image
from torchvision import transforms
from fvcore.nn import FlopCountAnalysis



sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from scripts.Evo1_cut3r import EVO1

class ImageBuffer:
    def __init__(self, window_size=4, num_views=3, target_size=448):
        self.window_size = window_size
        self.num_views = num_views
        self.target_size = target_size
        # 存储预处理后的 Tensor 列表，每个元素形状为 [V, C, H, W]
        self.buffer = []
        
        # 预处理转换（必须与训练时的 Normalize 严格一致）
        self.transform = transforms.Compose([
            transforms.Resize((target_size, target_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])
    def push(self, pil_images: List[Image.Image]):
        """
        接收当前时刻的视角图像列表（通常是 3 张）
        """
        if len(pil_images) != self.num_views:
            print(f"Warning: Expected {self.num_views} views, but got {len(pil_images)}")

        # 1. 预处理并堆叠当前时刻视角: [V, C, H, W]
        current_t_tensors = []
        for img in pil_images:
            # 确保是 RGB
            if img.mode != 'RGB':
                img = img.convert('RGB')
            current_t_tensors.append(self.transform(img))
        
        current_t_tensor = torch.stack(current_t_tensors) 

        # 2. 压入队列
        self.buffer.append(current_t_tensor)

        # 3. 超过窗口长度则弹出最旧的
        if len(self.buffer) > self.window_size:
            self.buffer.pop(0)
    def get_stacked_tensor(self):
        """
        返回组织好的 [F, V, C, H, W] 张量
        """
        if not self.buffer:
            return None
            
        # --- 核心修改：低于长度填充逻辑 ---
        # 如果当前 buffer 只有 1 帧，我们要复制它 3 次凑齐 4 帧
        # 如果有 2 帧，我们要复制第一帧 2 次凑齐 4 帧，以此类推
        temp_buffer = list(self.buffer) # 浅拷贝防止修改原始 buffer
        
        while len(temp_buffer) < self.window_size:
            # 始终拿当前队列里最老的一帧（index 0）填充在最前面
            temp_buffer.insert(0, temp_buffer[0])
            
        # 最终堆叠成 [4, 3, 3, 448, 448]
        return torch.stack(temp_buffer)
    def clear(self):
        """用于任务重置或切换场景"""
        self.buffer = []

class Normalizer:
    def __init__(self, stats_or_path):
        if isinstance(stats_or_path, str):
            with open(stats_or_path, "r") as f:
                stats = json.load(f)
        else:
            stats = stats_or_path

        def pad_to_24(x):
            x = torch.tensor(x, dtype=torch.float32)
            if x.shape[0] < 24:
                pad = torch.zeros(24 - x.shape[0], dtype=torch.float32)
                x = torch.cat([x, pad], dim=0)
            elif x.shape[0] > 24:
                raise ValueError(f"Input length {x.shape[0]} exceeds expected 24")
            return x


        if "state" in stats and "actions" in stats:
            # 新格式：直接是 {"state": {...}, "actions": {...}}
            robot_stats = stats
            state_key = "state"
            action_key = "actions"
        elif len(stats) == 1:
            robot_key = list(stats.keys())[0]
            robot_stats = stats[robot_key]
            state_key = "observation.state"
            action_key = "action"
        else:
            raise ValueError(f"norm_stats.json should contain only one robot key, but: {list(stats.keys())}")


        self.state_min = pad_to_24(robot_stats[state_key]["min"])
        self.state_max = pad_to_24(robot_stats[state_key]["max"])
        self.action_min = pad_to_24(robot_stats[action_key]["min"])
        self.action_max = pad_to_24(robot_stats[action_key]["max"])
        # robot_key = list(stats.keys())[0]
        # robot_stats = stats[robot_key]

        # self.state_min = pad_to_24(robot_stats["observation.state"]["min"])
        # self.state_max = pad_to_24(robot_stats["observation.state"]["max"])
        # self.action_min = pad_to_24(robot_stats["action"]["min"])
        # self.action_max = pad_to_24(robot_stats["action"]["max"])

    def normalize_state(self, state: torch.Tensor) -> torch.Tensor:
        state_min = self.state_min.to(state.device, dtype=state.dtype)
        state_max = self.state_max.to(state.device, dtype=state.dtype)
        return torch.clamp(2 * (state - state_min) / (state_max - state_min + 1e-8) - 1, -1.0, 1.0)

    def denormalize_action(self, action: torch.Tensor) -> torch.Tensor:
        action_min = self.action_min.to(action.device, dtype=action.dtype)
        action_max = self.action_max.to(action.device, dtype=action.dtype)
        if action.ndim == 1:
            action = action.view(1, -1)
        return (action + 1.0) / 2.0 * (action_max - action_min + 1e-8) + action_min


def load_model_and_normalizer(ckpt_dir):
    config = json.load(open(os.path.join(ckpt_dir, "config.json")))
    stats = json.load(open(os.path.join(ckpt_dir, "norm_stats.json")))

    config["finetune_vlm"] = False
    config["finetune_action_head"] = False
    config["finetune_fusion_block"] = False
    config["num_inference_timesteps"] = 32

    # 🔥 硬编码启动 CUT3R
    config["use_cut3r"] = True
    config["training"] = False

    model = EVO1(config).eval()
    ckpt_path = os.path.join(ckpt_dir, "mp_rank_00_model_states.pt")

    checkpoint = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(checkpoint["module"], strict=False)
    model = model.to("cuda")

    normalizer = Normalizer(stats)
    return model, normalizer



def decode_image_from_list(img_list):
    img_array = np.array(img_list, dtype=np.uint8)
    img = cv2.resize(img_array, (448, 448))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(img)
    return transforms.ToTensor()(pil).to("cuda")


global_image_buffer = ImageBuffer(window_size=4, num_views=3)
def infer_from_json_dict(data: dict, model, normalizer):
    device = "cuda"
    model_dtype = next(model.parameters()).dtype

    if data.get("reset", False):
        print("🔄 Received RESET signal, resetting memory state...")
        try:
            global_image_buffer.clear()
        except:
            print("No memory buffer to reset.")
  
    # images = [decode_image_from_list(img) for img in data["image"]]
    # assert len(images) == 3, "Must provide exactly 3 images."
    # for img in images:
    #     assert img.shape == (3, 448, 448), "image_size must be (3,448,448)"
    current_images = [decode_image_from_list(img) for img in data["image"]]
    assert len(current_images) == 3, "Must provide exactly 3 images."
    for img in current_images:
        assert img.shape == (3, 448, 448), "image_size must be (3,448,448)"
    
    
    # 2. 更新缓存
    global_image_buffer.push(current_images)
    
    # 3. 获取 [F, V, C, H, W] 张量，并增加 Batch 维度变为 [1, 4, 3, 3, 448, 448]
    multiframe_images_tensor = global_image_buffer.get_stacked_tensor()
    multiframe_images_tensor = multiframe_images_tensor.unsqueeze(0).to(model.device)
    print(f"multiframe_images_tensor.shape,{multiframe_images_tensor.shape}")

    current_img_mask = torch.tensor(data["image_mask"], dtype=torch.bool, device=device)
    multiframe_image_mask = current_img_mask.unsqueeze(0).repeat(4, 1)  # [V] -> [F, V]
    multiframe_image_mask = multiframe_image_mask.unsqueeze(0)  # [1, F, V]
    print(f"multiframe_image_mask,{multiframe_image_mask}")
    print(f"multiframe_image_mask.shape,{multiframe_image_mask.shape}")

    state = torch.tensor(data["state"], dtype=torch.float32, device=device)
    if state.ndim == 1:
        state = state.unsqueeze(0)
    if state.shape[1] < 24:
        state = torch.cat([state, torch.zeros((1, 24 - state.shape[1]), device=device)], dim=1)
    norm_state = normalizer.normalize_state(state).to(dtype=torch.float32)

    
    prompt = data["prompt"]
    image_mask = torch.tensor(data["image_mask"], dtype=torch.int32, device=device)
    action_mask = torch.tensor([data["action_mask"]],dtype=torch.int32, device=device)

    print(f"image_mask,{image_mask}")
    print(f"action_mask,{action_mask}")
    
    with torch.no_grad() and torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        action = model.run_inference(
            images=multiframe_images_tensor,
            image_mask=multiframe_image_mask,
            prompt=prompt,
            state_input=norm_state,
            action_mask=action_mask,
            
        )
        action = action.reshape(1, -1, 24)
        action = normalizer.denormalize_action(action[0])
        return action.cpu().numpy().tolist()
# def infer_from_json_dict(data: dict, model, normalizer):
#     device = "cuda"
#     model_dtype = next(model.parameters()).dtype

#     if data.get("reset", False):
#         print("🔄 Received RESET signal, resetting CUT3R state...")
#         try:
#             model.cut3r_encoder.reset_state()
#         except:
#             print("No spatial encoder")
  
#     images = [decode_image_from_list(img) for img in data["image"]]
#     assert len(images) == 3, "Must provide exactly 3 images."
#     for img in images:
#         assert img.shape == (3, 448, 448), "image_size must be (3,448,448)"

 
#     state = torch.tensor(data["state"], dtype=torch.float32, device=device)
#     if state.ndim == 1:
#         state = state.unsqueeze(0)
#     if state.shape[1] < 24:
#         state = torch.cat([state, torch.zeros((1, 24 - state.shape[1]), device=device)], dim=1)
#     norm_state = normalizer.normalize_state(state).to(dtype=torch.float32)

    
#     prompt = data["prompt"]
#     image_mask = torch.tensor(data["image_mask"], dtype=torch.int32, device=device)
#     action_mask = torch.tensor([data["action_mask"]],dtype=torch.int32, device=device)

#     print(f"image_mask,{image_mask}")
#     print(f"action_mask,{action_mask}")
    
#     with torch.no_grad() and torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
#         action = model.run_inference(
#             images=images,
#             image_mask=image_mask,
#             prompt=prompt,
#             state_input=norm_state,
#             action_mask=action_mask,
            
#         )
#         action = action.reshape(1, -1, 24)
#         action = normalizer.denormalize_action(action[0])
#         return action.cpu().numpy().tolist()


#async def handle_request(websocket, model, normalizer):
#    print("Client connected")
#    try:
#        async for message in websocket:
#            json_data = json.loads(message)
#            print(f"Received JSON observation")
#            actions = infer_from_json_dict(json_data, model, normalizer)
#            await websocket.send(json.dumps(actions))
#            print("Sent action chunk")
#    except websockets.exceptions.ConnectionClosed:
#        print("Client disconnected.")

def handle_state_update_request(data: dict):
    #print("🔄 STATE UPDATE: Calling CUT3R...", end=" ", flush=True)
    #start_time = time.time()
  
    # 解码图像
    images = [decode_image_from_list(img) for img in data["image"]]
    #print("\nDecoded images (before CUT3R):")
    #for i, img in enumerate(images):
    #    print(f"    Image {i}: shape={img.shape}, range=[{img.min().item():.4f}, {img.max().item():.4f}]")

    # 🔥 调用 _extract_spatial_features 更新CUT3R状态
    with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        spatial_tokens = model._extract_spatial_features(images)

    #elapsed = (time.time() - start_time) * 1000  # ms

    #print(f"✅  Done in {elapsed:.1f}ms")
    #print(f"  - Spatial tokens shape: {spatial_tokens.shape}")
    #print(f"  - dtype: {spatial_tokens.dtype}")

    return {"status": "state_updated"}

async def handle_request(websocket, model, normalizer):
    print("✅ Client connected")
    try:
        async for message in websocket:
            json_data = json.loads(message)
            
            # 🔥 检查是否只需要更新状态
            if json_data.get("update_only", False):
                print("[STATE UPDATE MODE]")
                result = handle_state_update_request(json_data)
                await websocket.send(json.dumps(result))
                print("Sent confirmation: state_updated\n")
            else:
                # 完整推理
                print("[FULL INFERENCE MODE]")
                actions = infer_from_json_dict(json_data, model, normalizer)
                await websocket.send(json.dumps(actions))
                print(f"Sent action chunk ({len(actions)} actions)\n")
    except websockets.exceptions.ConnectionClosed:
        print("Client disconnected.")


# === 启动服务 ===
if __name__ == "__main__":
    # ckpt_dir = "Your/Path/To/Checkpoint"
    #Example: ckpt_dir = "/home/dell/checkpoints/Evo1/Evo1_MetaWorld/"
    
    #ckpt_dir = "/opt/liblibai-models/user-workspace2/users/lyh/model_checkpoint/Evo1/lyh_train_cut3r_stage_2/step_best"
    ckpt_dir = "/opt/liblibai-models/user-workspace2/users/lyh/model_checkpoint/Evo1/Evo1_cut3r_3stages_stage3/step_80000"
    port = 9000

    print("Loading EVO_1 model...")
    model, normalizer = load_model_and_normalizer(ckpt_dir)

    async def main():
        print(f"EVO_1 server running at ws://0.0.0.0:{port}")
        async with websockets.serve(
            lambda ws: handle_request(ws, model, normalizer),
            "0.0.0.0", port, max_size=100_000_000
        ):
            await asyncio.Future()

    asyncio.run(main())
