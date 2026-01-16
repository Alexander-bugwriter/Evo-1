import asyncio
import websockets
import numpy as np
import json
import pathlib
import os
import logging
import math
import imageio
import random

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
os.environ["MUJOCO_GL"] = "osmes"
#os.environ["MUJOCO_GL"] = "egl"
LIBERO_DUMMY_ACTION = [0.0] * 6 + [0.0]

import argparse  # 添加到 imports
import datetime
import re
# 在 Args 类定义之前添加
parser = argparse.ArgumentParser()
parser.add_argument('--ckpt_name', type=str, 
                    default=f"Evo1_cut3r_libero_all",
                    help='Checkpoint name for logs and videos')
cmd_args = parser.parse_args()
######################################
class Args():
    horizon = 14
    max_steps = [25,25, 25, 95] 
    SERVER_URL = "ws://0.0.0.0:9000"
    #ckpt_name = f"Evo1_full_cut3r_libero_all"  
    ckpt_name = cmd_args.ckpt_name
    task_suites = ["libero_spatial", "libero_object", "libero_goal", "libero_10"] 
    log_file = f"./log_file/{ckpt_name}.txt"
    num_episodes = 10
    SEED = 42
    ping_interval = 60  # 每 60 秒发送一次 ping（默认是 20 秒）
    ping_timeout = 60   # 等待 pong 的超时时间（默认是 20 秒）
    close_timeout = 30  # 关闭连接的超时时间    
    

args = Args()

########################################

os.makedirs(os.path.dirname(args.log_file), exist_ok=True)
# ========= Logging =========
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        
        logging.FileHandler(args.log_file, mode='a'),
        logging.StreamHandler()
    ]

)
log = logging.getLogger(__name__)

def parse_last_completed(log_file):
    if not os.path.exists(log_file):
        return None, 0, -1, 0, 0, 0
    
    with open(log_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    last_suite = None
    last_task = None
    last_episode = -1
    suite_start_idx = -1
    task_start_idx = -1
    
    # 从后往前找
    for i in range(len(lines) - 1, -1, -1):
        line = lines[i]
        
        # 1. 找最后完成的 episode
        if last_episode == -1:
            match = re.search(r'Task (\d+) \| Episode (\d+): [✅❌]', line)
            if match:
                last_task = int(match.group(1))
                last_episode = int(match.group(2)) - 1
        
        # 2. 找最后的 task 开始位置
        if task_start_idx == -1 and last_task is not None:
            match = re.search(r'Start task(\d+):', line)
            if match and int(match.group(1)) - 1 == last_task:
                task_start_idx = i
        
        # 3. 找最后的 suite 开始位置
        if suite_start_idx == -1:
            match = re.search(r'Start task suite (\w+)', line)
            if match:
                last_suite = match.group(1)
                suite_start_idx = i
                break
    
    # 统计 suite 级别（从 suite 开始到现在）
    suite_success = 0
    suite_episodes = 0
    if suite_start_idx != -1:
        for line in lines[suite_start_idx:]:
            match = re.search(r'Task \d+ \| Episode \d+: ([✅❌])', line)
            if match:
                suite_episodes += 1
                if '✅' in match.group(1):
                    suite_success += 1
    
    # 统计 task 级别（从当前 task 开始到现在）
    task_success = 0
    if task_start_idx != -1:
        for line in lines[task_start_idx:]:
            match = re.search(r'Task \d+ \| Episode \d+: ([✅❌])', line)
            if match:
                if '✅' in match.group(1):
                    task_success += 1
    
    return (last_suite, 
            last_task if last_task is not None else 0, 
            last_episode,
            suite_success, suite_episodes, 
            task_success)

# ========= Photos to list[list[list[int]]] =========
def encode_image_array(img_array: np.ndarray):
    return img_array.astype(np.uint8).tolist()

# ========= Quaternion to Axis-Angle =========
def quat2axisangle(quat):
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den

# ========= Observation to JSON-compatible dict =========
def obs_to_json_dict(obs, prompt, resize_size=448,reset=False):
    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    dummy_proc = np.zeros((resize_size, resize_size, 3), dtype=np.uint8)
    #print(f"[DEBUG] agentview: shape={img.shape}, range=[{img.min()}, {img.max()}]")
    #print(f"[DEBUG] wrist: shape={wrist_img.shape}, range=[{wrist_img.min()}, {wrist_img.max()}]")
    data = {
        "image": [
            encode_image_array(img),
            encode_image_array(wrist_img),
            encode_image_array(dummy_proc)
        ],
        "state": np.concatenate((
            obs["robot0_eef_pos"],
            quat2axisangle(obs["robot0_eef_quat"]),
            obs["robot0_gripper_qpos"],
        )).tolist(),
        "prompt": prompt,
        "image_mask": [1, 1, 0],
        "action_mask": [1] * 7 + [0] * 17,
        "reset": reset,  # 🔥 NEW: reset 字段
    }
    return data

# ========= Get the environment of LIBERO =========
def get_libero_env(task, resolution=448, seed=args.SEED):
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env, task_description

# ========= Save the video log =========
def save_video(frames, filename="simulation.mp4", fps=20, save_dir="videos_2"):
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, filename)

    if len(frames) > 0:
        imageio.mimsave(filepath, frames, fps=fps)
        print(f"Video saved: {filepath} ({len(frames)} frames)")
    else:
        log.warning(f"⚠️ No frames to save. File not created: {filepath}")

# ========= Main Function =========
# async def run(SERVER_URL: str, max_steps: int = None, num_episodes: int = None, horizon = None, task_suite_name = None):
async def run(SERVER_URL: str, max_steps: int = None, num_episodes: int = None, horizon = None, task_suite_name = None, start_task_id=0, start_episode=0, prev_suite_success=0, prev_suite_episodes=0, prev_task_success=0):
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks

    print(f"Numbers of tasks: {num_tasks_in_suite}")

    # total_success = 0
    # total_episodes = 0
    # total_steps = 0
    total_success = prev_suite_success  # 恢复 suite 统计
    total_episodes = prev_suite_episodes
    total_steps = 0

    async with websockets.connect(
        SERVER_URL,
        ping_interval=args.ping_interval,
        ping_timeout=args.ping_timeout,
        close_timeout=args.close_timeout
    ) as ws:
        log.info(f"===========================Start task suite {task_suite_name}========================")

        for task_id in range(num_tasks_in_suite):
            if task_id < start_task_id:
                continue
            # 如果是恢复的 task，继承之前的 task_success
            if task_id == start_task_id:
                task_success = prev_task_success
            else:
                task_success = 0
            print(f"task_id{task_id}")
            #if task_id+1 not in [1,5,7,9] :
             #   continue

            task = task_suite.get_task(task_id)
            initial_states = task_suite.get_task_init_states(task_id)
            env, task_description = get_libero_env(task, resolution=448, seed=args.SEED)

            log.info(f"\n========= Start task{task_id+1}: {task_description} =========")

            #task_success = 0
            task_episodes = min(num_episodes, len(initial_states))
            # 计算从哪个 episode 开始
            ep_start = start_episode + 1 if task_id == start_task_id else 0
            
            # 🔥 如果这个 task 已经完成，跳过
            if ep_start >= task_episodes:
                log.info(f"⏭️  Task {task_id} already completed, skipping")
                continue            
            # 🔥 初始化 task_success（注意：删掉原来那行 task_success = 0）
            task_success = prev_task_success if task_id == start_task_id else 0

            # for ep in range(task_episodes):
            #ep_start = start_episode + 1 if task_id == start_task_id else 0  # 从完成的下一个开始
            for ep in range(ep_start, task_episodes):
                print(f"\n===== Task {task_id} | Episode {ep+1} =====")

                env.reset()


                obs = env.set_init_state(initial_states[ep])
                t = 0
                while t < 10:
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        

                prompt = str(task_description)
                
                reset_data = obs_to_json_dict(obs, prompt, reset=True)
                await ws.send(json.dumps(reset_data))
                print(f"[Episode {ep+1}] Sent RESET signal")
                result= await ws.recv()
                print("Received reset signal")
                # 等待服务器确认（可选，但建议加上）
                #_ = await ws.recv()
                #print(f"[Episode {ep+1}] Reset complete, action discarded")

                print(prompt)
                episode_done = False
                max_step = 0
                frames = []

                for step in range(max_steps):
                    max_step += 1

                    send_data = obs_to_json_dict(obs, prompt)
                    await ws.send(json.dumps(send_data))
                    #print(f"[Step {step}] Send observation")

                    result = await ws.recv()
                    try:
                        action_list = json.loads(result)
                        actions = np.array(action_list)
                        #print(f"[Step {step}] recivied actions (shape={actions[0][6]})")
                    except Exception as e:
                        print(f"❌ Action parsing failed: {e}, content: {result}")
                        break

                    
                    for i in range(horizon):
                        action = actions[i].tolist()
                        #print(action[:7])
                        if action[6]>0.5:
                            action[6] = -1
                        else:
                            action[6] = 1
                        
                        # action[6] = abs(1.0 - action[6])
                        
                        #print(f"gripper action", action[6])
                        try:
                            obs, reward, done, info = env.step(action[:7])
                        except ValueError as ve:
                            print(f"❌ the action is not valid: {ve}")
                            episode_done = False
                            break

                        
                        frame = np.hstack([
                            np.rot90(obs["agentview_image"], 2),
                            np.rot90(obs["robot0_eye_in_hand_image"], 2)
                        ])
                        frames.append(frame)

                        #print(f"[Step {step}] reward={reward:.2f}, done={done}")
                        if done:
                            print("Task completed")
                            episode_done = True
                            task_success += 1
                            total_success += 1
                            total_steps += max_step
                            break
                    if episode_done:
                        break

                
                save_video(frames, f"task{task_id+1}_episode{ep+1}.mp4", fps=30, save_dir=f"./video_log_file/{args.ckpt_name}/{task_suite_name}")

                if episode_done:
                    log.info(f"Task {task_id} | Episode {ep+1}: ✅ Success")
                else:
                    log.info(f"Task {task_id} | Episode {ep+1}: ❌ Fail")

                # exit(0)

            log.info(f"========= Task {task_id + 1} Summary: {task_success}/{task_episodes} Successful =========")
            total_episodes += task_episodes

        # ======= Overall Summary =======
        log.info("\n========= Overall Task Summary =========")
        log.info(f"✅ Total Successful Episodes: {total_success}/{total_episodes}")
        if total_episodes > 0:
            log.info(f"📊 Average Steps: {total_steps / total_episodes:.2f}")




# if __name__ == "__main__":
#     np.random.seed(args.SEED)
#     random.seed(args.SEED)
    
#     for name, max_steps in zip(args.task_suites, args.max_steps):
#         asyncio.run(run(SERVER_URL = args.SERVER_URL,
#                         max_steps=max_steps, 
#                         num_episodes=args.num_episodes,
#                         horizon=args.horizon,
#                         task_suite_name=name))
if __name__ == "__main__":
    np.random.seed(args.SEED)
    random.seed(args.SEED)
    
    # 解析上次运行位置
    resume_suite, resume_task, resume_episode, suite_succ, suite_eps, task_succ = parse_last_completed(args.log_file)

    if resume_suite:
        log.info(f"🔄 Resume: {resume_suite} task{resume_task} ep{resume_episode+1}")
        log.info(f"📊 Suite: {suite_succ}/{suite_eps}, Task: {task_succ}")
    suite_started = (resume_suite is None)
    
    for name, max_steps in zip(args.task_suites, args.max_steps):
        # 跳过已完成的 suite
        if not suite_started:
            if name == resume_suite:
                suite_started = True
            else:
                continue
        
        # 传入起始位置
        start_task = resume_task if name == resume_suite else 0
        start_ep = resume_episode if name == resume_suite else -1
        ps = suite_succ if name == resume_suite else 0
        pe = suite_eps if name == resume_suite else 0
        ts = task_succ if name == resume_suite else 0
        asyncio.run(run(SERVER_URL = args.SERVER_URL,
                        max_steps=max_steps, 
                        num_episodes=args.num_episodes,
                        horizon=args.horizon,
                        task_suite_name=name,
                        start_task_id=start_task,
                        start_episode=start_ep, prev_suite_success=ps, prev_suite_episodes=pe, prev_task_success=ts))
