import os
import uuid
import shutil
import json
import time
import torch
import sys
import subprocess
import signal
from contextlib import contextmanager
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from tqdm import tqdm

# ===================== 全局配置 =====================
# 定义目录路径（确保这些目录存在）
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE_DIR, "data", "inputs")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "outputs")
LIVE_PORTRAIT_PATH = os.path.join(BASE_DIR, "models", "LivePortrait")
PROGRESS_DIR = os.path.join(OUTPUT_DIR, ".progress")

# 创建必要目录
for dir_path in [INPUT_DIR, OUTPUT_DIR, PROGRESS_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# 提前添加模型路径（关键：将LivePortrait加入Python路径）
sys.path.insert(0, LIVE_PORTRAIT_PATH)
sys.path.append(os.path.join(BASE_DIR, "models", "CosyVoice"))
sys.path.append(os.path.join(BASE_DIR, "models", "MuseTalk"))

# 提前加载核心依赖
print("[全局初始化] 加载核心依赖...", end="")
try:
    import cv2
    import librosa
    import soundfile as sf
    # 加载LivePortrait核心模块（替代subprocess调用）
    from src.config.argument_config import ArgumentConfig
    from src.config.inference_config import InferenceConfig
    from src.config.crop_config import CropConfig
    from src.live_portrait_pipeline import LivePortraitPipeline
    print(" ✅")
except Exception as e:
    print(f" ❌ 依赖加载失败: {str(e)[:100]}")
    raise

# ===================== 工具函数 =====================
@contextmanager
def timeout(seconds):
    """超时保护装饰器"""
    def raise_timeout(signum, frame):
        raise TimeoutError(f"操作超时（{seconds}秒）")
    signal.signal(signal.SIGALRM, raise_timeout)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)

def save_task_progress(task_id: str, progress: int, status: str, error_msg: str = "", video_url: str = ""):
    """保存任务进度到文件"""
    progress_file = os.path.join(PROGRESS_DIR, f"{task_id}.json")
    data = {
        "task_id": task_id,
        "progress": progress,
        "status": status,  # running/completed/failed
        "error_msg": error_msg,
        "video_url": video_url,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(progress_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_task_progress(task_id: str):
    """读取任务进度"""
    progress_file = os.path.join(PROGRESS_DIR, f"{task_id}.json")
    if not os.path.exists(progress_file):
        return None
    with open(progress_file, "r", encoding="utf-8") as f:
        return json.load(f)

def partial_fields(target_class, kwargs):
    """过滤并初始化指定配置类（仅保留类中存在的字段）"""
    return target_class(**{k: v for k, v in kwargs.items() if hasattr(target_class, k)})

def fast_check_ffmpeg():
    """快速检查FFmpeg是否安装"""
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

# ===================== 核心生成函数 =====================
def generate_digital_human_sync(img_path: str, text: str, ref_audio_path: str):
    """同步生成数字人（完整版）- 直接调用LivePortrait核心逻辑"""
    task_id = str(uuid.uuid4())
    save_task_progress(task_id, 0, "running")
    
    # 创建总进度条
    total_progress = tqdm(total=100, desc="数字人生成总进度", 
                         bar_format="{l_bar}{bar}| {n:.1f}%/{total:.1f}% [{elapsed}<{remaining}]")
    
    try:
        # --- 步骤 1: 音频转换 (10%) ---
        total_progress.set_description("音频格式转换中")
        total_progress.update(10)
        save_task_progress(task_id, 10, "running")
        
        output_audio = os.path.join(OUTPUT_DIR, f"{task_id}_audio.wav")
        y, sr = librosa.load(ref_audio_path, sr=16000, duration=10)  # 限制10秒
        sf.write(output_audio, y, sr)

        # --- 步骤 2: 驱动视频准备 (20%) ---
        total_progress.set_description("驱动视频准备中")
        total_progress.update(10)
        save_task_progress(task_id, 20, "running")
        
        drive_video_path = os.path.join(LIVE_PORTRAIT_PATH, "assets/examples/driving/d0.mp4")
        if not os.path.exists(drive_video_path):
            raise Exception(f"驱动视频不存在: {drive_video_path}")

        # --- 步骤 3: 设备初始化 (30%) ---
        total_progress.set_description("设备初始化中")
        total_progress.update(10)
        save_task_progress(task_id, 30, "running")
        
        # ========== 核心修复1：设备配置（区分CPU/GPU，避免cuda:-1） ==========
        use_gpu = False  # 强制CPU模式（如需GPU改为True）
        if use_gpu and torch.cuda.is_available():
            device = "cuda"
            device_id = 0  # GPU ID只能是非负整数（0,1,2...）
        else:
            device = "cpu"
            device_id = 0  # CPU模式下device_id设为0（避免负数）
        total_progress.write(f"📌 使用计算设备: {device.upper()} (device_id={device_id})")

        # --- 步骤 4: LivePortrait 动作迁移 (80%) ---
        total_progress.set_description("动作迁移计算中")
        total_progress.update(50)
        save_task_progress(task_id, 80, "running")
        
        # 准备临时文件
        output_video_temp = os.path.join(OUTPUT_DIR, f"{task_id}_temp.mp4")
        final_video = os.path.join(OUTPUT_DIR, f"{task_id}_final.mp4")
        
        # ========== 核心修复2：移除output_name，仅保留ArgumentConfig标准参数 ==========
        # 1. 设置FFmpeg路径（兼容官方逻辑）
        ffmpeg_dir = os.path.join(LIVE_PORTRAIT_PATH, "ffmpeg")
        if os.path.exists(ffmpeg_dir):
            os.environ["PATH"] += os.pathsep + ffmpeg_dir
        
        # 2. 检查FFmpeg
        if not fast_check_ffmpeg():
            raise ImportError(
                "FFmpeg is not installed. Please install FFmpeg (including ffmpeg and ffprobe) before running this script. https://ffmpeg.org/download.html"
            )
        
        # 3. 构建ArgumentConfig参数（仅保留官方标准参数）
        args_dict = {
            "source": img_path,                # 源图片（对应官方-s参数）
            "driving": drive_video_path,      # 驱动视频（对应官方-d参数）
            "output_dir": OUTPUT_DIR,          # 输出目录（官方标准参数）
            "device_id": device_id,            # 修复：CPU/GPU均使用非负整数
            "fps_out": 10,                     # 输出帧率
            "flag_crop_driving_video": False,  # 不裁剪驱动视频
            "scale_crop_driving_video": 2.2,
            "vx_ratio_crop_driving_video": 0.0,
            "vy_ratio_crop_driving_video": -0.1
        }
        
        # 关键：过滤掉ArgumentConfig不存在的参数（双重保险）
        valid_args = {k: v for k, v in args_dict.items() if hasattr(ArgumentConfig, k)}
        # 手动初始化ArgumentConfig（模拟tyro解析）
        args = ArgumentConfig(**valid_args)
        
        # 4. 初始化配置类
        inference_cfg = partial_fields(InferenceConfig, args.__dict__)
        crop_cfg = partial_fields(CropConfig, args.__dict__)
        
        # ========== 核心修复3：手动指定设备（避免LivePortrait内部拼接错误设备字符串） ==========
        # 强制设置设备为CPU/GPU，覆盖配置类中的默认逻辑
        if hasattr(inference_cfg, 'device'):
            inference_cfg.device = device
        if hasattr(crop_cfg, 'device'):
            crop_cfg.device = device
        
        # 5. 创建Pipeline并执行（核心逻辑）
        with timeout(600):  # 10分钟超时
            live_portrait_pipeline = LivePortraitPipeline(
                inference_cfg=inference_cfg,
                crop_cfg=crop_cfg
            )
            live_portrait_pipeline.execute(args)
        
        # 验证LivePortrait输出（适配官方默认输出命名规则）
        temp_video_files = [
            f for f in os.listdir(OUTPUT_DIR) 
            if f.endswith(".mp4") and os.path.getctime(os.path.join(OUTPUT_DIR, f)) > (time.time() - 300)
        ]
        if not temp_video_files:
            raise Exception(f"LivePortrait生成失败，未找到临时视频文件")
        # 取最新生成的视频文件
        temp_video_path = os.path.join(OUTPUT_DIR, sorted(temp_video_files, key=lambda x: os.path.getctime(os.path.join(OUTPUT_DIR, x)))[-1])
        # 重命名到指定临时路径（兼容原有逻辑）
        shutil.move(temp_video_path, output_video_temp)
        total_progress.write(f"✅ LivePortrait 执行完成: {output_video_temp}")

        # --- 步骤 5: 合并音频 (95%) ---
        total_progress.set_description("音频视频合并中")
        total_progress.update(15)
        save_task_progress(task_id, 95, "running")
        
        # 使用ffmpeg合并音频
        ffmpeg_cmd = [
            "ffmpeg", "-y",
            "-i", output_video_temp,
            "-i", output_audio,
            "-c:v", "libx264",
            "-c:a", "aac",
            "-shortest",
            final_video
        ]
        subprocess.run(ffmpeg_cmd, capture_output=True, check=True)

        # --- 步骤 6: 清理和验证 (100%) ---
        total_progress.set_description("生成完成")
        total_progress.update(5)
        save_task_progress(task_id, 100, "completed", video_url=f"/results/{os.path.basename(final_video)}")
        
        # 清理临时文件
        if os.path.exists(output_video_temp):
            os.remove(output_video_temp)
        if os.path.exists(output_audio):
            os.remove(output_audio)
        
        # 验证视频
        if not os.path.exists(final_video) or os.path.getsize(final_video) < 1024:
            raise Exception("生成的视频文件无效")

        total_progress.write(f"🎉 数字人视频生成完成: {final_video}")
        total_progress.close()
        
        return {
            "status": "success",
            "task_id": task_id,
            "video_path": final_video,
            "video_url": f"/results/{os.path.basename(final_video)}"
        }

    except TimeoutError as e:
        total_progress.set_description("处理超时 ❌")
        total_progress.update(100 - total_progress.n)
        total_progress.write(f"⚠️  处理超时: {e}")
        total_progress.close()
        
        # 兜底生成静态视频
        final_video = os.path.join(OUTPUT_DIR, f"{task_id}_final.mp4")
        img = cv2.imread(img_path)
        if img is None:
            raise Exception("源图片读取失败，无法生成兜底静态视频")
        height, width = img.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(final_video, fourcc, 10.0, (width, height))
        for i in range(100):
            out.write(img)
        out.release()
        
        save_task_progress(task_id, 100, "completed", 
                          error_msg=f"处理超时，已生成静态视频: {str(e)}",
                          video_url=f"/results/{os.path.basename(final_video)}")
        raise Exception(f"处理超时，已生成静态视频: {e}")

    except Exception as e:
        total_progress.set_description("生成失败 ❌")
        total_progress.update(100 - total_progress.n)
        error_msg = str(e)[:200]
        total_progress.write(f"❌ 生成失败: {error_msg}")
        total_progress.close()
        
        save_task_progress(task_id, 0, "failed", error_msg=error_msg)
        raise Exception(error_msg)

# ===================== FastAPI 接口 =====================
app = FastAPI(title="Digital Human Factory API", version="1.0")

# 挂载输出目录（用于访问生成的视频）
app.mount("/results", StaticFiles(directory=OUTPUT_DIR), name="results")

@app.post("/api/generate", summary="提交数字人生成任务")
async def submit_task(
    image: UploadFile = File(...),
    text: str = Form(...),
    reference_audio: UploadFile = File(...)
):
    try:
        # 生成任务ID并保存上传文件
        task_id = str(uuid.uuid4())
        img_ext = image.filename.split('.')[-1]
        audio_ext = reference_audio.filename.split('.')[-1]
        
        # 保存文件
        img_path = os.path.join(INPUT_DIR, f"{task_id}_img.{img_ext}")
        audio_path = os.path.join(INPUT_DIR, f"{task_id}_ref.{audio_ext}")
        
        with open(img_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)
        with open(audio_path, "wb") as buffer:
            shutil.copyfileobj(reference_audio.file, buffer)
        
        # 同步执行生成任务（实际生产可改用 Celery）
        result = generate_digital_human_sync(img_path, text, audio_path)
        
        return JSONResponse({
            "code": 200,
            "msg": "任务执行成功",
            "data": {
                "task_id": result["task_id"],
                "status": "completed",
                "progress": 100,
                "video_url": result["video_url"]
            }
        })
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "code": 500,
                "msg": f"任务执行失败: {str(e)}",
                "data": None
            }
        )

@app.get("/api/task/{task_id}", summary="查询任务状态")
async def get_task_status(task_id: str):
    # 读取进度文件
    progress_data = load_task_progress(task_id)
    if progress_data:
        return JSONResponse({
            "code": 200,
            "msg": "查询成功",
            "data": progress_data
        })
    
    # 兼容逻辑：直接查找视频文件
    video_files = [f for f in os.listdir(OUTPUT_DIR) if f.startswith(task_id) and f.endswith('.mp4')]
    if video_files:
        return JSONResponse({
            "code": 200,
            "msg": "查询成功",
            "data": {
                "task_id": task_id,
                "status": "completed",
                "progress": 100,
                "video_url": f"/results/{video_files[0]}",
                "error_msg": ""
            }
        })
    else:
        return JSONResponse({
            "code": 200,
            "msg": "查询成功",
            "data": {
                "task_id": task_id,
                "status": "failed",
                "progress": 0,
                "video_url": "",
                "error_msg": "任务未找到或执行失败"
            }
        })

@app.get("/api/health", summary="健康检查")
async def health_check():
    return JSONResponse({
        "code": 200,
        "msg": "服务运行正常",
        "data": {
            "service": "digital-human-factory",
            "status": "running",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
    })

# ===================== 启动服务 =====================
if __name__ == "__main__":
    import uvicorn
    # 查找可用端口（避免端口占用）
    def find_free_port():
        import socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', 0))
            return s.getsockname()[1]
    
    port = find_free_port()
    print(f"\n🚀 服务启动中，端口: {port}")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )