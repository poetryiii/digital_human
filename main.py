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
PROGRESS_DIR = os.path.join(OUTPUT_DIR, ".progress")

# 🔥 关键修正：LIVE_PORTRAIT_PATH 指向包含 assets 的目录（不是demo.py的目录）
LIVE_PORTRAIT_PATH = os.path.join(BASE_DIR, "models", "LivePortrait")
# 🔥 关键修正：DEMO_SCRIPT_PATH 直接指向同级的 demo.py
DEMO_SCRIPT_PATH = os.path.join(BASE_DIR, "demo.py")  # demo.py和main.py同级

# 创建必要目录
for dir_path in [INPUT_DIR, OUTPUT_DIR, PROGRESS_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# 提前添加模型路径
sys.path.append(LIVE_PORTRAIT_PATH)
sys.path.append(os.path.join(BASE_DIR, "models", "CosyVoice"))
sys.path.append(os.path.join(BASE_DIR, "models", "MuseTalk"))

# 提前加载核心依赖
print("[全局初始化] 加载核心依赖...", end="")
try:
    import cv2
    import librosa
    import soundfile as sf
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

# ===================== 核心生成函数 =====================
def generate_digital_human_sync(img_path: str, text: str, ref_audio_path: str):
    """同步生成数字人（完整版）"""
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
        # 🔥 新增：检查同级的 demo.py 是否存在
        if not os.path.exists(DEMO_SCRIPT_PATH):
            raise Exception(f"demo.py 文件不存在！路径: {DEMO_SCRIPT_PATH}\n请确认 demo.py 和 main.py 在同一级目录")
        if not os.path.exists(drive_video_path):
            raise Exception(f"驱动视频不存在！路径: {drive_video_path}\n请检查 LivePortrait/assets 目录")

        # --- 步骤 3: 设备初始化 (30%) ---
        total_progress.set_description("设备初始化中")
        total_progress.update(10)
        save_task_progress(task_id, 30, "running")
        
        device = "cpu"  # 强制CPU更稳定，如需GPU改为 "cuda"
        total_progress.write(f"📌 使用计算设备: {device.upper()}")

        # --- 步骤 4: LivePortrait 动作迁移 (80%) ---
        total_progress.set_description("动作迁移计算中")
        total_progress.update(50)
        save_task_progress(task_id, 80, "running")
        
        # 准备临时文件
        output_video_temp = os.path.join(OUTPUT_DIR, f"{task_id}_temp.mp4")
        final_video = os.path.join(OUTPUT_DIR, f"{task_id}_final.mp4")
        
        # 🔥 修正：使用同级的 demo.py 路径
        cmd = [
            sys.executable,
            DEMO_SCRIPT_PATH,  # 直接用同级的 demo.py
            "--source_image", img_path,
            "--driving_video", drive_video_path,  # 驱动视频仍在 models/LivePortrait/assets 下
            "--output", output_video_temp,
            "--device", device,
            "--fps", "10"
        ]
        
        # 执行 LivePortrait
        with timeout(600):  # 10分钟超时
            result = subprocess.run(
                cmd,
                cwd=LIVE_PORTRAIT_PATH,
                capture_output=True,
                text=True,
                check=True
            )
        # total_progress.write(f"✅ LivePortrait 执行完成: {result.stdout[:200]}")
        # 🔥 新增：打印完整的输出和错误日志
        total_progress.write(f"✅ LivePortrait STDOUT: {result.stdout}")
        total_progress.write(f"❌ LivePortrait STDERR: {result.stderr}")  # 打印完整错误日志

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

    except subprocess.CalledProcessError as e:
        total_progress.set_description("执行失败 ❌")
        total_progress.update(100 - total_progress.n)
        error_msg = f"LivePortrait执行失败: {e.stderr[:200]}"
        total_progress.write(f"❌ {error_msg}")
        total_progress.close()
        
        # 兜底生成静态视频
        final_video = os.path.join(OUTPUT_DIR, f"{task_id}_final.mp4")
        img = cv2.imread(img_path)
        height, width = img.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(final_video, fourcc, 10.0, (width, height))
        for i in range(100):
            out.write(img)
        out.release()
        
        save_task_progress(task_id, 0, "failed", error_msg=error_msg)
        raise Exception(error_msg)

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
        
        # 异步执行生成任务（同步调用，实际生产可改用 Celery）
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