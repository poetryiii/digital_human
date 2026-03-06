import requests
import time
import sys
import json
from requests.exceptions import RequestException

# 配置
BASE_URL = "http://localhost:55071"
TIMEOUT = 10  # 单次请求超时时间（秒）
RETRY_INTERVAL = 2  # 轮询间隔（秒）
MAX_RETRIES = 5  # 提交任务最大重试次数

def print_color(text, color="white"):
    """带颜色输出，方便区分日志"""
    colors = {
        "red": "\033[91m",
        "green": "\033[92m",
        "yellow": "\033[93m",
        "blue": "\033[94m",
        "end": "\033[0m"
    }
    print(f"{colors.get(color, '')}{text}{colors['end']}")

def submit_task():
    """提交任务（带重试）"""
    # 准备测试文件（替换为你的文件路径）
    image_path = "1.png"
    audio_path = "1.mp3"
    text = "大家好，这是AI生成的数字人视频"

    # 验证文件存在
    import os
    if not os.path.exists(image_path):
        print_color(f"❌ 图片文件不存在: {image_path}", "red")
        return None
    if not os.path.exists(audio_path):
        print_color(f"❌ 音频文件不存在: {audio_path}", "red")
        return None

    # 多次重试提交
    for retry in range(MAX_RETRIES):
        try:
            print_color(f"\n[第 {retry+1} 次尝试] 提交任务...", "blue")
            files = {
                "image": open(image_path, "rb"),
                "reference_audio": open(audio_path, "rb")
            }
            data = {"text": text}
            
            # 发送请求（短超时，避免卡死）
            response = requests.post(
                f"{BASE_URL}/api/generate",
                files=files,
                data=data,
                timeout=TIMEOUT
            )
            
            if response.status_code == 200:
                result = response.json()
                task_id = result.get("task_id")
                print_color(f"✅ 任务提交成功！Task ID: {task_id}", "green")
                return task_id
            else:
                print_color(f"❌ 提交失败，状态码: {response.status_code}", "red")
                print_color(f"响应内容: {response.text}", "yellow")
                
        except RequestException as e:
            print_color(f"⚠️  请求异常: {str(e)}", "yellow")
            if retry < MAX_RETRIES - 1:
                print_color(f"等待 {RETRY_INTERVAL} 秒后重试...", "yellow")
                time.sleep(RETRY_INTERVAL)
            else:
                print_color(f"❌ 多次重试失败，退出", "red")
                return None
        finally:
            # 确保文件句柄关闭
            for f in files.values():
                f.close()

def poll_task_status(task_id):
    """轮询任务状态（支持优雅退出）"""
    if not task_id:
        return

    print_color(f"\n开始轮询任务状态 (Task ID: {task_id})...", "blue")
    print_color("提示：按 Ctrl+C 可停止轮询，任务仍会在服务端继续运行", "yellow")
    
    try:
        while True:
            try:
                response = requests.get(
                    f"{BASE_URL}/api/task/{task_id}",
                    timeout=TIMEOUT
                )
                
                if response.status_code == 200:
                    status = response.json()
                    status_text = status.get("status")
                    progress = status.get("progress", 0)
                    
                    if status_text == "completed":
                        video_url = BASE_URL + status.get("video_url")
                        print_color(f"\n🎉 任务完成！", "green")
                        print_color(f"视频地址: {video_url}", "green")
                        break
                    elif status_text == "failed":
                        error = status.get("error_msg")
                        print_color(f"\n❌ 任务失败: {error}", "red")
                        break
                    else:
                        print_color(f"\r⏳ 处理中... 进度: {progress}%", "blue", end="")
                        sys.stdout.flush()
                
                else:
                    print_color(f"\n⚠️  查询失败，状态码: {response.status_code}", "yellow")
                
            except RequestException as e:
                print_color(f"\n⚠️  查询异常: {str(e)}", "yellow")
            
            time.sleep(RETRY_INTERVAL)
            
    except KeyboardInterrupt:
        print_color(f"\n\n🛑 已停止轮询！", "yellow")
        print_color(f"任务仍在服务端运行，可手动查询：{BASE_URL}/api/task/{task_id}", "yellow")

if __name__ == "__main__":
    try:
        # 1. 提交任务
        task_id = submit_task()
        
        # 2. 轮询状态
        if task_id:
            poll_task_status(task_id)
            
    except KeyboardInterrupt:
        print_color(f"\n\n🛑 用户中断操作，程序退出", "yellow")
        sys.exit(0)
    except Exception as e:
        print_color(f"\n❌ 程序异常: {str(e)}", "red")
        sys.exit(1)