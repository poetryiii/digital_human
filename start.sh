#!/bin/sh
# ---------------- 强制使用 Bash 执行 ----------------
if [ -z "$BASH_VERSION" ]; then
    echo "检测到非 Bash 环境，正在尝试用 bash 重新执行..."
    if command -v bash >/dev/null 2>&1; then
        exec bash "$0" "$@"
    else
        echo "❌ 错误: 未找到 bash，请先安装 bash。"
        exit 1
    fi
fi

set -e

export HF_ENDPOINT=https://hf-mirror.com
REPO_LIVEPORTRAIT="KlingAIResearch/LivePortrait"
REPO_COSYVOICE="FunAudioLLM/CosyVoice"
REPO_MUSETALK="TMElyralab/MuseTalk"

# 智能寻找 Git
if [ -x /usr/bin/git ]; then
    GIT_CMD="/usr/bin/git"
else
    GIT_CMD="git"
fi

echo "========================================="
echo "  数字人工厂 - 全版本兼容最终版"
echo "========================================="

# 1. Python 环境
echo "[1/5] 配置 Python 环境..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi

# 使用 . 代替 source 兼容所有 sh
. venv/bin/activate

# 降级 huggingface_hub 到稳定版本 (避免新版API变更)
pip install --upgrade pip
pip install "huggingface_hub==0.24.6"  # 选择一个稳定版本，避免API变动

# 2. 克隆代码
echo "[2/5] 下载模型代码..."
mkdir -p models && cd models

# LivePortrait (核心模型，必须下载)
if [ ! -d "LivePortrait" ]; then
    GIT_LFS_SKIP_SMUDGE=1 "$GIT_CMD" clone https://github.com/KlingAIResearch/LivePortrait.git
fi

# CosyVoice (代码仓库公开)
if [ ! -d "CosyVoice" ]; then
    GIT_LFS_SKIP_SMUDGE=1 "$GIT_CMD" clone https://github.com/FunAudioLLM/CosyVoice.git
fi

# MuseTalk (代码仓库公开)
if [ ! -d "MuseTalk" ]; then
    GIT_LFS_SKIP_SMUDGE=1 "$GIT_CMD" clone https://github.com/TMElyralab/MuseTalk.git
fi
cd ..

# 3. 安装依赖
echo "[3/5] 安装 Python 库..."
pip install -r requirements.txt
pip install insightface onnxruntime-gpu onnxruntime diffusers safetensors funasr modelscope

# 4. 模型特定依赖
echo "[4/5] 补充模型依赖..."
cd models/LivePortrait && pip install -r requirements.txt 2>/dev/null || true && cd ../..
cd models/CosyVoice && pip install -r requirements.txt 2>/dev/null || true && cd ../..

# 5. 下载权重 (兼容所有 huggingface_hub 版本)
echo "[5/5] 下载模型权重 (兼容模式)..."

# 创建 Python 下载脚本 (移除特定异常类，改用通用捕获)
cat > download_models.py << 'EOF'
import os
from huggingface_hub import snapshot_download

# 配置
HF_ENDPOINT = os.getenv("HF_ENDPOINT", "https://hf-mirror.com")
os.environ["HF_ENDPOINT"] = HF_ENDPOINT

def download_model(repo_id, local_dir, exclude_patterns=None, required=False):
    """
    下载 HuggingFace 模型
    required: 是否为必须下载的核心模型 (False 则跳过失败)
    """
    # 检查目录是否已有文件
    if os.path.exists(local_dir):
        file_list = [f for f in os.listdir(local_dir) if not f.startswith('.')]
        if len(file_list) > 0:
            print(f"✅ {local_dir} 已存在文件，跳过下载")
            return True
    
    print(f"📥 开始下载 {repo_id} 到 {local_dir}")
    try:
        # 创建目录
        os.makedirs(local_dir, exist_ok=True)
        
        # 兼容新旧版本的 snapshot_download
        snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            ignore_patterns=exclude_patterns or [],
            resume_download=True,
            max_workers=4
        )
        print(f"✅ {repo_id} 下载完成")
        return True
    except Exception as e:
        # 通用异常捕获，不依赖特定异常类
        error_msg = str(e).lower()
        if required:
            print(f"❌ 核心模型 {repo_id} 下载失败，无法继续！错误: {e}")
            raise
        else:
            if "401" in error_msg or "unauthorized" in error_msg or "not found" in error_msg:
                print(f"⚠️  {repo_id} 下载失败 (授权/仓库不存在)，跳过该模型")
            else:
                print(f"⚠️  {repo_id} 下载失败: {e}，跳过该模型")
            return False

# 下载各个模型 (标记核心模型为 required=True)
if __name__ == "__main__":
    # LivePortrait (核心模型，必须下载)
    download_model(
        repo_id="KlingTeam/LivePortrait",
        local_dir="models/LivePortrait/pretrained_weights",
        exclude_patterns=["*.git*", "README.md", "docs", "assets"],
        required=True
    )
    
    # CosyVoice (非核心，跳过失败)
    download_model(
        repo_id="FunAudioLLM/CosyVoice",
        local_dir="models/CosyVoice/pretrained_models",
        required=False
    )
    
    # MuseTalk (非核心，跳过失败)
    download_model(
        repo_id="TMElyralab/MuseTalk",
        local_dir="models/MuseTalk/weights",
        required=False
    )
    
    print("\n📢 下载完成提示：")
    print("  - LivePortrait 核心权重已保证下载")
    print("  - CosyVoice/MuseTalk 如下载失败，请手动从官方渠道获取权重后放到对应目录")
EOF

# 执行 Python 下载脚本
python download_models.py

# 删除临时脚本
rm -f download_models.py

echo "========================================="
echo "  ✅ 部署完成！"
echo "  注意事项："
echo "  1. LivePortrait 核心权重已确保下载完成"
echo "  2. CosyVoice/MuseTalk 如需使用，请手动获取权重："
echo "     - CosyVoice: https://github.com/FunAudioLLM/CosyVoice"
echo "     - MuseTalk: https://github.com/TMElyralab/MuseTalk"
echo "  启动服务命令："
echo "  bash -c 'source venv/bin/activate && python main.py'"
echo "========================================="