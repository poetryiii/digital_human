digital_human_factory/
├── main.py                # FastAPI 主入口
├── worker.py              # GPU 任务调度与核心逻辑
├── schemas.py             # 数据模型定义
├── start.sh               # 一键部署脚本
├── requirements.txt       # Python 依赖
├── config.py              # 路径配置
├── models/                # 第三方模型存放目录
│   ├── LivePortrait/
│   ├── CosyVoice/
│   └── MuseTalk/
└── data/                  # 数据目录
    ├── inputs/
    └── outputs/

pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu118

echo "[6/6] 启动服务..."
echo "环境配置完成！"
echo "运行 'source venv/bin/activate && python main.py' 启动服务器。"


给一套【图片+自定义音色生成数字人】完整工业级方案，包含完整代码

包含：

✔ FastAPI 接口
✔ CosyVoice 音色克隆
✔ MuseTalk 口型同步
✔ LivePortrait 动作驱动
✔ start.sh 自动安装部署及其自动下载模型
✔ API 调用示例
✔ GPU 调度

注意：其中LivePortrait的代码地址是https://github.com/KlingAIResearch/LivePortrait/，他那个模型LivePortrait/checkpoints/face_encoder.onnx是404，如果需要使用的话，请使用最新版代码库里的使用案例，旧版不可用