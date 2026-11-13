#!/usr/bin/env bash
# fancap setup: 一键部署到 RK3588 (Ubuntu 22.04 aarch64) 或任意 Linux arm64/x64
# 用法: bash setup.sh
set -e

INSTALL_DIR="${FANCAP_DIR:-$HOME/fancap-work}"
ENV_NAME="${FANCAP_ENV:-fancap}"
PORT="${FANCAP_PORT:-8300}"
CONDA_DIR="${FANCAP_CONDA:-$HOME/miniconda3}"
RUN_USER="${SUDO_USER:-$USER}"

echo "==> fancap setup"
echo "    安装目录: $INSTALL_DIR"
echo "    conda 环境: $ENV_NAME"
echo "    服务端口: $PORT"

# ---------- 1. 依赖检查 ----------
for cmd in ffmpeg python3 git; do
  command -v "$cmd" >/dev/null || { echo "缺少 $cmd，请先安装: sudo apt install -y $cmd"; exit 1; }
done

# ---------- 2. conda 环境 ----------
if [ ! -x "$CONDA_DIR/bin/conda" ]; then
  echo "==> 未找到 conda，安装 miniconda…"
  wget -q https://github.com/anaconda/miniconda/releases/latest/download/Miniconda3-latest-Linux-aarch64.sh \
    -O /tmp/miniconda.sh || wget -q https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-latest-Linux-aarch64.sh -O /tmp/miniconda.sh
  bash /tmp/miniconda.sh -b -p "$CONDA_DIR"
fi
source "$CONDA_DIR/etc/profile.d/conda.sh"

if ! conda env list | grep -q "^$ENV_NAME "; then
  echo "==> 创建 conda 环境 $ENV_NAME (python 3.10)…"
  conda create -n "$ENV_NAME" python=3.10 -y
fi
conda activate "$ENV_NAME"

PYBIN="$CONDA_DIR/envs/$ENV_NAME/bin/python"

# ---------- 3. python 依赖 ----------
echo "==> 安装 python 依赖…"
pip install -q sherpa-onnx pysubs2 fastapi uvicorn python-multipart httpx python-dotenv

# sherpa-onnx 需 RKNN 构建版（带 NPU 支持）；清华镜像/PyPI 的默认 wheel 无 RKNN。
# 若板卡有 NPU，重装 rk-npu 源的 wheel：
if [ -e /sys/kernel/debug/rknpu/version ] || ls /usr/lib/librknnrt.so >/dev/null 2>&1 || ls /lib/librknnrt.so >/dev/null 2>&1; then
  echo "==> 检测到 Rockchip NPU runtime，安装 RKNN 构建的 sherpa-onnx…"
  pip install -q sherpa-onnx -f https://k2-fsa.github.io/sherpa/onnx/rk-npu-cn.html --force-reinstall --no-deps || \
    echo "    (可选) RKNN wheel 安装失败，将退回 CPU 推理"
else
  echo "==> 未检测到 NPU runtime，使用 CPU 推理（较慢）"
fi

# ---------- 4. 模型下载 ----------
MODELS_DIR="$INSTALL_DIR/models"
mkdir -p "$MODELS_DIR" "$INSTALL_DIR/input" "$INSTALL_DIR/output"
SV_URL="https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-rk3588-10-seconds-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2"
SV_DIR="$MODELS_DIR/sherpa-onnx-rk3588-10-seconds-sense-voice-zh-en-ja-ko-yue-2024-07-17"

if [ ! -f "$SV_DIR/model.rknn" ] && [ ! -f "$SV_DIR/model.int8.onnx" ]; then
  echo "==> 下载 SenseVoice 模型 (~400MB，需较长时间)…"
  wget -c -q --show-progress "$SV_URL" -O "$MODELS_DIR/sv.tar.bz2" || \
    wget -c -q --show-progress "https://gh-proxy.com/$SV_URL" -O "$MODELS_DIR/sv.tar.bz2"
  tar xjf "$MODELS_DIR/sv.tar.bz2" -C "$MODELS_DIR" && rm -f "$MODELS_DIR/sv.tar.bz2"
fi

if [ ! -f "$MODELS_DIR/silero_vad.onnx" ]; then
  echo "==> 下载 Silero VAD…"
  wget -c -q "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx" -O "$MODELS_DIR/silero_vad.onnx" || \
    wget -c -q "https://gh-proxy.com/https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx" -O "$MODELS_DIR/silero_vad.onnx"
fi

# ---------- 5. 代码 ----------
echo "==> 部署代码…"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cp "$SCRIPT_DIR/fancap.py" "$SCRIPT_DIR/server.py" "$SCRIPT_DIR/jellyfin.py" "$INSTALL_DIR/" 2>/dev/null || true

# 配置文件（已存在则不覆盖）
mkdir -p "$HOME/.config/fancap"
if [ ! -f "$HOME/.config/fancap/config.yaml" ]; then
  cat > "$HOME/.config/fancap/config.yaml" <<EOF
asr:
  backend: auto
  model_dir: $SV_DIR
translate:
  base_url: ""
  api_key: ""
  model: ""
  prompt: ""
jellyfin:
  base_url: ""
  api_key: ""
EOF
  echo "==> 已生成默认配置 $HOME/.config/fancap/config.yaml（稍后在 Web 设置页填写）"
fi

# ---------- 6. systemd service ----------
if [ "$(id -u)" = "0" ] && [ -n "$RUN_USER" ]; then
  echo "==> 安装 systemd service…"
  sed -e "s|User=orangepi|User=$RUN_USER|" \
      -e "s|WorkingDirectory=.*|WorkingDirectory=$INSTALL_DIR|" \
      -e "s|ExecStart=.*|ExecStart=$PYBIN -m uvicorn server:app --host 0.0.0.0 --port $PORT|" \
      "$SCRIPT_DIR/fancap.service" > /etc/systemd/system/fancap.service
  systemctl daemon-reload
  systemctl enable --now fancap
  echo "==> 服务已启用并启动 (端口 $PORT)"
else
  echo "==> [跳过 systemd] 需要 root 安装服务；可稍后手动执行:"
  echo "    sudo cp fancap.service /etc/systemd/system/ && sudo systemctl enable --now fancap"
  echo "    手动启动: cd $INSTALL_DIR && $PYBIN -m uvicorn server:app --host 0.0.0.0 --port $PORT"
fi

echo ""
echo "==> 完成! 浏览器访问 http://$(hostname -I | awk '{print $1}'):$PORT"
echo "    1. 设置页填入翻译端点（可选）"
echo "    2. 设置页填入 Jellyfin URL + API Key（可选）"
echo "    3. 提交视频或在 Jellyfin 里勾选剧集"
