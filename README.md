# fancap — 番剧中文字幕生成工具

为番剧自动生成中文字幕：语音识别跑在 **RK3588 NPU** 上（SenseVoice-Small, RKNN），翻译走可配置的 OpenAI 兼容端点，支持直接对接 **Jellyfin** 媒体库批量处理整季剧集。

## 功能

- **场景 1（无字幕）**：提取音频 → Silero VAD 切分 → SenseVoice NPU 识别 → LLM 翻译成中文 → 生成 srt/ass
- **场景 2a（有外文软字幕）**：直接抽取 → 翻译 → 生成新字幕
- **Jellyfin 集成**：浏览媒体库、勾选整季批量处理、自动跳过已有中文字幕的集、生成后直接回传到 Jellyfin
- **Web UI**：任务提交/进度/日志、翻译端点与 Prompt 在线编辑

## 环境要求

- Linux（aarch64 推荐，RK3588 已验证；x64 也可跑 CPU 版）
- Python 3.10+ / conda（可选但推荐）
- ffmpeg
- RK3588 NPU：需 `librknnrt.so`（驱动 ≥0.9.8 即官方最新推荐值）
- 无 NPU 时自动回退 CPU（int8 onnx，较慢）

## 快速开始

```bash
git clone https://github.com/minquejoe/fancap.git
cd fancap
sudo bash setup.sh        # 需要 root 装 systemd 服务；不 sudo 则跳过服务安装
```

`setup.sh` 会依次：检查依赖 → 建 conda 环境 `fancap` → 装 python 依赖（检测到 NPU 时自动装 RKNN 构建的 sherpa-onnx）→ 下载模型（SenseVoice ~400MB + Silero VAD）→ 部署代码 → 生成默认配置 → 安装并启动 systemd 服务。

完成后浏览器访问 `http://<板卡IP>:8300`。

### 手动启动（不用 systemd）

```bash
conda activate fancap
cd ~/fancap-work
python -m uvicorn server:app --host 0.0.0.0 --port 8300
```

## 配置

配置文件：`~/.config/fancap/config.yaml`（Web 设置页在线编辑，保存即热生效）

```yaml
asr:
  backend: auto                # auto | rknn | onnx-cpu
  model_dir: ~/fancap-work/models/sherpa-onnx-rk3588-10-seconds-sense-voice-...
translate:
  base_url: "https://api.deepseek.com/v1"   # 任何 OpenAI 兼容端点（vLLM/Ollama/LiteLLM 均可）
  api_key: "sk-..."
  model: "deepseek-chat"
  prompt: ""                   # 留空用默认提示词；自定义模板见下
jellyfin:
  base_url: "http://192.168.1.x:8096"
  api_key: "xxxxxxxxxxxxxxxx"
```

### 翻译 Prompt

默认提示词（番剧优化：逐条翻译、口语化、口癖一致）：

```text
你是番剧字幕翻译。把下面的{src}字幕翻译成简体中文。
要求：
- 逐条翻译，保持行数一致，每行一条
- 保持口语化、符合番剧语气，人称/口癖前后一致
- 只输出翻译结果，不要解释

字幕：
{lines}
```

- `{src}` = 源语言（日语/英语等，自动检测）
- `{lines}` = 每批 40 行字幕
- Web 设置页可直接编辑并保存，留空恢复默认
- **未填 api_key 时跳过翻译，直接输出原文**

### Jellyfin API Key

Jellyfin 控制台 → 控制台 → **API 密钥** → 添加。生成后填入 Web 设置页。
注意：字幕上传需要该 Key 所属账号有**媒体库管理/上传字幕权限**（管理员账号默认有）。

## 使用

### Web UI（推荐）

访问 `http://<板卡IP>:8300`：

1. **本地任务**：上传视频/音频 → 自动（识别+翻译）或仅翻译
2. **Jellyfin**：填 URL + API Key → 测试连接 → 浏览媒体库 → 选剧 → 勾选集（无中文的默认勾选）→ 处理选中
3. 任务队列实时显示进度（抽音频/VAD/识别/翻译/上传），已上传 Jellyfin 的集显示 `已上传 Jellyfin`
4. 本地任务的成品在任务队列点下载（srt + ass 双格式）

### CLI（可选）

```bash
python fancap.py transcribe <视频或音频> <输出前缀>   # 识别 + 翻译 + 生成 srt/ass
python fancap.py translate <字幕文件> <输出前缀>      # 已有外文字幕直接翻译
```

## 性能参考（RK3588 实测）

| 项目 | 数值 |
|---|---|
| ASR 引擎 | SenseVoice-Small（RKNN fp16，NPU 三核） |
| 识别速度 | RTF ≈ 0.09（5s 音频 ~0.45s） |
| 一集 24 分钟番剧 | 全流程 ~2.5 分钟（不含翻译网络耗时） |
| 内存占用 | 模型加载 ~1GB，其余忽略不计 |

## 已知限制

- RKNN 模型固定 10s 输入 → VAD 切分后每段 ≤10s（自动处理）
- Jellyfin 字幕上传仅支持 srt（API 限制），ass 样式需本地任务输出
- 日语识别偶有同音字错误（模型限制）；翻译环节 LLM 看上下文可部分修正
- 已上传的字幕如需重新生成：在 Jellyfin 里删除该字幕轨道后重跑该集

## 目录结构

```
fancap/
├── fancap.py          # 核心 pipeline：ffmpeg → VAD → SenseVoice → 翻译 → srt/ass
├── server.py          # FastAPI Web UI + 任务队列 + Jellyfin API
├── jellyfin.py        # Jellyfin 客户端（浏览/流式/字幕上传）
├── fancap.service     # systemd 服务模板
├── setup.sh           # 一键部署脚本
└── README.md
```

## 故障排查

| 现象 | 处理 |
|---|---|
| RKNN 报 `Invalid RKNN model version` | 升级 `/usr/lib/librknnrt.so`（rknn-toolkit2 仓库的 runtime） |
| RKNN 报 `exportDataSize large then model size` | 代码已通过子进程隔离规避；确认跑的是最新 server.py |
| Jellyfin 401 | Key 无效或账号权限不足；确认用 `Authorization: MediaBrowser Token="..."` 头 |
| 字幕上传 405 | 端点是 `POST /Videos/{itemId}/Subtitles`（JSON + base64 Data），确认 jellyfin.py 是最新版 |
| GitHub 下载慢 | setup.sh 自动尝试 gh-proxy.com 镜像；也可手动下载模型后放到 models/ |

## 致谢

- [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx)（ASR 运行时 + RKNN 支持）
- [SenseVoice](https://github.com/FunAudioLLM/SenseVoice)（FunASR 语音识别模型）
- [Silero VAD](https://github.com/snakers4/silero-vad)（语音活动检测）
- [pysubs2](https://github.com/tkarabela/pysubs2)（字幕读写）
