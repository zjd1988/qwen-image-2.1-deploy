# Qwen-Image-2.1 部署验证目录

> 面向 agent/维护者的完整复现步骤与边界条件见 [AGENTS.md](./AGENTS.md)。

基于上游仓库 [QwenLM/Qwen-Image-2.1](https://github.com/QwenLM/Qwen-Image-2.1) 搭建，包含两条已验证的推理路线：
**Diffusers 基线**（官方推荐）和 **vLLM-Omni 在线服务**。

## 全新机器复现步骤

前置条件：NVIDIA GPU（≥24GB 显存）、驱动 ≥ 570（CUDA 12.8，支持 cu129 构建；cu13 需驱动 ≥ 580）、
Python 3.12、约 45GB 空闲磁盘。国内网络建议 `export HF_ENDPOINT=https://hf-mirror.com`，
PyPI 用阿里云镜像 `https://mirrors.aliyun.com/pypi/simple`。

```bash
# 0. 克隆本仓库（仓库根目录即部署目录）并安装 uv
git clone https://github.com/zjd1988/qwen-image-2.1-deploy.git
cd qwen-image-2.1-deploy
curl -LsSf https://astral.sh/uv/install.sh | sh && source ~/.local/bin/env

# 1. 下载模型权重（31GB，含 text_encoder/transformer/vae/scheduler/processor）
bash scripts/download_model.sh          # 基座
bash scripts/download_model.sh turbo    # 可选：8 步加速版（已有基座时可用
                                        # scripts/merge_te_shards.py 免去 17.5GB TE 下载）

# 2. Diffusers 基线环境
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python \
    --index-url https://download.pytorch.org/whl/cu129 torch==2.13.0 torchvision==0.28.0
uv pip install --python .venv/bin/python \
    --index-url https://mirrors.aliyun.com/pypi/simple \
    'transformers>=5.17' accelerate pillow safetensors sentencepiece
uv pip install --python .venv/bin/python 'git+https://github.com/huggingface/diffusers'
# 注意：diffusers 必须装 git main，发布版没有 QwenImage21Pipeline

# 3. Diffusers 冒烟验证（1024x1024 / 20 步，约 1 分钟，产出 outputs/verify_t2i.png）
bash scripts/run_verify.sh

# 4. vLLM-Omni 环境（clone PR#7759 分支 + editable 安装）
bash scripts/omni_setup.sh
# 关键修正：PyPI 的 vllm 0.29.0 是 cu13 构建，在驱动 570 上会报
# "CUDA driver version is insufficient"，必须换装 cu129 wheel：
uv pip install --python .venv-omni/bin/python \
    --index-url https://mirrors.aliyun.com/pypi/simple \
    "vllm @ https://wheels.vllm.ai/98dff2a81d747d1dba01a47f939f48c3526d4206/vllm-0.29.0%2Bcu129-cp38-abi3-manylinux_2_28_x86_64.whl"

# 5. vLLM-Omni 冒烟验证
bash scripts/omni_serve.sh        # 前台启动，健康检查约 1 分钟
bash scripts/omni_curl_test.sh    # 另开终端，产出 outputs/verify_omni_t2i.png
```

驱动版本与 CUDA 标签的对应关系、24GB 显存的边界（omni 上限 ~512px、TP=2 为何无效等）
见下文及 [AGENTS.md](./AGENTS.md)。

## 硬件与驱动（本机实测）

- 2 × RTX 3090 24GB（sm_86 Ampere），驱动 570.133.07（CUDA 12.8）
- 模型 bf16 总量约 33GB（文本编码器 Qwen3-VL 8B 17.5GB + DiT 7B 14.2GB + VAE 1.4GB），**单卡放不下**
- torch cu129 构建在该驱动上实测可用；cu13 构建不可用（vLLM PyPI 默认 wheel 是 cu13，必须换 cu129 wheel）
- 3090 无 FP8 tensor core（需 sm_89+/Hopper），vLLM-Omni 的 FP8 量化不可用

## 目录

```
deploy/
├── .venv/                  # Diffusers 基线环境 (torch 2.13.0+cu129, transformers 5.19, diffusers git)
├── .venv-omni/             # vLLM-Omni 环境 (vllm 0.29.0+cu129 + vllm-omni PR#7759 分支)
├── vllm-omni/              # vllm-omni 源码 (Qwen-Image-2.1 支持尚未合入主干)
├── models/Qwen-Image-2.1/  # 基座权重 31GB（hf-mirror 下载）
├── models/Qwen-Image-2.1-Turbo/  # Turbo 8步加速版 31GB（download_model.sh turbo）
├── outputs/                # 生成结果与 omni_server.log
└── scripts/
    ├── download_model.sh   # 下载权重（HF_ENDPOINT=hf-mirror.com，参数 base|turbo）
    ├── merge_te_shards.py  # 用基座 TE 分片本地重建 Turbo 的 TE 单文件（省 17.5GB 下载）
    ├── t2i.py              # Diffusers 文生图（--ratio 支持原生 2K 宽高比）
    ├── edit.py             # Diffusers 图像编辑（--image 可重复，最多 10 张）
    ├── run_verify.sh       # Diffusers 冒烟验证：1024x1024 / 20 steps
    ├── omni_setup.sh       # 搭建 vLLM-Omni 环境
    ├── omni_serve.sh       # 启动 omni 在线服务（单卡 + DiT 逐层 offload）
    ├── omni_curl_test.sh   # curl 冒烟：POST /v1/images/generations 512x512
    └── omni_edit.sh        # curl 编辑：POST /v1/images/edits（参考图+提示词）
```

## 验证结果（2026-10-08 实测）

| 路线 | 配置 | 结果 |
|---|---|---|
| Diffusers | 单卡 + `enable_model_cpu_offload()`，1024×1024 / 20 步 | ✅ 45s，`outputs/verify_t2i.png`（霓虹灯牌文字渲染正确） |
| Diffusers Turbo | 同上，1024×1024 / **固定 8 步** | ✅ 42s，`outputs/verify_turbo_t2i.png`（霓虹灯中英文渲染完美） |
| Diffusers Turbo 编辑 | 1280×960 / 8 步 + 参考图 | ✅ 41s（人脸一致性良好，与基座 40 步 85s 效果相当；样例含真人照片，已删除） |
| vLLM-Omni | 单卡 + DiT 逐层 offload，512×512 / 40 步 | ✅ 85s，`outputs/verify_omni_t2i.png`（茶壶质感正常） |
| vLLM-Omni | 1024×1024 | ❌ OOM：TE 常驻 17.5GB + VAE + DiT 流式缓冲后，24GB 不够 1024px 激活值 |

## Qwen-Image-2.1-Turbo（8 步加速版，2026-10-09 发布）

[Qwen/Qwen-Image-2.1-Turbo](https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo)：与基座同架构
（7B DiT + Qwen3-VL 8B TE + VAE），蒸馏为 8 步采样、CFG=1。用法与基座完全一致，
只需换 `--model` 并加 `--kv-cache`：

```bash
bash scripts/download_model.sh turbo     # 下载权重（31GB；xet 仓库必须禁 xet，脚本已内置）

CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/t2i.py \
    --model models/Qwen-Image-2.1-Turbo --kv-cache \
    --prompt "..." --output outputs/turbo.png

CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/edit.py \
    --model models/Qwen-Image-2.1-Turbo --kv-cache \
    --image input.jpg --prompt "..." --width 1280 --height 960 \
    --output outputs/turbo_edit.png
```

实测结论（RTX 3090 + CPU offload）：

- **固定 8 步**：采样表随 checkpoint 保存并自动加载，传 `num_inference_steps` 不生效（实测传 40 仍跑 8 步）。
  diffusers 必须升级到含 PR #14950 的最新 git main，否则认不出保存的采样表。
- **速度**：1024×1024 约 42s（基座 40 步约 85s，约 2× 加速）。offload 场景下单步耗时被
  权重搬运主导，8 步 vs 40 步的 5× 理论加速体现不出来；常驻 GPU（≥48GB 卡）时差距会更大。
- **`--kv-cache` 在 offload 下提速不明显**（41.8s vs 42.8s），可开可不开；vLLM-Omni 路线下
  prefix KV cache 才有意义。
- **质量**：8 步的文字渲染、编辑人脸一致性与基座 40 步相当（见上表样例）。
- **文本编码器与基座逐字节相同**：Turbo 的 TE 只是基座 4 分片的单文件重打包
  （已与官方下载流的前 1.2GB 逐字节比对验证）。若已有基座权重，可用
  `scripts/merge_te_shards.py` 本地合并出 TE 单文件，省 17.5GB 下载。
- vLLM-Omni 对 Turbo 的支持未验证（PR #7759 分支为基座适配）。

vLLM-Omni 踩坑记录（均为本机实测结论）：

1. **vllm 必须用 cu129 wheel**：PyPI 默认 vllm 0.29.0 是 cu13 构建，在驱动 570 上 flash-attn 初始化直接报
   "CUDA driver version is insufficient"。换装：
   `uv pip install "vllm @ https://wheels.vllm.ai/98dff2a81d747d1dba01a47f939f48c3526d4206/vllm-0.29.0%2Bcu129-cp38-abi3-manylinux_2_28_x86_64.whl"`
2. **TP=2 不解决问题**：文本编码器不参与 DiT 的 TP 切分，每卡仍要 17.5GB，加载即 OOM。
3. **模块级 offload（`mode: module`）在 TP 下崩溃**（DTensor 搬运报 invalid argument）；单卡可用但与 DiT 交换时的瞬态显存仍超 24GB。
4. **唯一可行配置**（`scripts/omni_serve.sh`）：单卡 + `{"mode":"layer","components":["dit"]}` 逐层流式 offload DiT，TE 常驻。代价是每步都要 H2D 流式加载 DiT 层，速度较慢。
5. omni 请求必须显式传 `num_inference_steps`（服务端默认 50≠40）和 `true_cfg_scale: 1.0`（默认 4.0 会使 DiT 计算翻倍）。

## 使用

```bash
cd deploy

# Diffusers 基线（推荐，支持到原生 2K）
bash scripts/run_verify.sh
.venv/bin/python scripts/t2i.py --ratio 16:9 --steps 40 \
    --prompt "A panoramic mountain landscape" --output outputs/out.png
.venv/bin/python scripts/edit.py --image input.png --steps 40 \
    --prompt "Change the background to a sunset beach" --output outputs/edit.png

# vLLM-Omni 在线服务（当前配置限 ~512px）
bash scripts/omni_serve.sh          # 前台启动；Ctrl-C 停止
bash scripts/omni_curl_test.sh      # 另开终端验证
```

显存充足（≥48GB 单卡或驱动升级到 580+ 后用 FP8）时 omni 的性能优势（prefix KV cache、
CUDA graph、连续批处理）才能真正发挥。prompt_rewrite/ 提示词改写模型未包含在本部署中，
如需可参照上游 `prompt_rewrite/README.md` 另行部署。

## 参考图编辑：给照片换场景/服装（如纪念照）

已用一张双人生活照验证过：保持人物五官年龄不变，换成园林/红底棚拍/教堂/海边等场景。
任何新照片按以下步骤生成即可。

**1. 准备照片**：放到本目录外任意路径（如 `../test.jpg`）。正面、五官清晰、两人无遮挡的效果最好。

**2. 选一张空闲 GPU**：`nvidia-smi` 看哪张卡空闲。若 omni 服务占着 GPU0，就用
`CUDA_VISIBLE_DEVICES=1` 跑在 GPU1 上（两个环境互不影响）。

**3. 运行编辑命令**（模板）：

```bash
cd deploy
CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/edit.py \
    --image ../你的照片.jpg \
    --prompt "中文提示词：同两位老人，<场景>，<服装>，<表情动作>，<光线>，<风格>。" \
    --width 1280 --height 960 --steps 40 --seed 42 \
    --output outputs/结果名.png
```

参数说明与经验：

- `--prompt`：用中文即可。开头写"同两位老人/同一人，保持五官与年龄特征"能显著稳住人脸一致性；
  然后按 场景→服装→表情动作→光线→风格 的顺序描述。中西文提示词都支持，不必双语都写。
- `--width/--height`：不填则沿用输入图尺寸。1280×960（4:3）是兼顾质量与显存的安全值；
  竖版棚拍可用 960×1280。单卡 24GB + CPU offload 下别超过 ~1.3M 像素。
- `--steps`：40 是质量/速度平衡点（约 90s/张）；快速预览可降到 20。
- `--seed`：同一提示词换 seed 即可"多抽几张"挑最满意的，脸部一致性不受 seed 影响。
- 多参考图：`--image a.jpg --image b.jpg`（最多 10 张），适合"两人各给一张单人照合成合影"。

**4. 查看结果**：输出在 `outputs/` 下，务必打开目检（人脸是否像、手部是否自然、
提示词元素是否齐全），不满意就改提示词或换 seed 重跑。

**批量生成**：把多条命令用 `&&` 串成一个脚本顺序跑即可（每张约 90s）。
2026-10-09 曾用同一流程验证过红底结婚照/教堂白纱/海边夕阳等 5 个场景，人脸一致性均良好
（样例含真人照片，验证后已删除，不会随仓库分发）。

## vLLM-Omni API 用法（在线服务方式）

omni 服务（`bash scripts/omni_serve.sh` 启动，端口 8091）兼容 OpenAI Images API，
适合集成到程序里批量/远程调用。24GB 单卡下输出上限约 512px，正式出图请用上面的
Diffusers 方式；omni 适合快速预览或作为服务接口验证。

**文生图**（`POST /v1/images/generations`，JSON）：

```bash
bash scripts/omni_curl_test.sh   # 内置冒烟：512x512 茶壶 → outputs/verify_omni_t2i.png
```

**参考图编辑**（`POST /v1/images/edits`，multipart/form-data）：

```bash
bash scripts/omni_edit.sh <图片> "<提示词>" [尺寸] [输出] [seed]
# 例：
bash scripts/omni_edit.sh ../test.jpg \
    "同两位老人，中式红底结婚照，女士红色旗袍，男士深色中山装，胸前小红花，囍字元素，暖色柔光棚拍，高清写实。" \
    512x384 outputs/edit_omni.png 42
```

注意事项（均为实测）：

- 请求必须带 `num_inference_steps: 40` 和 `true_cfg_scale: 1.0`（服务端默认 50 / 4.0，
  后者会让 DiT 双倍计算），脚本里已内置。
- **大图输入先缩小**（如 768×576）：服务端会把原图整张过 VAE 编码，2560×1920 原图直接 OOM。
- **编辑接口依赖 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`**（omni_serve.sh 已内置）：
  不加时分配器碎片会让编辑请求在还剩几百 MB 空闲时也报 `HTTP 500 CUDA out of memory`。
  服务长期运行后基线会涨，若编辑开始持续 OOM 且 GPU 上没有其他占用进程，重启 omni 服务即可恢复。
- 编辑接口已验证可用（样例含真人照片，验证后已删除）。
- 停止服务：`ss -ltnp | grep 8091` 拿 pid → `kill -9 -$(ps -o pgid= -p <pid> | tr -d ' ')`。
