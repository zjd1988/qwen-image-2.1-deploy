# AGENTS.md — Qwen-Image-2.1 部署目录

本目录是 Qwen-Image-2.1 的可复现部署环境。修改本目录任何内容后，请同步更新本文件。

## 目录结构与技能

- `.agents/skills/qwen-image-2-1-prompter/`：第三方提示词专家技能（上游
  [iamyoki/qwen-image-2.1-skill](https://github.com/iamyoki/qwen-image-2.1-skill)，Apache-2.0），
  agent 在本目录工作时自动加载，用于把口语化需求重写成 Qwen-Image-2.1 官方规范的提示词。
  同步上游更新时整体替换该目录即可

## 环境基线（改动前必须核对）

- GPU: 2 × RTX 3090 24GB（sm_86）；驱动 570.133.07，CUDA 12.8
- 包管理用 uv（二进制在 `/data/zhaojd-a/github_codes/uv`），不用 conda/pip
- 网络：huggingface.co 不通，用 `HF_ENDPOINT=https://hf-mirror.com`；PyPI 用阿里云镜像
  `https://mirrors.aliyun.com/pypi/simple`（清华 tuna 在本机大 wheel 上只有 ~200KB/s，勿用）；
  github.com 与 download.pytorch.org 直连可用
- CUDA 版本约束：cu129 构建可用；**cu13 构建不可用**（驱动不足 580）；PyPI 上 vllm/torch 默认最新版可能是 cu13，必须显式选 cu129 版本

## 两个互不复用的 Python 环境

| 环境 | 用途 | 关键版本 |
|---|---|---|
| `.venv/` | Diffusers 基线（t2i.py / edit.py） | torch 2.13.0+cu129、torchvision 0.28.0+cu129、transformers 5.19.0、diffusers 0.41.0.dev0（git main） |
| `.venv-omni/` | vLLM-Omni 在线服务 | vllm **0.29.0+cu129**（非 PyPI 默认 cu13 wheel）、vllm-omni PR#7759 分支（`vllm-omni/` 目录 editable 安装） |

注意：

- torchvision 必须与 torch 同源同 CUDA 标签（都是 +cu129），混装 PyPI 版会报 `operator torchvision::nms does not exist`
- diffusers 必须装 git main（`git+https://github.com/huggingface/diffusers`），发布版没有 `QwenImage21Pipeline`
- vllm-omni 对 Qwen-Image-2.1 的支持未合入主干，在 PR #7759 分支：
  `git fetch origin pull/7759/head:qwen-image-2.1 && git checkout qwen-image-2.1`

## 完整复现步骤

前置：约 45GB 磁盘、模型下载约 1.5h（hf-mirror 6MB/s）、uv 可用。

```bash
cd /data/zhaojd-a/github_codes/qwen-image-2.1-deploy/deploy
UV=/data/zhaojd-a/github_codes/uv
export UV_HTTP_TIMEOUT=300

# 1. 模型权重（31GB，含 text_encoder/transformer/vae/scheduler/processor）
bash scripts/download_model.sh

# 2. Diffusers 基线环境
$UV venv --python 3.12 .venv
# torch 走 pytorch cu129 索引（pytorch 索引的 nvidia 依赖会 301 到 pypi.nvidia.cn，
# 极慢；如有现成的 cu129 环境可直接 cp -al 硬链接 site-packages 里的
# torch/torch-*/torchgen/nvidia*/cuda_toolkit* 目录，秒级完成）
$UV pip install --python .venv/bin/python \
    --index-url https://download.pytorch.org/whl/cu129 torch==2.13.0 torchvision==0.28.0
$UV pip install --python .venv/bin/python \
    --index-url https://mirrors.aliyun.com/pypi/simple \
    'transformers>=5.17' accelerate pillow safetensors sentencepiece
$UV pip install --python .venv/bin/python 'git+https://github.com/huggingface/diffusers'

# 3. Diffusers 验证（单卡 24GB，CPU offload，约 1 分钟出图）
bash scripts/run_verify.sh   # 产出 outputs/verify_t2i.png

# 4. vLLM-Omni 环境
bash scripts/omni_setup.sh   # clone PR#7759 分支 + vllm + editable 安装
# 关键修正：PyPI 的 vllm 0.29.0 是 cu13 构建，必须换装 cu129 wheel，否则
# flash-attn 初始化报 "CUDA driver version is insufficient for CUDA runtime version"
$UV pip install --python .venv-omni/bin/python \
    --index-url https://mirrors.aliyun.com/pypi/simple \
    "vllm @ https://wheels.vllm.ai/98dff2a81d747d1dba01a47f939f48c3526d4206/vllm-0.29.0%2Bcu129-cp38-abi3-manylinux_2_28_x86_64.whl"

# 5. vLLM-Omni 服务验证
bash scripts/omni_serve.sh        # 前台；健康检查通过后（约 1-2 分钟）
bash scripts/omni_curl_test.sh    # 另开终端；产出 outputs/verify_omni_t2i.png
```

## vLLM-Omni 在 24GB 卡上的已知边界（实测，勿再踩）

- bf16 全模型 ~33GB 超单卡；TP=2 **无效**（文本编码器不切分，每卡仍需 17.5GB）
- `mode: module` offload 在 TP 下崩溃（DTensor 搬运 `CUDA error: invalid argument`）；单卡下 TE/DiT 双向交换的瞬态也超 24GB
- 唯一可行：`--diffusion-offload-config '{"mode":"layer","components":["dit"],"pin_memory":false}'`（DiT 逐层流式，TE 常驻），**上限约 512×512**，1024px 激活值 OOM
- `text_encoder` 不支持 layer 模式（模型未声明 streamable plan）
- 请求必须带 `num_inference_steps: 40`（服务端默认 50）和 `true_cfg_scale: 1.0`（默认 4.0 会让 DiT 双倍计算）
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`：**编辑接口必须加**（已写进 omni_serve.sh）。不加时分配器碎片会让输入图 VAE 编码（~400MB）在还剩几百 MB 空闲的情况下也 500 OOM；T2I 加不加都能跑
- offload 组件只有 `dit` 和 `text_encoder` 两个可选（`vae` 不是合法组件，填了启动直接 ValueError）
- `/v1/images/edits` 参考图编辑可用（multipart 上传，输出上限同 ~512px），输入图会先整张过 VAE 编码：
  大图必须先在客户端缩小（如 768×576）。GPU0 上若有其他进程占显存，编辑接口会先顶不住
  （T2I 耐受性更好），清出余量或重启服务可恢复
- 停止服务必须杀**监听进程所属整个进程组**（worker 是 spawn 的 `python -c`，按名字 pkill 杀不到）：
  `ss -ltnp | grep 8091` 拿 pid → `kill -9 -$(ps -o pgid= -p <pid> | tr -d ' ')`

## Turbo（Qwen-Image-2.1-Turbo，8 步加速版）

- 下载：`bash scripts/download_model.sh turbo`。**必须 `HF_HUB_DISABLE_XET=1`**（脚本已内置）——
  Turbo 是 xet 存储仓库，hf_transfer/hf_xet 会绕开 hf-mirror 直连 `cas-*.xethub.hf.co` 报 401/403；
  且镜像签发的 cas-bridge URL 有效期 1 小时，17.5GB 单文件在 ~3MB/s 下必然过期
- **TE 与基座逐字节相同**（单文件重打包）：已有基座权重时用 `scripts/merge_te_shards.py`
  本地合并 4 分片重建（~40s，流式拷贝不占内存），省 17.5GB 下载；产物用法见脚本 docstring
- diffusers 必须含 PR #14950（pipeline 加载 checkpoint 保存的采样表），用最新 git main
- 固定 8 步：传 `num_inference_steps` 不生效（实测传 40 仍跑 8 步）
- 实测（3090 + CPU offload）：1024×1024 T2I 42s、1280×960 编辑 41s，约为基座 40 步的 2×；
  加速比受 offload 搬运限制，常驻 GPU 时收益更大。`use_kv_cache` 在 offload 下无明显提速
- 样例：`outputs/verify_turbo_t2i.png`（编辑样例含真人照片，验证后已删除）
- cache-dit（DBCache 加速）：**不可用**（2026-10-12 核实 PyPI 1.5.2 与 git main 均只适配
  Qwen-Image 1.0；2.1 block 签名完全不同，1.0 patch functor 不兼容，非一行补丁可解。
  基线备查：基座 40 步 1024×1024 = 76.1s，`scripts/bench_cache_dit.py`）

## 验证标准

改动后必须通过的检查：

1. `bash scripts/run_verify.sh` 出图且图像内容符合提示词（`outputs/verify_t2i.png`）
2. omni 服务 `/health` 200 + `bash scripts/omni_curl_test.sh` 出图（`outputs/verify_omni_t2i.png`）
3. 两张图都要打开目检，不能只看退出码
