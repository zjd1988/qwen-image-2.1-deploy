# Qwen-Image-2.1 部署验证目录

> 面向 agent/维护者的完整复现步骤与边界条件见 [AGENTS.md](./AGENTS.md)。

基于上游仓库 `../Qwen-Image-2.1` 搭建，包含两条已验证的推理路线：
**Diffusers 基线**（官方推荐）和 **vLLM-Omni 在线服务**。

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
├── models/Qwen-Image-2.1/  # 模型权重 31GB（hf-mirror 下载）
├── outputs/                # 生成结果与 omni_server.log
└── scripts/
    ├── download_model.sh   # 下载权重（HF_ENDPOINT=hf-mirror.com）
    ├── t2i.py              # Diffusers 文生图（--ratio 支持原生 2K 宽高比）
    ├── edit.py             # Diffusers 图像编辑（--image 可重复，最多 10 张）
    ├── run_verify.sh       # Diffusers 冒烟验证：1024x1024 / 20 steps
    ├── omni_setup.sh       # 搭建 vLLM-Omni 环境
    ├── omni_serve.sh       # 启动 omni 在线服务（单卡 + DiT 逐层 offload）
    └── omni_curl_test.sh   # curl 冒烟：POST /v1/images/generations 512x512
```

## 验证结果（2026-10-08 实测）

| 路线 | 配置 | 结果 |
|---|---|---|
| Diffusers | 单卡 + `enable_model_cpu_offload()`，1024×1024 / 20 步 | ✅ 45s，`outputs/verify_t2i.png`（霓虹灯牌文字渲染正确） |
| vLLM-Omni | 单卡 + DiT 逐层 offload，512×512 / 40 步 | ✅ 85s，`outputs/verify_omni_t2i.png`（茶壶质感正常） |
| vLLM-Omni | 1024×1024 | ❌ OOM：TE 常驻 17.5GB + VAE + DiT 流式缓冲后，24GB 不够 1024px 激活值 |

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
