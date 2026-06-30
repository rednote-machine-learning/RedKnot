<div align="center" id="redknottop">
<img src="RedKnot_Logo.png" alt="RedKnot logo" width="600" margin="10px"></img>

**Head-Classified KV Reuse + Elastic Sparsity for Long-Context LLM Inference**

[![SGLang](https://img.shields.io/badge/built%20on-SGLang-blue)](https://github.com/sgl-project/sglang)
[![license](https://img.shields.io/badge/license-Apache--2.0-green)](./LICENSE)
[![models](https://img.shields.io/badge/models-Qwen3%20%7C%20Mistral%20%7C%20Llama3.3%20%7C%20Qwen3.5--MoE%20%7C%20DeepSeek--V4-orange)]()

</div>

--------------------------------------------------------------------------------

## 简介 / About

**RedKnot** 是构建在 [SGLang](https://github.com/sgl-project/sglang) 之上的长上下文推理加速集成。它的核心思想是：**不是所有 attention head 都需要全量 KV，也不是所有 token 都需要走全量 FFN**。RedKnot 通过

- **Head 分类（Head Classification）**：把每个 `(layer, kv_head)` 划分为 `global / local / retrieval / dense` 四类，按类别决定 KV 的存储与复用策略；
- **离线 KV 复用 + RoPE 重定位（Offline KV Reuse）**：对可复用的 segment 离线存储 KV，在线只对必要的 token 做选择性重算，并通过 RoPE 重定位保证数值对齐；
- **Elastic Sparsity / Sparse FFN**：基于 attention 重要性做 token-selective FFN，跳过低贡献 token 的前馈计算；
- **SegPagedAttention 运行时**：per-head page table + 分段 KV store，使不同 head 类别可以有不同的可见窗口；
-  **DeepSeek-V4** 和 **Qwen 3.5 全系列** 将在下个版本全面更新和适配，当前仅开源基础版本。
在 **接近无损精度**（部分场景甚至优于 dense baseline）的前提下，把长上下文 prefill 的 **FLOPs 降低约 50%–70%**、**TTFT 提速 1.35x–2.2x**（随上下文长度增大收益越大）。
- **论文 / Paper**：**RedKnot: Efficient Long-Context LLM Serving with Head-Aware KV Reuse and SegPagedAttention** — Yang Liu, ZhaoKai Luo, HuaYi Jin, ZhiYong Wang, RuoZhou He, BoYu Wang, Guanjie Chen, Junhao Hu（<https://arxiv.org/abs/2606.06256>）
> 本仓库基于 SGLang 构建，保留了 SGLang 全部的高性能 serving 能力（RadixAttention、零开销调度器、PD 分离、连续批处理、量化等），RedKnot 作为 attention 层的扩展集成在 `python/sglang/srt/layers/attention/redknot/`。

## 核心机制 / Key Ideas

| 机制 | 说明 | 代码位置 |
|---|---|---|
| Head 分类配置 | `global / local / retrieval / dense` 四类策略 + JSON 加载 | `redknot/head_config.py`, `head_profiler.py` |
| 离线 KV cache + RoPE 重定位 | segment 级离线 KV 存储与复位（数值对齐） | `redknot/offline_cache.py`, `rope_helper.py` |
| Head-aware attention 恢复 | FlashAttention-2 / FA-3 分桶注意力 | `redknot/ops_flash.py`, `ops_flash3.py` |
| Sparse FFN（Elastic Sparsity） | token-selective FFN，按重要性跳算 | `redknot/sparse_ffn.py` |
| SegPagedAttention 运行时 | per-head page table + 分段 KV store | `redknot/segpaged.py`, `segpaged_v2/` |
| DeepSeek-V4 MLA 集成 | 复用 indexer top-k 做选择性重算 | `redknot/deepseek_v4_mla.py`, `dsv4_offline_reuse.py` |
| PD KV 传输 / head-aware 调度 | head-class KV 分片传输与容量模型 | `redknot/pd_transfer.py`, `scheduler.py` |

更详细的阶段规划见 `python/sglang/srt/layers/attention/redknot/ROADMAP.md`。

## 快速开始 / Getting Started

RedKnot 复用 SGLang 的安装方式：

```bash
# 安装（开发模式）
pip install -e "python[all]"
```

部分模型（`Qwen3.5-*` 的 `qwen3_5_moe` 架构、`DeepSeek-V4`）需要 transformers 5.x，
仓库自带 `.venv_tf5`（transformers 5.12.0）。系统 transformers 4.57 无法加载这些模型。

- 安装：<https://docs.sglang.io/get_started/install.html>
- 快速上手：<https://docs.sglang.io/basic_usage/send_request.html>

## RAG 精度 / 速度基准 / Benchmarks

所有 benchmark 脚本位于 `test/srt/redknot/`，与之配套的其他脚本、绘图与文档已归档到 `test/srt/redknot/utils/`。
每个 benchmark 都用 **诚实的 dense baseline**（一次完整 FlashAttention-2 prefill）对比 RedKnot 的 head-class KV 复用路径，报告精度（SQuAD F1 / EM）、速度（TTFT、speedup、decode tok/s）与后台开销。

> 硬件：NVIDIA L20Y ×8（每张 80GB）｜每模型 4 样本｜日期：2026-06-26

### Qwen3-32B — HotpotQA ✅ 最佳

| 上下文 | base F1 | RedKnot F1 | base TTFT | RedKnot TTFT | speedup | FLOPs 节省 |
|---|---|---|---|---|---|---|
| 16K | 0.750 | **1.000** | 3.24s | 2.33s | **1.39x** | 69.2% |
| 24K | 1.000 | **1.000** | 5.25s | 2.96s | **1.77x** | 70.9% |
| 32K | 0.750 | **1.000** | 7.74s | 4.02s | **1.93x** | 72.2% |

RedKnot F1 **始终 ≥ baseline**（无损甚至更好），TTFT speedup 随上下文增长（1.39→1.93x），FLOPs 节省稳定 ~70%。

### Qwen3.5-35B-A3B (MoE) — LongBench

| 上下文 | 数据集 | std F1 | RedKnot F1 | compute 节省 | TTFT speedup |
|---|---|---|---|---|---|
| 16K | triviaqa | 1.000 | 1.000 | 46.4% | **1.87x** |
| 32K | multifieldqa_en | 0.792 | 0.576 | 50.4% | **2.02x** |
| 64K | triviaqa | 0.875 | 0.750 | 53.8% | **2.16x** |

TTFT speedup 随上下文增长（1.87→2.16x）；16K 无损，长上下文有一定下降（线性注意力 + MoE 稀疏的代价）。

### Mistral-7B-Instruct-v0.3 — HotpotQA

| 上下文 | base F1 | RedKnot F1 | base TTFT | RedKnot TTFT | speedup | FLOPs 节省 |
|---|---|---|---|---|---|---|
| 16K | 0.250 | 0.475 | 0.70s | 0.52s | **1.35x** | 51.5% |
| 24K | 0.250 | 0.250 | 1.12s | 0.80s | **1.39x** | 50.2% |
| 32K | 0.688 | 0.100 | 1.67s | 1.24s | **1.35x** | 49.1% |

7B 小模型在长上下文 RAG 上基线本身能力弱，F1 噪声大；系统指标（TTFT speedup ~1.35x / FLOPs 节省 ~50%）稳定。

### 已知问题 / Known Issues

- **Llama-3.3-70B-Instruct**：baseline 正常，但 RedKnot 解码路径在 LongBench 长上下文下输出退化（重复 token），且单卡 INT4 易 OOM、多卡 bf16 触发 cross-device 错误。属 RedKnot 在 Llama3.3 上的既有算法/配置问题，待单独排查 `driver_batched` 的 Llama 兼容性与 `head_class/llama-70B_*.json` 配置质量。

## 运行方式 / How to Run

基准脚本统一放在 `test/srt/redknot/`，运行时依赖同目录下的 `head_class/`、`sparse_ffn_params/`、`datasets/` 配置与数据目录，以及 `utils/`（如 `fp8_offline_patch.py`）。

### 一键复现

```bash
cd test/srt/redknot

# 默认中小模型（HotpotQA 的 Mistral/Qwen3 + LongBench 的 Llama/Qwen35）
bash run_all_rag.sh

# 自定义模型与规模
RK_MODELS="mistral qwen3" RK_SAMPLES=4 RK_LENGTHS=16K,24K,32K \
  bash run_all_rag.sh
```

### 一键运行全部 RAG benchmark

依次跑完五个模型的 RAG 基准（Qwen3.5-MoE / Qwen3 / Mistral / Llama3.3 / DeepSeek-V4）：

```bash
cd test/srt/redknot

python benchmark_RedKnot_Qwen35_RAG.py
python benchmark_RedKnot_Qwen3_RAG.py
python benchmark_RedKnot_Mistral_RAG.py
python benchmark_RedKnot_Llama3.3_RAG.py
python benchmark_RedKnot_DeepSeekV4_RAG.py
```

> 注：`Qwen3.5-MoE` 与 `DeepSeek-V4` 需 transformers 5.x，请改用 `.venv_tf5/bin/python` 运行对应脚本。

### 单模型最小验证（几分钟出结果）

```bash
# Qwen3-32B（INT4 NF4，单卡）
REDKNOT_N_SAMPLES=1 REDKNOT_LENGTHS=16K REDKNOT_MAX_NEW=8 \
  CUDA_VISIBLE_DEVICES=0 python test/srt/redknot/benchmark_RedKnot_Qwen3_RAG.py

# Qwen3.5-35B-A3B / Qwen3.5-397B-A17B（MoE，需 transformers 5，使用 .venv_tf5）
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1 \
REDKNOT_N_SAMPLES=1 REDKNOT_MAX_NEW=8 CUDA_VISIBLE_DEVICES=0,1 \
  .venv_tf5/bin/python test/srt/redknot/benchmark_RedKnot_Qwen35_397B_RAG.py

# Mistral-7B-Instruct-v0.3（bf16，单卡）
REDKNOT_N_SAMPLES=4 CUDA_VISIBLE_DEVICES=0 \
  python test/srt/redknot/benchmark_RedKnot_Mistral_RAG.py

# Llama-3.3-70B-Instruct（INT4 NF4，单卡）
REDKNOT_N_SAMPLES=3 CUDA_VISIBLE_DEVICES=0 \
  python test/srt/redknot/benchmark_RedKnot_Llama3.3_RAG.py

# DeepSeek-V4（MLA + indexer，需大显存 / .venv_tf5）
CUDA_VISIBLE_DEVICES=0 \
  .venv_tf5/bin/python test/srt/redknot/benchmark_RedKnot_DeepSeekV4_RAG.py
```

### 关键环境变量

| 变量 | 说明 |
|---|---|
| `REDKNOT_N_SAMPLES` | 评测样本数 |
| `REDKNOT_LENGTHS` | 上下文长度（HotpotQA 模型，如 `16K,24K,32K`） |
| `REDKNOT_DATASETS` | LongBench 数据集（LongBench 模型） |
| `REDKNOT_MAX_NEW` | 最大生成 token 数 |
| `REDKNOT_DTYPE` | `int4` 或 `bf16` |
| `REDKNOT_COMPILE` | 是否开启 `torch.compile`（`0`/`1`） |
| `CUDA_VISIBLE_DEVICES` | 可见 GPU |

全部运行日志保存在 `test/srt/redknot/rag_logs/`。
## 致谢 / Acknowledgment

RedKnot 构建于 [SGLang](https://github.com/sgl-project/sglang) 之上，并复用了其生态中诸多项目的设计与实现：
- [SGLang](https://github.com/sgl-project/sglang)
- [vLLM](https://github.com/vllm-project/vllm)

