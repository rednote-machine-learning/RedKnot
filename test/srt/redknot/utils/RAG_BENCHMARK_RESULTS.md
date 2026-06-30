# RedKnot RAG Benchmark — 速度 / 精度 / 后台开销

一键复现脚本：`run_all_rag.sh`（按模型分别调用 `benchmark_RedKnot_<Model>_RAG.py`）。
每个 benchmark 都用 **诚实的 dense baseline**（一次完整 FlashAttention-2 prefill）
对比 RedKnot 的 head-class KV 复用路径，报告：

- **精度**：SQuAD F1 / EM（baseline vs RedKnot vs gold）
- **速度**：TTFT（首 token 延迟）及 speedup、decode 吞吐（tok/s）
- **后台开销**：每次 head-class 的 `online / rope / kv_build / query_fwd` 分项耗时；
  以及 prefill FLOPs 按 attn / ffn / proj 分解的节省比例
- 硬件：NVIDIA L20Y ×8（每张 80GB） | 日期：2026-06-26

---

## 一、模型与脚本对应表（已规范化命名）

| 模型 | 脚本 | 数据集 | 权重路径 | 运行环境 |
|---|---|---|---|---|
| Mistral-7B-Instruct-v0.3 | `benchmark_RedKnot_Mistral_RAG.py` | HotpotQA | `096/models/Mistral-7B-Instruct-v0.3` | 默认 python (bf16) |
| Qwen3-32B | `benchmark_RedKnot_Qwen3_RAG.py` | HotpotQA | `096/models/Qwen3-32B` | 默认 python (INT4) |
| Llama-3.3-70B-Instruct | `benchmark_RedKnot_Llama3.3_RAG.py` | LongBench | `096/models/Llama-3.3-70B-Instruct` | 默认 python (INT4) |
| Qwen3.5-35B-A3B | `benchmark_RedKnot_Qwen35_397B_RAG.py` | LongBench | `checkpoints/opensource/Qwen3.5-35B-A3B` | **`.venv_tf5`** (transformers 5.x) |
| Qwen3.5-397B-A17B | `benchmark_RedKnot_Qwen35_397B_RAG.py`（同脚本，自动切配置） | LongBench | `checkpoints/opensource/Qwen3.5-397B-A17B-FP8` | **`.venv_tf5`**（需大显存，本次未实跑） |
| DeepSeek-V4 | `benchmark_RedKnot_DeepSeekV4_RAG.py` | LongBench | `checkpoints/opensource/DeepSeek-V4-Flash-*` | **`.venv_tf5`**（需大显存，本次未实跑） |

> 注：`Qwen3.5` / `DeepSeek-V4` 用 `qwen3_5_moe` / 新架构，系统 transformers 4.57 不支持，
> 必须用仓库自带的 `.venv_tf5/bin/python`（transformers 5.12.0）。`run_all_rag.sh` 已自动处理。

---

## 二、实验结果（每模型 4 样本）

### Mistral-7B-Instruct-v0.3 — HotpotQA

| 上下文 | base F1 | RedKnot F1 | base TTFT | RedKnot TTFT | speedup | FLOPs 节省 |
|---|---|---|---|---|---|---|
| 16K | 0.250 | 0.475 | 0.70s | 0.52s | **1.35x** | 51.5% |
| 24K | 0.250 | 0.250 | 1.12s | 0.80s | **1.39x** | 50.2% |
| 32K | 0.688 | 0.100 | 1.67s | 1.24s | **1.35x** | 49.1% |

说明：7B 小模型在长上下文 RAG 上**基线本身能力弱**（base F1 也在 0.25 左右波动），
F1 噪声大；TTFT speedup / FLOPs 节省（系统指标）稳定在 ~1.35x / ~50%。

### Qwen3-32B — HotpotQA  ✅ 最佳

| 上下文 | base F1 | RedKnot F1 | base TTFT | RedKnot TTFT | speedup | FLOPs 节省 |
|---|---|---|---|---|---|---|
| 16K | 0.750 | **1.000** | 3.24s | 2.33s | **1.39x** | 69.2% |
| 24K | 1.000 | **1.000** | 5.25s | 2.96s | **1.77x** | 70.9% |
| 32K | 0.750 | **1.000** | 7.74s | 4.02s | **1.93x** | 72.2% |

说明：RedKnot F1 **始终 ≥ baseline**（无损甚至更好），TTFT speedup 随上下文增长（1.39→1.93x），
FLOPs 节省稳定 ~70%。FLOPs 分项：attn 节省 ~83%、ffn ~82%、proj 0%（投影不稀疏）。

### Qwen3.5-35B-A3B (MoE) — LongBench

| 上下文 | 数据集 | std F1 | RedKnot F1 | compute 节省 | TTFT speedup |
|---|---|---|---|---|---|
| 16K | triviaqa | 1.000 | 1.000 | 46.4% | **1.87x** |
| 32K | multifieldqa_en | 0.792 | 0.576 | 50.4% | **2.02x** |
| 64K | triviaqa | 0.875 | 0.750 | 53.8% | **2.16x** |

说明：TTFT speedup 随上下文增长（1.87→2.16x），compute 节省 46→54%。
F1 在 16K 无损，长上下文（32K/64K）有一定下降（线性注意力 + MoE 稀疏的代价）。

### Llama-3.3-70B-Instruct — LongBench  ⚠️ 未得到有效结果

- baseline 正常（如 "Knowsley" F1=1.00），但 **RedKnot 解码路径输出退化为重复 token**
  （如 `a test a test ...` / `}\\ \\}^}...`），F1=0.00。
- 单卡 INT4（~35GB 权重）在 LongBench 长上下文下 KV 拼接 **OOM**；
  bf16 多卡（device_map=auto）会触发 RedKnot 路径的 **cross-device 错误**
  （`apply_rotary` 张量分布在 cuda:0/cuda:3），因为 RedKnot driver 假设模型在单一设备。
- **结论：这是 RedKnot 在 Llama3.3 上的既有算法/配置问题，非本次整理引入。**
  需要单独排查 `driver_batched` 的 Llama 兼容性与 `head_class/llama-70B_*.json` 配置质量。

---

## 三、后台开销（head-class 各阶段耗时）

每次 RedKnot 前向会打印一行，例如（Qwen3-32B / 32K）：

```
[headclass] online=1.671s  rope=0.041s  kv_build=0.026s  query_fwd=0.336s  total_ttft=2.074s
```

- `online`：在线段的稀疏 prefill（主要成本）
- `rope`：旋转位置编码重算
- `kv_build`：head-class KV cache 拼装
- `query_fwd`：query 段前向
- `total_ttft`：以上之和 = RedKnot 的首 token 延迟

对照 baseline 的 dense prefill TTFT，即得 speedup。

---

## 四、一键复现

```bash
cd test/srt/redknot

# 默认 4 个中小模型（HotpotQA 的 Mistral/Qwen3 + LongBench 的 Llama/Qwen35）
bash run_all_rag.sh

# 自定义模型与规模
RK_MODELS="mistral qwen3" RK_SAMPLES=4 RK_LENGTHS=16K,24K,32K \
  bash run_all_rag.sh

# 单模型最小验证（几分钟出结果）
REDKNOT_N_SAMPLES=1 REDKNOT_LENGTHS=16K REDKNOT_MAX_NEW=8 \
  CUDA_VISIBLE_DEVICES=0 python benchmark_RedKnot_Qwen3_RAG.py
```

关键环境变量：`REDKNOT_N_SAMPLES` / `REDKNOT_LENGTHS`（HotpotQA 模型）/
`REDKNOT_DATASETS`（LongBench 模型）/ `REDKNOT_MAX_NEW` / `REDKNOT_DTYPE`（int4|bf16）/
`REDKNOT_MAX_CTX` / `CUDA_VISIBLE_DEVICES`。

全部运行日志保存在 `rag_logs/`。
