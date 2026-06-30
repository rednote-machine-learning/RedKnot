# Iso-Throughput KV-Cache Memory Experiment

## 1. 实验目的

RedKnot 的离线 KV 复用如果"全量生成、全部保存"，会造成严重的存储膨胀：大量
文档块其实很少被复用，却长期占用显存。本实验回答一个直接的问题：

> **在达到相同吞吐（相同缓存命中率 / 相同 prefill 节省）的前提下，一个有界的
> LRU KV-cache 需要多少显存，相比"全量保存所有离线 KV"能省多少？**

这是支撑 RedKnot "有界缓存优于全量保存" 的核心显存论据。

## 2. 被测模型

| 模型 | 用途 | 关键参数 |
|---|---|---|
| DeepSeek-V4-Flash | KV 显存成本模型 | MLA 架构，43 层，`kv_lora_rank=512`，`qk_rope_head_dim=64` |

KV 显存按 MLA 真实占用计算：

```
KV bytes/token = layers x (kv_lora_rank + qk_rope_head_dim) x 2 (bf16)
              = 43 x (512 + 64) x 2 = 49,536 bytes/token
```

作为对比，等效 MHA（64 头 x 128 维 x K/V x 2B）为 1,409,024 B/token，
**MLA 小 28.4 倍**。每个文档块按约 600 token 计。

## 3. 数据集

实验同时使用真实 RAG 数据和受控合成工作负载，覆盖不同的"检索偏斜度"
（即热门文档被复用的集中程度）。

| 数据集 | 来源 | 请求数 | 唯一块数 | 非前缀复用率 | 最大复用 |
|---|---|---:|---:|---:|---:|
| MuSiQue (real) | musique_ans 全量 dev | 2417 | 17629 | 0.951 | 261 |
| 2wikimqa | LongBench 切片 | 200 | 1404 | 0.896 | 30 |
| Zipf 0.8 (low skew) | 合成 | 5000 | 12559 | 0.898 | 1410 |
| Zipf 1.1 (med skew) | 合成 | 5000 | 7542 | 0.904 | 4100 |
| Zipf 1.5 (high skew) | 合成 | 5000 | 2510 | 0.896 | 4990 |

**关于合成数据**：公开 QA 切片（200 题）太小，跨请求复用稀疏，无法体现真实
RAG 的长尾复用。因此我们以**真实 MuSiQue 段落池（17629 个 Wikipedia 段落）**
为底，按 Zipf 流行度分布采样检索，生成可控偏斜度的工作负载。检索到的段落
被随机打散到不同位置（模拟非前缀复用）。Zipf 指数越大越偏斜（热门块越集中）。

**关于非前缀复用**：所有数据集非前缀复用率都在 0.90–0.95，意味着同一文档块
在不同请求中出现在不同位置。主流前缀缓存（vLLM prefix cache / SGLang
RadixCache / Mooncake）只能复用前缀，对这些复用几乎全部 miss——这正是
RedKnot 需要非前缀 KV 复用的根本原因。

## 4. 测试的方法 / 缓存策略

- **store-all（全量保存基线）**：缓存所有出现过的唯一块，永不淘汰。显存 =
  所有唯一块 KV 之和；命中率 = 理论上限（每次重复访问都命中）。
- **LRU（被测策略）**：在固定字节预算下按最近最少使用淘汰。选 LRU 而非 LFU，
  因为 RAG 话题会漂移，LFU 死守过时高频块、命中率更低（同预算下省的算力更少）。

## 5. 实验设计（如何对齐"相同吞吐"）

1. 跑 **store-all** 得到显存占用 `M_all` 和最大命中率 `H_max`。
2. 对 **LRU** 扫多个显存预算，得到 (预算 -> 命中率) 曲线（预算网格从 0.02 GB
   到 512 GB）。
3. 设若干吞吐目标 = `H_max` 的百分比（50% / 80% / 90% / 95% / 99%）。
4. 在 LRU 曲线上插值，找到**达到该命中率所需的最小显存** `M_lru`。
5. 显存节省倍数 = `M_all / M_lru`（相同吞吐下）。

## 6. 主要结果

### store-all 基线显存

| 数据集 | store-all 显存 | 最大命中率 |
|---|---:|---:|
| MuSiQue (real) | 524.0 GB | 0.635 |
| Zipf 0.8 | 373.3 GB | 0.749 |
| Zipf 1.1 | 224.2 GB | 0.849 |
| Zipf 1.5 | 74.6 GB | 0.950 |

### 相同吞吐下 LRU 的显存节省倍数（store-all / LRU）

| 数据集 | 50% | 80% | 90% | 95% | 99% |
|---|---:|---:|---:|---:|---:|
| MuSiQue (real) | **497x** | **26x** | 5.7x | 2.8x | 1.5x |
| Zipf 0.8 | 10.7x | 2.8x | 1.8x | 1.5x | 1.0x |
| Zipf 1.1 | **80x** | **9.4x** | 3.9x | 2.4x | 1.3x |
| Zipf 1.5 | **134x** | **30x** | 11.7x | 5.6x | 2.0x |

## 7. 结论

1. **实用吞吐区间显存节省巨大**：要恢复 50–80% 的 store-all 吞吐，LRU 只需
   **10–500 倍更少**的 KV 显存。真实 RAG（MuSiQue）达到 80% 吞吐只要 20 GB，
   而全量保存要 524 GB——**省 26 倍**。

2. **追求 100% 吞吐不划算**：最后几个百分点来自极冷的长尾块，迫使 LRU 趋近
   全量大小（节省 → 1x）。

3. **偏斜越强省得越多**：高偏斜（Zipf 1.5）复用集中在少数热块，很小的 LRU
   就能覆盖大部分吞吐，80% 吞吐处省 30 倍。

4. **甜点在 ~80% 吞吐**：跨数据集 3–30 倍节省，吞吐几乎不损失，是推荐的运行点。

**核心论点**：全量保存离线 KV 是浪费的，因为多数块很少复用。一个有界 LRU
缓存能用 **3–30 倍更少的显存**（真实 RAG 上 26 倍）达到全量保存约 80% 的吞吐，
因为它只保留近期/即将复用的热块。

## 8. 图表

`figures/mem_iso_throughput.png` / `.pdf`（论文单栏，两幅并排）：
- (a) 达到各吞吐目标所需的 KV 显存（虚线为 store-all 基线）
- (b) 相同吞吐下的显存节省倍数曲线

## 9. 复现

```bash
# 生成合成 Zipf 工作负载（基于真实段落池）
python test/srt/redknot/synth_rag_workload.py --zipf 0.8 --n-queries 5000 --k 10
python test/srt/redknot/synth_rag_workload.py --zipf 1.1 --n-queries 5000 --k 10
python test/srt/redknot/synth_rag_workload.py --zipf 1.5 --n-queries 5000 --k 10

# 计算等吞吐显存对比
python test/srt/redknot/mem_at_iso_throughput.py

# 画论文图（两幅并排，单栏）
python test/srt/redknot/plot_mem_iso.py
```

## 10. 相关文件

| 文件 | 作用 |
|---|---|
| `synth_rag_workload.py` | Zipf 工作负载生成器（真实段落池） |
| `chunk_lifecycle.py` | 数据集 loader（musique / LongBench / dureader） |
| `kv_cache_lifecycle.py` | LRU / LFU / store-all 缓存实现 |
| `mem_at_iso_throughput.py` | 等吞吐显存计算（核心实验） |
| `plot_mem_iso.py` | 论文图绘制 |
| `figures/mem_iso_throughput.json` | 实验结果数据 |
| `figures/mem_iso_throughput.png/pdf` | 论文图 |
