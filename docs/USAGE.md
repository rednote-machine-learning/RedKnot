# 安装、请求协议与指标说明

将 RedKnot 的非前缀 chunk、local/global 头拆分与在线聚合接入 vLLM V1。
这是独立、显式启用的研究版插件，**不修改现有 SGLang RedKnot 或 vLLM 源码**。
不是普通前缀 KV Connector，也不是未经验证的全模型加速承诺。

## 当前支持边界

| 能力 | 当前状态 |
| --- | --- |
| Qwen2 / Qwen3 / Llama 静态 RoPE、MHA/GQA | 原生 attention backend 与 runner 接入已实现，GPU 端到端待验证 |
| 非前缀 chunk | 按真实 token 内容与模型/策略身份命中，支持 chunk 移位 |
| local/global 分头 | local 清洁行复用离线注意力输出；global 与边界/查询行在线计算 |
| MHA/GQA decode 一致性 | 把复用 local K/V 写入真实物理 KV 页面，后续使用原生 decode |
| KV Manager | CPU 字节预算、LRU、整请求租约、使用中禁止淘汰、整 chunk 原子提交 |
| DeepSeek V4 Flash-0731 MLA | 专用 native FlashMLA/runner 接入、选头 attention 与 `z_off` 聚合已实现；CPU 与 sparse GPU 微测试通过，**完整模型生成待验证** |
| Sparse FFN/MoE、SegPaged、并发调度 | 未移植；保持原生 FFN/MoE 与 vLLM 页面分配 |

首版固定 TP/PP/DP=1、V1 runner、eager、单请求完整 prefill；关闭 APC、
chunked prefill、KV Connector、推测解码与 LoRA。默认 `engine_family: mha`
只接受未量化 FP16/BF16 权重及匹配的 KV dtype，并拒绝动态/缩放 RoPE、
MRoPE、滑窗和 Qwen3.5 混合状态模型。Flash 必须显式选择
`engine_family: deepseek_v4_flash`，保留它原生的 YaRN、SWA/C4/C128、
FP4/FP8 权重及 FP8 latent KV；不能用 MHA 配置静默代替。

本版仍计算完整 QKV 投影和 FFN，仅省去清洁 local query 行的注意力计算。
CPU 到 GPU 拷贝、分组 kernel 启动和 eager 开销可能抵消收益；必须实测，
**不能据缓存比例声称全模型节约 70% 或 TTFT 加速 2–5 倍**。

## 固定源码版本

- RedKnot：`55ee4e8401603f8d2612877e4053e18b37b1c1bd`
- vLLM：`e52be1a62d3879b1202f4f355d3c3472b560c6f2`
- Python：3.12 或更高版本。

插件启用前按 family 校验关键源码 SHA-256：MHA 十一个文件，DSV4 十五个
原生模型/metadata/projection/FlashMLA 接口文件。接口漂移直接拒绝启动。
不要随意更新散列来绕过检查；升级需要重新适配和测试。

## 安装与 CPU 检查

在**单独准备、已安装上述 vLLM 的环境**中安装本项目，不会自动安装或升级
PyTorch、CUDA、vLLM，也不下载模型：

```bash
cd /workspace/vllm-RedKnot
uv pip install --python /path/to/isolated/.venv/bin/python --no-deps .
CUDA_VISIBLE_DEVICES='' /path/to/isolated/.venv/bin/python -m vllm_redknot doctor
CUDA_VISIBLE_DEVICES='' PYTHONPATH=. /path/to/isolated/.venv/bin/python -B \
  -m unittest discover -s tests -v
```

未设置 `VLLM_REDKNOT_CONFIG` 时，插件注册入口不修改 runner 或 backend。
数值测试需要 Torch；没有 Torch 时会明确 skip，不算通过数值验证。
`doctor` 只检查源码与配置，不代表 GPU 推理成功。

## 配置与启动

参考 `examples/config.template.json`，务必填写真实 checkpoint revision、
与模型一致的 `rope_theta`，以及该模型已验证的 local **KV-head** 分类。
模板中的 head 0 只是格式示例，不是推荐比例。层号未列出的层保持原生计算。
`rotary_dim: null` 由实际模型推导并校验。

非前缀独立 chunk 会改变跨 chunk 上下文，属于显式可控近似，不保证与 dense
数学等价。必须同时在配置及请求中设置 `allow_approximate: true` 才能复用。
默认拒绝近似请求并退回原生计算。建议先用模型专属验证集选定头比例和
`boundary_tokens`，保留若干前层 dense，不要跨模型直接复用头编号。

可导入 SGLang 的 MHA/GQA `kv_head_classification` JSON：

```bash
.venv/bin/python -m vllm_redknot import-head-policy /path/to/head_policy.json \
  --model-revision CHECKPOINT_REVISION --rope-theta ACTUAL_ROPE_THETA
```

它只输出分类配置；不会搬用尚未适配的 retrieval 窗口、FFN 比例或 MLA 布局。

先检查启动命令；去掉 `--dry-run` 才会真正使用 GPU：

```bash
.venv/bin/python -m vllm_redknot serve \
  --model /workspace/Models/YOUR_LOCAL_MODEL \
  --config /path/to/reviewed-config.json --max-model-len 32768 --dry-run
```

默认监听 `127.0.0.1:18731`。禁止将研究版未鉴权服务直接暴露到公网。
复用请求必须来自可信客户端；namespace 是隔离键，不是认证机制。
当前没有 HTTP 错误映射/多租户权限层，非法计划可能导致该执行步报错。

### DeepSeek V4 Flash 专用路径

`examples/deepseek_v4_flash_policy.json` 提供显式 Flash 配置：第 3–39 层
各 56 个 local **query heads**，global 为 0/8/16/24/32/40/48/56；其他层
全部原生计算。这个比例来自 Flash 参考策略，尚未在本移植版完成模型精度验收。
它不是 56 个独立 KV heads：Flash 的 latent KV 是共享的。

```bash
.venv/bin/python -m vllm_redknot serve \
  --model /workspace/Models/DeepSeek-V4-Flash-0731 \
  --config examples/deepseek_v4_flash_policy.json \
  --dtype bfloat16 --max-model-len 4096 --dry-run
```

专用 backend 是 `FLASHMLA_SPARSE_DSV4`，不是 `CUSTOM`。目前只支持 TP1
的 Flash-0731 几何和固定 revision。配置中的 revision 是调用方声明，
不是权重校验；加载前仍须独立验证完整 checkpoint，不能加载未完成下载。

- capture：不改变原生输出，另用 masked local attention 经过原生 inverse
  RoPE 和 FP8 `wo_a`，保存 CPU `z_off`。每个 chunk 全层原子提交。
- reuse：保持全量原生 KV、compressor、indexer；global 行全算，local 仅算
  dirty/query 行。使用 16-head tile 的选头内核，不补齐为原生 64 头伪省算。
- 原生 inverse RoPE + `wo_a` 生成在线 z，clean 行加入对应源位置的 `z_off`，
  最后调用一次 `wo_b`。不对 `z_off` 再次施加 RoPE。decode 保持原生执行。
- 当前 `wo_a` 是 masked 输入的**全宽** FP8 运算，未省其 GEMM；也不跳过
  FFN/MoE、latent KV 写入或压缩器。BF16 部分和有舍入差异，非前缀上下文
  复用另有算法近似，二者都必须用真实生成验证。

Flash CPU payload 字节为 `sum(T * (2 * G * R + 8))`，包括 BF16 z 和
int64 源位置，不包含 Python 元数据、临时张量、原生 GPU KV/权重。
示例 8 GiB 是按需使用的缓存上限，非预分配；长上下文可能超预算而回退。

选头内核的显式 GPU 微检查（先确认 GPU 空闲）：

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. .venv/bin/python \
  benchmarks/check_dsv4_sparse_kernel.py --run --device cuda:0
```

微内核数值通过不等于模型复现成功。小输入下拷贝、校验同步和分次 launch
可能使本路径比原生更慢，当前没有完整模型 TTFT/F1 加速结果。

## 请求协议

使用原生 `SamplingParams(extra_args={"redknot": PLAN})`。
在线复用前，先将每个完整 chunk 作为独立 prompt 发送一次 `capture`。
chunk 范围按 **token 下标**表示左闭右开区间，不是字符数或物理 block id。

```python
capture_plan = {
    "mode": "capture",
    "namespace": "my-corpus-v1",
    "chunks": [{"start": 0, "end": len(chunk_token_ids)}],
}
reuse_plan = {
    "mode": "reuse",
    "namespace": "my-corpus-v1",
    "allow_approximate": True,
    "chunks": [{"start": 20, "end": 20 + len(chunk_token_ids)}],
}
```

复用计划中的 token 片段必须与 capture 的 token 完全一致。可包含多个有序、
不重叠的 chunk；它们之间、之前、之后的所有非 chunk 行均在线重算。
移位 chunk 的前 `boundary_tokens` 行也重算。任何 chunk/层缓存缺失、
身份不符或执行模式不支持，都在跳算前整步退回原生。
选择 `mode: recomputed` 可显式走同一 backend 的原生基线。

`runtime.py` 管理计划与事务，`runner.py` 校验整步几何并传递上下文，
`vllm_backend.py` 执行分头 attention 和 KV scatter。租约在异常时也释放，
只有所有目标层 capture 成功后才把完整 chunk 提交到缓存。
Flash 对应使用 `dsv4_runtime.py`、`dsv4_runner.py`、`dsv4_backend.py`，
共享计划与缓存管理但不覆盖独立 local KV（共享 latent KV 始终在线更新）。

CPU 缓存 payload 字节可用下式估计（求和仅包含配置中的层）：

`sum(T * D * (2 * local_KV_heads + local_Q_heads) * dtype_bytes)`

预算不包含 Python 元数据、采集中的临时张量、GPU 原生 KV 与在线 scratch。
不减少 vLLM 原生 KV 预留显存；不能把 CPU payload 字节报告成总进程内存。

## 复现与指标

`benchmarks/benchmark_redknot.py` 提供成对 recomputed / reuse 实验入口，
CPU 测试覆盖结果统计与失效检测。具体参数用 `--help` 查看。

- 输入为 token 化的 case/chunks/query，最多 8 块；可提供参考答案。
- 单独报告离线 capture 和预热耗时，至少 3 轮不计时预热。
- 至少 10 组成对热态测试，交替次序；相同 prompt、贪心固定长输出。
- 使用 vLLM 的 `first_token_latency`，不混减 wall-clock 与 monotonic 时间。
- 输出原始答案、热态 TTFT、F1 与下降百分点；无参考答案时保留原始输出供比对，
  F1 留空，不假装是任务准确率。没有真实命中/跳过计算计数时标为未合格。
- 单请求测试不声称 QPS 或并行提升。CPU 契约测试不替代 B300/H200 模型实测。

设计映射和下一步扩展见 `docs/PORTING.md`。实际执行证据见
`docs/VALIDATION.md`；不要把尚未执行的 GPU 结果写入发布宣传。
