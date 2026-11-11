# KV Cache Engram：独立原生 GPU 小批量验证

本目录不导入、不修改、不恢复 v8，也不恢复其他已暂停实验。GPU 启动只能经
`launch_pilot.py` 调用身份固定的原有 GPU lifecycle supervisor；它负责准备期保留占卡、
释放、实验进程清理和恢复。启动器在前后分别核验八卡实际上下文和 holder 进程身份。

## 冻结验证范围

- HotpotQA、2WikiMQA 各 1 题校准、各 2 题评估，共 6 个请求。
- 每题上下文恰为 8 × 32768 = 262144 tokens，另加模板和问题；真实 LongBench
  材料扩展组合，并非原生 256K 题。保留完整 primary，不按结果换题。
- 按来源文本先划分 calibration/evaluation，排除已登记的旧 50 题及 donor 材料。
  此来源隔离不等于证明与模型训练材料或语义内容独立。
- 同模型、硬件、配置下比较标准全量在线计算与 gated KV-only 路径。
  完整 Q、attention、hidden、MoE 和原生 global KV/index 共享保持不变；只尝试
  在 14 个 encoder 层减少 SWA KV 投影/归一化的在线行数。
- 每个实际 32K 缓存块首尾 128 行、模板、问题和 native chunk 的跨块残片强制重算。
  512 行为 gate 单位，使用当前 hybrid hidden 计算探针，不使用全量参考答案决定复用。

## 顺序与准入

1. `split_identity` 在真实 8×32K 输入上检查完整 Q/KV 拆分、packed-SF gather、
   canonical KV 及 RoPE/cast 后数值一致性；失败立即中止，不进入近似复用。
2. 各独立 32K 文档生成离线 canonical BF16 KV。校准题另生成全量上下文 KV，
   保存逐层、逐文档的两份原始快照及摘要，不覆盖旧证据。
3. 校准题执行全行误差核验，测试预声明的 32/128 个分层探针和四档误差阈值。
   全行工程判据固定为相对 L2 ≤ 0.01 且绝对误差 ≤ 0.05；这些值不是 EM/F1 容忍度。
4. 每层只从校准题中选择无误放块、无超标复用行的候选；没有合格候选则全量重算。
   选择后冻结，不根据四题评估结果调阈值。探针本身不能证明所有未采样行安全。
5. 评估题保留独立全量 shadow 核验、真实 gated 生成及两臂 C1 测速。
   shadow 使用干净全量 hidden，不能冒充运行时 hybrid hidden 下的全行误差证明；
   最终还须以实际生成 EM/F1 判断累计误差。

校准通过仅允许本次隔离 pilot，不产生线上部署准入证书。CPU 测试或数值资格通过，
均不等于精度、吞吐或延迟收益通过。

## 指标和负结果

- 质量：自然 EOS、最多 64 tokens，对固定 gold 计算 EM/F1；保留完整回答、零分和截断。
- 性能：C1 每臂预热后各 3 轮，固定生成 32 tokens；两臂顺序按
  `(case_index + case_index//2 + repeat) % 2` 轮换，各数据集内部平衡。
- 主要指标为含线上计划、探针、gate、缓存恢复及回退开销的热态 TTFT。
  模型加载、离线 KV 构建和诊断性全量 shadow 不计入该热态指标，构建耗时另列；
  不声称冷启动端到端加速。本 pilot 不测 C4。
- 实际复用率是 14 层 SWA KV 行的计数，不是全模型算力节省率。零复用、变慢、
  精度回退及 heldout 误放均保留，不择优删题。
- 有状态执行错误使请求失效并中止本次进程，不在已污染状态上伪装成安全全量结果；
  原 supervisor 负责清理和恢复占卡，不自动重试。

## CPU 与只读结果检查

```bash
python3 GatedEngram/validate_cpu.py --output /absolute/new-cpu-evidence.json
python3 GatedEngram/gpu_pilot/read_results.py /absolute/run/benchmark
```

结果分析器只纳入两臂质量和三对 C1 均完成、独立 raw phase/冻结配置/校准封存校验通过的
评估题。校准题不计入精度。首个完整 HotpotQA 或 2WikiMQA 配对即可读取，不必等整批。
不足四题时明确分母；两题/集只能用于小样本工程判断，不能声称统计显著或稳定泛化。
