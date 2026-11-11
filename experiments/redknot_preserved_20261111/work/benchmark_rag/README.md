# benchmark_rag_redknot.py

保存的 Full / S1.2 长上下文多文档 QA benchmark。入口支持原固定输入中的 **64K、128K 或两者**，S1.2 固定使用 group17 与 world-idle bypass。Full 是标准重算；原 S1.1 不再作为正式比较臂。

**本轮实验已经按用户要求暂停。打包和 CPU 检查不会继续 GPU 实验。** 本目录保留已完成的三题及全部来源；其余七题没有完成。未来 GPU 执行必须显式使用 `--run`。

交付检查已完成：服务器新包装的 37 项 CPU 回归通过；工作区入口实际执行 `--length both --cases 10 --check-only`，通过全部 20 条原始 64K/128K 输入的真实 CPU 预检，`gpu_initialized=false`、`gpu_released=false`。证据为 [delivery-check.json](delivery-check.json)、[package-cpu-proof.json](package-cpu-proof.json) 与 [validation/cpu-preflight.json](validation/cpu-preflight.json)。本地及服务器 NativeSnapshot 均已保存完整 403 个源文件。此检查不宣称新入口已完成 GPU 运行或 64K 性能测试。

## 入口与用法

工作区的 `benchmark_rag_redknot.py` 是连接已配置服务器包的入口；服务器包中的同名文件直接执行本包。无 action 时只显示帮助。

```sh
# 只检查原输入、冻结源码、配置和 CPU preflight；不会查询/释放 GPU。
python benchmark_rag_redknot.py --length both --cases 1 --check-only

# 未来明确决定进行新 GPU 实验时才执行：
python benchmark_rag_redknot.py --length 64k --cases 1 --run
python benchmark_rag_redknot.py --length 128k --cases 1 --run
python benchmark_rag_redknot.py --length both --cases 1 --run
```

参数：

- `--length {64k,128k,both}`：默认 `128k`。`both` 先执行所选 64K，再执行所选 128K。
- `--cases {1..10}`：默认 1，表示**每个所选长度**的题数；`both --cases 3` 共执行 6 题。每个长度按原题序取前 n 题。
- `--check-only` 与 `--run` 互斥。CPU 检查不会证明 GPU 数值资格或性能。
- 服务器入口另支持 `--definition PATH`，默认同目录 `benchmark-definition.json`；Mac 代理使用已固定的服务器定义。
- 服务器入口另支持 `--output-root PATH`，默认同目录 `runs/`，每次创建新的独立输出目录。

服务器入口根据 definition 使用固定的 `child_python`。需要完整 NativeSnapshot 的 403 个原始文件、原始 manifest 与 companion 输入文件、有效模型和工具链路径；GPU 执行还要求 definition 中存在实际 CPU proof 引用。原八卡 holder 必须正在运行，以便生命周期 wrapper 认证、临时释放并在结束/错误/超时后恢复。不要从只含 390 个文件的旧本地 native 镜像直接推定完整服务器包可运行。

新实验使用原生命周期实现，从**该次实际 GPU release** 起最多 7200 秒，并恢复原 holder；服务器入口的 Ctrl-C 由 diPython 验证身份后直接转发给原 supervisor，并等待清理。Mac 代理的 `--run` 将完整服务器入口独立启动，Mac 断开不会中断服务器推理和最终审计。自然 EOS 可能生成很长回答，不能保证所选全部题目在时间预算内完成。只接纳完整闭合的题目。

## 执行与审计合同

64K 原题号为 `3,5,…,21`，128K 为 `2,4,…,20`。实际输入来自同一固定原 manifest，SHA-256 seal 为 `fb6bba62937fb8005f05df9a3a213729c172932a30e5509b136000692881aa67`，不进行本次人工 padding 或截断。它们是官方固定 Loong 自然多文档 QA，**没有新增检索器评测**；问题数不等于独立上下文数。

每次未来运行必须重新完成实际八 rank native/metadata 资格；每题建立新 bank，并完成同一稀疏策略 dense group1 与 S1.2 group17 的数值资格：全部原始 chunk / 40 层输出、四类 mHC、source2/cache 状态，以及完整最多 32 token 或提前 EOS 的生成轨迹。64K 的私有审计适配只扩展输入 band/index/ordinal 合同，不修改冻结的算法、native 或 group17 runtime。

正式比较的每一臂都有自然 EOS 完整回答、一次固定 32 token warmup、三次轮换顺序的 clean online TTFT。完整质量回答没有人为输出 token 上限，技术上受模型 270336 总上下文预算与运行时间预算约束。32 token 数值资格和计时阶段不代替完整质量回答。

`read_s12_reproduction_results.py` 独立验证实际输入、源码、原始记录 SHA/seal、数值资格、八 rank bypass、完整回答和计时。64K 与 128K 分别汇总。它不复用旧运行的数值资格或性能；同策略数值一致也不证明 S1.2 与 Full 的事实准确率相同。

主要实现：

- `run_s12_reproduction.py`：新的参数化运行封装；closure 发布屏障修复保留。
- `reproduction_contract.py`：私有 64K/128K 审计 namespace，冻结模块不被修改。
- `read_s12_reproduction_results.py`：本次运行独立 CPU 审计；完整题目才能进入性能结果。
- `GatedEngram/`、`NativeSnapshot/`、`idle_optimizer.py`、`profile_group_contract.py`：固定实现与依赖。

未来输出位于 `runs/reproduction.*/`，包含 `Run/benchmark/` 原始证据、`Lifecycle/` 清理与 holder 证据，以及独立审计 `result.json`。默认检查不会生成 GPU 性能结果。

## 已保存的暂停结果

本轮原计划为 10 个 128K 问题、6 个共享上下文。实际只接受 **题号 12、14、16：三题、三个上下文**；未完成 `2,4,6,8,10,18,20`。没有新的 64K S1.2 GPU 性能结论。

| 原题号 | 输入 tokens | Full TTFT median / s | S1.2 TTFT median / s | TTFT Full / S1.2 | Full 完整回答 / s · tokens | S1.2 完整回答 / s · tokens |
|---|---:|---:|---:|---:|---:|---:|
| 12 | 129627 | 33.903115 | 7.443482 | 4.554739× | 42.663121 · 70 | 301.250338 · 2323 |
| 14 | 126144 | 33.196811 | 7.214963 | 4.601106× | 46.973313 · 113 | 348.620227 · 2732 |
| 16 | 123440 | 32.236333 | 7.007850 | 4.600032× | 38.398337 · 51 | 10.220902 · 26 |

三题 TTFT median 速度比的几何平均为 **4.585241×**。**这不是完整回答也加速 4.585×。** 例如题号 12、14 的 S1.2 回答明显更长，实际完整回答耗时反而更大。以上耗时都是对应完整自然 EOS 请求的实际单次值，包含输出长度差异，不是等长度 decode 吞吐。

查看 [完整问题、原指令、原始 gold、六份自然 EOS 回答和全部 token IDs](paused-report.html)，或读取 [结构化结果](paused-results.json) 与 [保存结果 manifest](preserved-results-manifest.json)。

## 原始状态与来源

首次运行写完第 12 题完整证据后发生跨 rank closure 文件发布竞态，原 run 保持 `failed`、原 FinalAudit 保持 `invalid`。独立派生审计重新验证全部原门槛，只接受完整第 12 题；没有改写原始记录。恢复运行增加发布屏障后，完成题号 14、16，随后被用户明确暂停。其原始 supervisor 失败状态保留，暂停原因单独由 authenticated SIGTERM receipt 说明。

- 父运行原 invalid audit seal：`26f6aff30d0688e04f248d2a4a31a185dfb168ddb6a4364bfe03acf71fd19816`
- 第 12 题派生 audit seal：`dcc16626ae323b8558daf5e0333f7f4167f852ae71e1720563cca1e7fe5b4e54`
- 恢复运行 partial audit seal：`33550db2868d26595d3f5e836fefae866dd90b3740b250418f0f8547ff8e8713`
- 合并暂停报告 seal：`0494948ca9cbdc462c95ccf7be262c0bed3c41a6ff64299a1f47b8199cd02c60`
- 保存结果 manifest seal：`5c24f61ed1b52bd8938e01bf6af11ff7d525ed7adbc31596c4323070e4a93e69`

最终原 holder 已恢复，实际验证每卡 37 个样本都在 70–90% 带内。采样计划跨度 180 秒，原始查询时间戳实际跨度 `179.98183369636536` 秒，结果保留该精确值。manifest 含原始暂停请求、认证信号回执、terminal、holder 验证和所有审计引用的文件 SHA/seal。

原始 JSON 导出保存在仓库 `outputs/s12-multi128-final/` 与 `outputs/s12-recovery128-final/`；HTML/JSON 报告已内嵌完整接受回答与结果，可独立查看。性能结果按完整单题保留，两臂与重复均来自同一次运行；此前单题锚点的旧测量未混入。
