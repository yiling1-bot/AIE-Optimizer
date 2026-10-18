# 第十四轮：通用 SSA、多输出等价类与可生成的共享 DAG

本轮接通“原函数语义 → 等价类 → 全程序共享 DAG → 完整代码生成”。
旧两层 `Expr` 路径保留；通用静态整数张量 SSA 路径使用同一个共享
DAG 抽取引擎，按原函数的全部输出联合选择实现。

**AIE1 自动生成和真实编译绑定尚未接通，不能据此宣称 VCK190 已能
自动优化任意导入程序。** 本轮没有新原生编译、Simulator、周期采集
或成本拟合；算术工作量与器件延迟分别记录。

## 1. 等价类直接接通 SSA

`Optimizer/program_egraph.py` 的 `TensorEGraph` 消费经过重放校验的
`aie-optimizer-tensor-ssa-v1`。节点键包含算子、结果类型、语义属性和
子等价类；输入按原函数参数位置区分。源位置、改写来源和保守整数
区间不作为值身份，因此相同计算能跨源位置共享。

所有结果按原签名顺序保存为根，包括重复结果、直接返回输入的结果
以及不同形状/位宽的结果。未使用参数仍保留。同余闭包合并相等
子表达式的用户；内部可以有循环等价类，抽取必须选择有限无环 DAG。
范围和初始化事实在恢复 SSA 时由 `GraphBuilder` 重新推导。

现有 `ProgramRuleRegistry.register` 提供匹配、守卫、改写及版本。
引擎将原始项的有限祖先图构造成局部见证，重放规则，把旧根与新根
合并。它不为每种局部组合复制整份输入程序，但匹配暂时仍要构造
局部见证，尚未实现大规模增量 e-matching。

## 2. 八个可调用规则族

`Optimizer/program_equalities.py` 复用 M/K 规则并增加以下能力：

| 规则族 | 实际改写与守卫 |
| --- | --- |
| `tensor.matmul.partition_k` | 分割归约；原初始化只计一次；模整数累加，保持原转换位置 |
| `tensor.matmul.partition_m` | 行分片及初始化行切片，按原顺序拼接 |
| `tensor.matmul.partition_n` | 列分片及初始化列切片，按原顺序拼接 |
| `tensor.elementwise.commute` | 同类型 add/mul/min/max 交换；不交换矩阵乘操作数 |
| `tensor.elementwise.associate` | 同宽模整数 add/mul 重结合 |
| `tensor.matmul.associate` | `(AB)C ↔ A(BC)`；乘数/累加器同宽，两级初始化为显式零，无中间截断 |
| `tensor.matmul.factor_left` | `AB+Z0+AC+Z1 → A(B+C)+(Z0+Z1)`；公共 A，B/C 同类型，乘数均为累加位宽 |
| `tensor.transpose.compose` | 合成逻辑转置；恒等排列可直接返回源值 |

矩阵链和因子合并仅适用于这里声明的整数模环，不能越过 INT8 量化、
饱和或不同宽度的 signed promotion。未知条件拒绝匹配。规则族数量
不代表已系统覆盖八篇或全部 AIE 论文。

## 3. 全程序抽取与结构覆盖

`global_extract.extract_dags` 的一个变量是一整个等价类的实现。多个
输出和用户引用同一类时，该实现只选择、计算及计费一次。抽取按
真实子依赖拒绝环，并使用显式遍历避免深 SSA 依赖 Python 递归。

`Optimizer/program_search.py` 的 `enumerate_egraph_plans`：

1. 保留原可执行参考，在预算内展开等价类。
2. 为不同规则族构造第一批完整见证，再展开组合，避免分块组合
   占满深度优先遍历的前几项。见证补全优先保留原计算；该偏好只
   影响遍历顺序，不把节点数、MAC 或内存当作预测时间。
3. 声明预算内保留的完整 DAG 全部进入部署规划；不同 DAG 轮流
   获得物理名额，组合连续拓扑分区、融合、标量循环和注册式微内核。
4. 按生成计算、接口和连接去重，再调用
   `evaluator(certified_contract, plan)`。只在相同工作量/成本场景内
   排条件延迟与核数前沿，保留缺测候选与参考。

语义遍历以零作为通用时间下界，**不作 MAC→延迟换算剪枝**。
状态、应用、项数、DAG 与物理方案均有明确预算。报告保存耗尽、
遗漏、拒绝和未探索状态。抽取引擎在所给有限图/成本空间内的性质，
不等于整个联合搜索或硬件延迟已达到全局最优。

`max_states` 分别约束每次见证构造和主遍历，不是整轮总状态数。
最多一次原结构见证、每条已应用规则一次见证及一次主遍历；总遍历
状态上界为 `(实际规则应用数 + 2) × max_states`。主报告的
`dag_extraction.states` 只计主遍历，不能当整轮运行开销。

当前仍是每类一个代表的共享 DAG。任意克隆/重算、使用点不同布局、
任意任务分区、一般路由和完整核复用调度仍需扩展。

## 4. 证明必须经过生成入口

`aie-optimizer-tensor-egraph-extraction-v1` 保存原 SSA 身份、引擎源码
身份、规则目录/版本、实际局部步骤、原输出根与最终选择。
`check_extraction` 从原程序重建图，重放规则和同余闭包，再生成 SSA；
不接受提交的 union-find 状态或自称相等的 ID。

新来源 `verified_egraph_extraction` 包含原契约与证书。契约、计划、
代码生成和生成物复核都要重放证明。禁止嵌套此来源；继续探索须从
原契约重新开始。可信 Python 规则与代数论证是可信边界，证书不是
外部形式化证明。

`program_backend` 使用**原程序**产生参考；原来源为 C++ 时继续独立
运行原 C++，为 MLIR 时继续执行原标量区域。生成物保存
`equivalence.json`，导入来源保存 `source_import.json`。不会仅用优化
后的图校验优化后的代码。

`program_costs` 先校验证明，再以原契约作为比较身份；组件签名仍按
当前生成体、类型、端口与位置区分。这解决可信语义变体在条件成本
接口的比较身份，**没有放宽 AIE1 同编译限制**，也没有为未编译 SSA
绑定任意现存 ELF。

## 5. 命令与扩展

```bash
python -m Optimizer program-egraph-space program.json --max-dags 16 --max-plans 64 --output egraph-space.json
python -m Optimizer program-egraph-generate egraph-space.json --index 1 --frames 4 --output candidate
python -m Optimizer cpu candidate
python tools/program_egraph_witness.py --output egraph-witness
```

`program-egraph-space --evaluation config.json` 可传组件 registry 和
selection；完整覆盖才返回条件时间。无评价器时全部保持
`needs_evaluator`，索引不是性能顺序。

第三方规则通过同一个注册表传给 `TensorEGraph`、契约/计划验证及
生成的 `rewrite_registry`；目录和版本必须一致。

## 6. 验证与证据

- 全部224项软件回归通过，无失败、错误或跳过；三个 CLI 步骤贯通。
- 新增18项专项测试：极值模整数、非法转换/初始化边界、同余、
  多输出共享、小图独立穷举、循环拒绝、1050层 SSA、证明篡改、
  扩展规则、遍历预算、结构名额及独立源参考生成。
- 关键反例：单输出成本12的因子合并，加入共享输出后成本22；
  联合抽取选择成本21的共享原结构。10/1是软件检查成本，非器件周期。
- 六类见证复用同一引擎：三层量化图、FIR、双输出分支、C++ 残差、
  同宽矩阵链和共享因子图。16个工程、4帧、1040个输出值检查通过，
  其中10个工程核对原 MLIR/C++ 来源。
- 矩阵链见证生成256与64个整数 MAC 的同核方案，均数值通过；
  这证明计算结构改变，尚未证明延迟降低四倍。
- 评价测试故意让算术量较大的结构条件延迟更低，确认不会先按
  MAC 淘汰它。软件成本没有进入真实校准数据。

最终本地材料在 `output/research/AIE-Optimizer/phase14/`：
`reuse-witness-final/`、`regression-final.*`、`model-compatibility.json`。
`reuse-witness/` 是发现遍历偏置前的中间记录，不是当前源码的最终
证明重放材料。

生成器身份已改变，并纳入等价引擎/规则源码。第十二轮模型在当前
源码上按设计拒绝，原因 `cost model identity/implementation changed`；
历史归档和性能结果保留，不能只换摘要后继续用于新代码。下一次
原生成本实验应冻结新生成器，重新声明 TRAIN/留出角色，再采样。

## 下一段实现

优先补 AIE1 目标实现注册与生成，核对原语义证书 → 生成文件 →
真实编译输入/ELF/映射的绑定，再开放 VCK190 跨编译与跨核数比较。
AIE1 数值模型不迁移为 AIE-ML 数值。共享/重算实例、布局、DMA/锁/
缓冲和跨帧成本还需扩展及留出验证。
