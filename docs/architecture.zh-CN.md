# 架构与源码阅读路线

## 1. 输入程序如何变成数值契约

从 `Optimizer/__main__.py` 看命令分派，再读 `contracts.py` 与 `semantics.py`。
契约把数学函数、数值范围、目标资源、输入交付条件和量化位置固定下来。
核数是实现选择，不是算法语义的一部分。

`importing/mlir_frontend.py` 通过 xDSL 解析 MLIR，`cpp_capture.py` 捕获
Clang 类型、操作符、编译选项和源闭包，`cpp_lift.py` 恢复静态循环计算。
二者归一到 `importing/ir.py` 的张量 SSA。
`admission.py` 检查这个程序是否能由既有两层 MMUL 后端完整实现。
因此“能导入”与“能生成”是两次独立检查。

第十一轮新增 `program_ir.py`：对整个 SSA 图重新做类型、范围、初始化和
依赖重放，将 C++ 源参数位置适配为稠密 SSA 输入，保留可逆输入绑定和
完整源签名。`program-*` 路径直接使用此图，不依赖两层 Spec。

`importing/check.py` 提供源与 IR 的交叉检查；C++ 原程序参考由
`cpp_check.py` 使用独立主机编译器执行。`native_ast.py` 只审计目标
编译器的 AST，尚未实现 AIE API 的自动数值语义恢复。

## 2. 同一算法如何拥有多种实现

`registry.py` 定义 `Rule`、匹配、条件检查、改写与重放。
`builtin_rules.py` 和 `egraph_rules.py` 提供张量分解及等价规则。
规则需保持原数值契约，拒绝未知或不合法条件。

`egraph.py` 管理 e-class、节点共享、同余闭包及证书记录；
`egraph_rules.py` 在预算内展开等价空间。
这里的证书由当前可信规则实现重放检查，不是独立形式化证明。

`program_rules.py` 是同一张量 SSA 上的通用规则注册接口。
`program_equalities.py` 注册8个可调用规则族，包括 M/K/N 分解、
同宽整数矩阵链重结合/因子合并和逐元素/转置等价。

第十四轮 `program_egraph.py` 将完整 SSA 接入类型等价类、hash-consing
和同余闭包，支持全部输出作为联合根。原始项保留无环见证；循环
等价类只能抽取有限无环实现。来源和区间不参与值身份，恢复 SSA
时重新检查类型、初始化与区间。证明从原图重放规则，不信任序列化
的并查集；新来源契约把证明绑定到计划与生成入口。

`global_extract.py` 对共享 DAG 做有界抽取，显式考虑共享节点和
重算边界。`egraph_planner.py` 把抽出的表达式绑定到可生成计划。
循环 e-graph 和无法绑定的表达式不会被直接当成可执行程序。
新旧路径复用该抽取引擎；通用路径当前每类选择一个共享代表，尚不
表达任意克隆与使用点不同的物理布局。

## 3. 算法、核与执行如何一起选择

`execution_ir.py` 描述 operator DAG、输入输出端口、核、动作和存储。
`execution_rules.py` 改写融合、寄存器网格、输出目标、循环次序、
临时空间和 acquire 顺序；`scratch_schedule.py` 搜索拓扑执行次序。

`joint_extract.py` 将完整 DAG 选择、行分工、核部署、汇聚和执行变体
送往同一个评价器，避免提前用 MAC 数或 scratch 大小代替延迟剪枝。
搜索仍有表达式、执行深度、计划数和状态预算；预算耗尽不等于找到最优。

`program_planner.py` 对完整 SSA 组合语义改写、单使用者内联融合、
拓扑图切分和逐节点实现族，支持多个输入/输出及核。按生成计算和连接
合并相同实现；未知成本候选与参考实现保留。当前分区仍限定为连续拓扑区间。

`program_kernels.py` 将实现 ID/版本、适用守卫、生成回调和暂存需求
注册在一起。规划器发现合法选项，绑定每个物化节点的实现；生成器
调用回调。当前包括标量 ijk/ikj 和 AIE-ML MMUL grid1x1/1x2，
与融合、分工和核部署组合。搜索受总计划预算和实现预算约束。

`program_search.py` 先给不同规则族分配完整见证名额，再展开多输出
DAG 组合；各 DAG 轮流获得物理方案名额。参考、缺测与预算遗漏均
记录。完整契约/计划送评价器，语义遍历用零时间下界，不按 MAC
提前筛选；物理分区与实现组合仍为明确预算内的有限提议集。

## 4. Analyzer 如何评价一个实现

`analyzer_bridge.py` 调用随仓库提供的 `Model/Timing` 支撑模块。
`phase_model.py` 将执行计划降成 acquire/compute/copy/release 阶段图，
绑定输入、缓冲槽位、别名和资源顺序；`phase_costs.py` 检查组件成本
适用条件、证据及成本族，再计算条件时间。

`stage_costs.py` 记录核内工作量和成本覆盖，`compiled_phase_inventory.py`
与 `compiled_profile.py` 读取编译产物中的映射与资源证据。
缺少覆盖的成本不会变成零；死锁也不会被当成普通有限延迟。

条件成本排序和真实仿真前沿分别保存。尚未完成 AIE-ML 全阶段校准，
不能用合成阶段成本宣称获得硬件最优实现。

`program_costs.py` 对新通用路径产生首帧组件签名并复用同一个 Analyzer
成本注册表与时钟因果图求解器。签名包含算子、类型、访问变换、融合、
端口、位置和生成体摘要。它尚未绑定真实编译通道或跨帧槽位，不能继承
旧路径的 buffer 动态模型覆盖，也不能自动复用旧两层的测量成本。

`program_profile.py` 在相同 ELF 上复跑 profiler，要求输出值和时间与
普通仿真一致，保存 outlined 计算区域的 inclusive cycles 和 ELF 摘要。
`program_cost_model.py` 只读取冻结 TRAIN 集，拟合限域 GEMM 的计算
及合并非计算残差，再通过时钟因果图预测。名义 PLIO 服务与残差的
相加是有条件的首帧近似，不等于独立校准锁、DMA 与重叠协议。
留出预测先锁定，再运行候选并复核误差、顺序和选择损失。

第十三轮另加 `compiled_analyzer.py` 与隔离 worker，冻结完整外部
AIE1 Analyzer 的源码，通过正常提取、准备和预测接口评价已有编译
候选。`compiled_search.py` 复用条件排名，但按实际编译、输入内容与
成本场景分组；相同请求及解析上下文的复用继续检查完整文件 pin。
`packet_schedules.py` 提供逐目的端 FIFO 不变的调度见证。

这条路径没有把 AIE-ML 生成器变成 AIE1。当前 `AnalyzerEvaluator`
需要已有 compiled binding；ID/核数对应不替代数学等价证明。通用
SSA/e-graph 已建立到生成入口的语义证据；仍需 AIE1 生成及真实
编译绑定，才能开放该评价器的跨结构、跨核数比较。
`program_costs.py` 只在校验证书后，以原契约归一条件成本比较身份；
不会把 AIE-ML 源程序对应到任意已有 AIE1 ELF。
见 [第十三轮报告](analyzer-integration-v13.zh-CN.md)和
[第十四轮实现](tensor-egraph-v14.zh-CN.md)。

## 5. 如何得到可验证代码

`ir_backend.py` 将受支持的执行 IR 降成核内代码，`backend.py` 生成
完整 ADF 工程、输入、golden 和 manifest。
`compute.hpp.in` 与 `operator_runtime.hpp.in` 提供核内模板。

`runner.py` 先核对契约、计划和文件摘要，再运行 CPU 或原生流程，
复核完整输出、映射、接口与计时条件。
`frontier.py` 仅在相同条件下构建已验证的延迟—核数前沿。
`search.py` 管理有时间与次数预算的评估，不把工具失败当成数学不等价。

`program_backend.py` 按算子组合生成新路径的计算函数、多个 ADF 端口、
graph 和独立源参考见证；i8/i16/i32 的模整数语义通过显式无符号累加
实现。`program_native.py` 检查实际映射、每个 PLIO 类型和全部输出。
新路径按 `program_kernels.py` 回调生成物化算子；MMUL 的逻辑行主序
打包和结果散写也在生成函数内。主机执行使用对应微块的标量仿真，
AIE 指令、编译映射及延迟由 Vitis 原生流程验证。
等价抽取生成继续用原始程序作为参考，原 C++/MLIR 来源不被优化图
替代。生成器身份包括等价引擎/规则文件；旧冻结模型与新源码不匹配
时拒绝使用，历史性能证据保留在原归档中。

## 扩展一条规则的最小闭环

1. 明确等价关系及形状、量化、溢出、别名和生命周期条件。
2. 在对应注册表实现匹配、检查、改写及版本化元数据。
3. 保证执行 IR 校验、后端降级和资源检查能够处理结果。
4. 添加语义重放、拒绝条件和独立数值参考的测试。
5. 补齐成本覆盖与原生验证后，才将结果纳入硬件性能比较。

单独把论文机制登记成规格卡，不会自动获得上述可执行能力。
