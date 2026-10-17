# 架构与源码阅读路线

## 1. 输入程序如何变成数值契约

从 `Optimizer/__main__.py` 看命令分派，再读 `contracts.py` 与 `semantics.py`。
契约把数学函数、数值范围、目标资源、输入交付条件和量化位置固定下来。
核数是实现选择，不是算法语义的一部分。

`importing/mlir_frontend.py` 通过 xDSL 解析 MLIR，`cpp_capture.py` 捕获
Clang 类型、操作符、编译选项和源闭包，`cpp_lift.py` 恢复静态循环计算。
二者归一到 `importing/ir.py` 的张量 SSA。
`admission.py` 再检查这个程序是否能由当前两层后端完整实现。
因此“能导入”与“能生成”是两次独立检查。

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

`global_extract.py` 对共享 DAG 做有界抽取，显式考虑共享节点和
重算边界。`egraph_planner.py` 把抽出的表达式绑定到可生成计划。
循环 e-graph 和无法绑定的表达式不会被直接当成可执行程序。

## 3. 算法、核与执行如何一起选择

`execution_ir.py` 描述 operator DAG、输入输出端口、核、动作和存储。
`execution_rules.py` 改写融合、寄存器网格、输出目标、循环次序、
临时空间和 acquire 顺序；`scratch_schedule.py` 搜索拓扑执行次序。

`joint_extract.py` 将完整 DAG 选择、行分工、核部署、汇聚和执行变体
送往同一个评价器，避免提前用 MAC 数或 scratch 大小代替延迟剪枝。
搜索仍有表达式、执行深度、计划数和状态预算；预算耗尽不等于找到最优。

## 4. Analyzer 如何评价一个实现

`analyzer_bridge.py` 调用随仓库提供的 `Model/Timing` 支撑模块。
`phase_model.py` 将执行计划降成 acquire/compute/copy/release 阶段图，
绑定输入、缓冲槽位、别名和资源顺序；`phase_costs.py` 检查组件成本
适用条件、证据及成本族，再计算条件时间。

`stage_costs.py` 记录核内工作量和成本覆盖，`compiled_phase_inventory.py`
与 `compiled_profile.py` 读取编译产物中的映射与资源证据。
缺少覆盖的成本不会变成零；死锁也不会被当成普通有限延迟。

条件成本排序和真实仿真前沿分别保存。尚未完成 AIE-ML 全阶段校准
与留出排名验证，不能用合成阶段成本宣称获得硬件最优实现。

## 5. 如何得到可验证代码

`ir_backend.py` 将受支持的执行 IR 降成核内代码，`backend.py` 生成
完整 ADF 工程、输入、golden 和 manifest。
`compute.hpp.in` 与 `operator_runtime.hpp.in` 提供核内模板。

`runner.py` 先核对契约、计划和文件摘要，再运行 CPU 或原生流程，
复核完整输出、映射、接口与计时条件。
`frontier.py` 仅在相同条件下构建已验证的延迟—核数前沿。
`search.py` 管理有时间与次数预算的评估，不把工具失败当成数学不等价。

## 扩展一条规则的最小闭环

1. 明确等价关系及形状、量化、溢出、别名和生命周期条件。
2. 在对应注册表实现匹配、检查、改写及版本化元数据。
3. 保证执行 IR 校验、后端降级和资源检查能够处理结果。
4. 添加语义重放、拒绝条件和独立数值参考的测试。
5. 补齐成本覆盖与原生验证后，才将结果纳入硬件性能比较。

单独把论文机制登记成规格卡，不会自动获得上述可执行能力。
