# 第十二轮：注册式微内核、周期证据与留出选择

日期：2026-10-18。本轮在完整程序 SSA 路径中加入逐算子的实现选择，
把可运行的标量 baseline 扩展为可组合、可比较的实现族。

## 实现族如何进入优化

`program_kernels.py` 定义可调用实现注册表：ID、版本、算子集合、守卫、
代码生成回调、额外暂存量及是否进入探索。规划器从注册表发现合法实现，
为各节点分别绑定选择；生成器调用回调，不按应用名称分派完整模板。

首批实现为：

| ID | 计算组织 | 边界 |
| --- | --- | --- |
| `scalar.elementwise` | 模整数算术、转换和访问变换 | 保持原 SSA 语义 |
| `scalar.matmul.ijk` / `.ikj` | 两种标量矩阵循环顺序 | 通用回退，显式模累加 |
| `aie_ml.mmul.i8.grid1x1` | 4×8×8 MMUL；逻辑行主序打包与输出散写 | i8×i8→i32，完整微块，独立点积和初始化后结果均不溢出 |
| `aie_ml.mmul.i8.grid1x2` | 两个输出微块共享 A 装载、保留独立累加器 | 以上条件，N 为 16 的倍数 |

MMUL 的模式与分块输入解释依据 [AMD AIE API 2024.2](https://download.amd.com/docnav/aiengine/xilinx2024_2/aiengine_api/aie_api/doc/group__group__mmul.html)。
本轮保持逻辑行主序接口，内核内部完成打包；这些指令和临时数组都属于
被评价的实现，不能只报告矩阵乘法指令而忽略打包开销。

原初始化只加入一次，量化操作的位置保持不变。微内核不修改全局饱和/
舍入状态；守卫要求 i32 转换范围成立。非法显式选择会拒绝；搜索仍保留
标量候选和未知成本候选。额外数组进入栈/双缓冲资源检查，实际放置及
代码存储继续由目标编译器验证。

```mermaid
flowchart LR
  A[完整张量 SSA] --> B[逐节点实现注册与守卫]
  B --> C[分工 融合 核部署 实现组合]
  C --> D[生成 ADF 与核代码]
  D --> E[原生输出 映射 ELF 与 profile]
  E --> F[仅 TRAIN 的成本拟合]
  F --> G[冻结留出预测]
  G --> H[留出编译仿真与排名复核]
```

## 周期证据

计算区域保留为 outlined `compute_k*` 函数。`program_profile.py` 使用
同一个已编译 ELF 运行 simulator profile，复核全部数值及输出时间与普通
仿真相同，再记录函数调用数、inclusive 周期、原始 XML 摘要和 ELF 摘要。
不把 self 与 descendant 周期重复相加。

第一版成本族限定单核、单帧、所有输入已获取的完整 GEMM 计算区域。
函数周期是该编译上下文中的 elapsed cycles，包含自身的存储和取指影响，
不能称作无停顿指令周期。profile 复跑不是独立重复实验。

## 成本与候选选择

`program_cost_model.py` 区分三部分：

1. **计算。** 从 TRAIN 原生 profile 的区域周期建立实现族插值模型。
2. **传输。** 按类型、填充后的帧大小及 PLIO 时钟计算名义服务时间；
   它不是已恢复的真实 DMA 仲裁模型。
3. **其余延迟。** TRAIN 首帧时间减去计算和名义传输，得到合并残差。
   启动、锁、release、运行时等贡献尚不能分别辨识。

这三部分进入 Analyzer 的时钟因果图，输出有明确条件的预测与候选前沿。
模型保存源实现指纹、工具/平台/时钟上下文，以及 TRAIN 源码、配置、
报告、输出和 ELF/profile 的字节摘要。数据或实现改变会拒绝复用。

首个模型域是 M=8、N=16、K=8..32（步长 8）、零初始化 i8 GEMM 输出
i32、tile (0,2)、单帧单核。它不覆盖多核、跨帧、量化链、其他算子或
不同编译上下文；不支持的候选返回 `needs_measurement`，不返回零成本。
模型始终保留 `native_admitted=false`，排名验证单独记录。

## 冻结实验设计

设计文件由 `tools/microkernel_campaign.py design` 生成，种子为 20261018：

- TRAIN：K=8 和 K=32，各比较标量 ijk、MMUL grid1x1、grid1x2，共 6 案例。
- HO：K=16 的相同三种实现；在执行之前保存模型与预测锁。
- APP：三层量化链，标量与 MMUL grid1x1，同核、同输入、同接口对照。

核数、工具、时钟、接口和每个形状的输入固定；同角色内随机运行顺序，
最多两个并行工作。实验单位为一个编译候选的一次普通原生运行。
连续帧用于正确性/有限帧间隔，不能增加独立重复数。

拟合接口只接收设计中完整的 TRAIN 集合，拒绝 HO/APP 混入。预测接口
没有候选输出时间参数。先锁定预测，再允许 HO 执行；排名复核输出误差、
成对顺序、选中方案和真实选择损失。不会用 HO 结果回填模型系数。

## 使用入口

```bash
python -m Optimizer program-kernels --output implementations.json
python -m Optimizer program-space program.json --max-implementations 4 --output space.json
python -m Optimizer program-fit-model design.json train-artifacts.json --output model.json
python -m Optimizer program-freeze-predictions design.json model.json --output prediction-lock.json
python -m Optimizer program-predict program.json plan.json model.json --frames 1 --output prediction.json
python -m Optimizer program-space program.json --kernel-model model.json --model-frames 1 --output ranked-space.json
```

`train-artifacts.json` 是冻结 TRAIN 名称到已验证本地原生工程目录的映射。
它不接受仅有手填周期的条目；需要原始编译、输出、配置、ELF 和 profile。
微内核注册、源图规则注册、完整方案抽取、成本注册分别维护自己的边界。

## 本轮验证结果

**187 项回归全部通过，无跳过。** 新增覆盖包括微内核极值与非零初始化、
非法类型/形状/溢出拒绝、逐节点混合实现、外部实现注册、成本签名绑定、
TRAIN/HO 隔离、帧数及编译上下文拒绝、预测锁与计划绑定。

原生平台为 VEK280 / AIE-ML，Vitis 2024.2 build 5239630，
计算时钟 1.25 GHz，128-bit PLIO 200 MHz，实际均为 1 核。
下列时间为**仿真起点到首帧全部输出完成**，包含启动与本地打包开销：

| 案例 | 角色 | 标量 ijk / ns | MMUL grid1x1 / ns | MMUL grid1x2 / ns |
| --- | --- | ---: | ---: | ---: |
| M8 N16 K8 | TRAIN | 3940 | 3295 | 3010 |
| M8 N16 K32 | TRAIN | 67510 | 10110 | 8600 |
| M8 N16 K16 | HO | 5920 | 5595 | 4860 |

同形状的输入、接口、核、工具、平台和时钟的比较签名一致。
K32 的 grid1x2 相对标量为 7.85×，这是校准组中的收益。
留出 K16 的 grid1x2 为 1.218×，首帧延迟降低 17.91%。
不能将 K32 的收益推断为任意程序的收益。

三层量化程序使用 4×8 激活、8×8 权重和 4 帧输入。标量与三个矩阵节点
均用 grid1x1 的实现分别为 3065 ns 和 3005 ns，首帧降低 1.96%。
有限帧间隔分别为 [2340, 2330, 2340] ns 与 [2265, 2270, 2265] ns；
它们不是独立重复，也不代表无限稳态吞吐。两种实现的全部 256 个逻辑
输出检查通过，说明微内核可在完整多层图中组合，但小图收益有限。

11 个候选普通原生运行全部通过，共 17 帧、1408 个逻辑输出。
9 个 GEMM 的同 ELF profile 复跑也通过，输出值和时间与普通运行一致；
不把这些复跑额外算成 9 次独立性能实验。下载后重新检查生成文件、
原始输出、目标映射、接口、时钟、ELF 和 profile。最终生成相关源码与
上传编译快照逐字节相同。

软件跨应用见证复跑三层链、FIR、分支图和 C++ 双输出：在 256 计划预算
下，18 个工程、72 帧、1472 个逻辑输出检查通过。这是正确性见证，
不提供跨应用成本精度证据。64 计划预算只保留到 17 个见证，三层图的
三核方案被预算截断；扩展实现族后，候选族之间的预算分配仍需改进。

### 冻结预测与真实排序

预测锁在 HO 编译/仿真前产生并上传，服务器确认 HO 目录尚不存在后
启动。启动记录绑定设计、模型、预测锁摘要及具体计划，最终复核时也
检查了锁先于运行，且没有使用 HO 或 APP 拟合系数。

| K16 实现 | 冻结预测 / ns | 原生结果 / ns | 绝对相对误差 |
| --- | ---: | ---: | ---: |
| 标量 ijk | 25135 | 5920 | 324.58% |
| MMUL grid1x1 | 5570 | 5595 | 0.45% |
| MMUL grid1x2 | 4875 | 4860 | 0.31% |

三组成对顺序全部正确；选中的 grid1x2 是该留出集合的实测最快方案，
选择损失为 0。CLI `program-space --kernel-model` 产生相同选择，
仍保留另外 2 个没有校准覆盖的候选。

**排序正确不等于模型整体准确。** 标量误差很大，三个候选的平均绝对
相对误差为 108.44%。原始计算区域周期为 K8=3847、K16=6271、
K32=83156，明显不符合跨端点线性增长。编译策略、代码体积或存储行为
是否造成这一变化，仍需反汇编和编译报告诊断；本轮不预设原因，也不
用 K16 结果回填系数。当前只能说这一留出形状选对，不能声称模型已
可靠覆盖全部 AIE-ML 程序，`native_accuracy_admitted` 保持 false。

### 本地证据与复现

证据位于工作区 `output/research/AIE-Optimizer/phase12/`，不随源码仓库
发布 SDK、平台或本机证据路径：

- `design/design.json`：冻结角色、计划、运行顺序、边界。
- `model.json`、`prediction-lock.json`、`prediction-freeze-record.json`：校准、预测与冻结时刻。
- `training-evidence-v1.*`、`application-evidence-v1.*`、`heldout-evidence-v1.*`：原始证据和逐文件 SHA256。
- `validation-report.json`、`heldout-ranking.json`、`application-comparison.json`：原生审计与对照。
- `frozen-source-audit.json`、`tests-checkpoint.*`、`software-reuse-expanded/report.json`：源码一致性和软件验证。
- `audit_results.py`：重放归档及数值/映射/profile 审计，不拟合或修改模型。

冻结标识前缀：设计 `9c826acaaaa5569b`，模型 `4cfb3c905740fe37`，
预测锁 `d8ad7385f08cf5f1`。训练/应用/留出归档 SHA256 前缀分别为
`99a64ad4fff1ff4f` / `5270ca8650468572` / `5e749861fc0878ea`。
原模型的证据绝对路径也纳入摘要；移动归档后应重新拟合并另存新模型，
原冻结锁保留为此次事先预测的历史证据。

## 仍需补齐

1. 先诊断标量的编译行为变化，加入编译特征和分段成本域，并预先划分
   新 TRAIN/HO；扩展 M/N/K 及新的留出形状，拒绝跨未知编译模式插值。
2. 从编译产物恢复真实 buffer、锁和通道，将独立传输、acquire/release、
   启动和跨帧重叠分别测量与绑定，替换当前名义传输和合并残差。
3. 扩展多算子/多核成本组合和实现族，改进语义、实现和部署族的预算
   分配；保持未知候选、参考候选和编译失败证据。
4. 系统文献覆盖、新 SSA/e-graph 统一、完整 ORCA、更大真实应用和板卡
   持续分别验收。一个 K 形状的留出验证不足以证明跨应用排名可靠性。
