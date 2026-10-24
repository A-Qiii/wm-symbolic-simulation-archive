from common import *
import csv,shutil
def main():
 a=read(R/'AUDIT.json');g=read(R/'supported_start_geometry.json');protocol=read(E/'configs/protocol.json');runs=read(E/'results/runs.json');source=E/'source_evidence/pybullet327_source';C=source/'examples/SharedMemory/PhysicsServerCommandProcessor.cpp';S=source/'src/BulletDynamics/Featherstone/btMultiBodyConstraintSolver.cpp';I=source/'Extras/Serialize/BulletWorldImporter/btMultiBodyWorldImporter.cpp';V=source/'examples/SharedMemory/plugins/eglPlugin/eglRendererVisualShapeConverter.cpp';F=source/'src/BulletDynamics/ConstraintSolver/btContactSolverInfo.h';last=g['last']
 report=f'''# 原生恢复语义诊断：有限复现通过，分叉设计可挽救

在当前PyBullet/CALVIN配置、已有首根和24步短前缀范围内，原生检查点恢复后可精确复现轨迹。动作前不同的力矩/约束力是尚未刷新的上一次求解输出；从首个真实物理tick开始，所有被比较物理字段、新产生的力读数、接触、图像和标签均相同。**这解除的是当前已知读数判据的工程疑点，不能视为完整实验或任意场景分叉已通过。** 原失败试采仍按原协议保留失败，未回写成绩。

## 预先冻结的有限设计

只用已有 `fork12_A_00`，没有新布局、数据集扩采或训练。旧 `.bullet` 档案载入相同物体/约束后，恢复Python/控制器状态和原电机命令，建立一个原生内存检查点S；每段动作均从S恢复。没有手工重置物理关节、施加外力或修改接触。

H为原保存的首个TCP目标/姿态/闭合命令重复24步。L为原lift命令的前24步：先保持12步，再以2.5mm/步抬高目标12步，共30mm。每控制步8个物理tick；每段0.8秒。没有重放原来72步完整lift，不能声称已复验该较长恢复历史。

| 次序/记录名 | 固定动作 | 恢复前历史 | 比较目的 |
| --- | --- | --- | --- |
| H1_reference | H×24 | 档案载入后的S | 基准 |
| H2_after_same_H | H×24 | 刚执行H1 | 没有插入不同动作的重复控制 |
| L1_short_lift | L×24 | H2 | 产生预定短lift历史；前12步也与H相同 |
| H3_after_L1 | H×24 | 短lift L1 | 检查不同动作历史是否污染同一H后继 |
| L2_short_lift | L×24 | H3 | L的相同动作复现 |
| H4_after_L2 | H×24 | 短lift L2 | 再次验证短lift后的恢复 |

全部6段完成，恰好144控制步、1152执行tick。成功启动有2个原生初始化/复位tick；此前两个零前缀启动各2tick，合计所有启动1158tick。没有补做第7段。冻结判据为每个原生单位上的精确比较，容差0，接触只排序以消除返回顺序差；未按标签结果挑选数值容差。

最终协议SHA256：`{sha(E/'configs/protocol.json')}`。所有原始状态、动作、三阈值标签和同时间双相机RGB/depth/mask保存于本报告对应实验的private目录，仅作为工程审计，不转为训练数据。

## 为什么要区分历史输出与后继状态

| 来源 | 核验发现 | 对判据的影响 |
| --- | --- | --- |
| [官方API指南](https://raw.githubusercontent.com/bulletphysics/bullet3/master/docs/pybullet_quickstartguide.pdf) | motor torque与constraint force描述的是上一仿真步/最近仿真步的结果 | 单看动作前读数不同，不能直接推出后继不同 |
| [3.2.7服务端源码]({C}:8059)；[约束接口]({C}:13045) | 两类读数取约束保存的applied impulse再除以时间步 | 不是当前电机目标；保持原始读数，不把它们填成检查点值 |
| [原生恢复入口]({C}:14837)；[对象/接触恢复]({I}:103) | 恢复已有body姿态、速度与接触manifold；已有电机命令和这些已施加冲量不在这一路径逐项恢复 | Python记忆与13条相关电机命令显式恢复/重发，作为恢复实现 |
| [求解器分支]({S}:1448) | 非接触多体约束只有启用`SOLVER_USE_ARTICULATED_WARMSTARTING`时才用旧冲量热启动；另一分支从0开始，求解后写回新的冲量 | 不能把这些读数在所有配置下都当成无关历史 |
| [默认配置]({F}:100)；[启用接口]({C}:11697) | 对应版本默认solver位不含上述articulated标志；原CALVIN只设置deterministicOverlappingPairs，没有设置该参数 | 源码支持当前配置下“旧读数不进入这一热启动分支”的解释；实测仍是必要证据 |

普通接触的warm starting与这里的多体非接触约束分支不同，不能概括为“所有热启动均关闭”。本次保存/恢复接触并逐项验证。官方PyPI 3.2.7源码包SHA256 `042879db8d101ac7590dee475fc6aded508b85fe1273fdbbfde1d88bd200e14f` 已校验；安装二进制hash、源码和文档留存。版本匹配并非二进制构建完全可证明的凭据，因此结论同时依靠真实复现，而不是仅凭源码推断。

电机重发覆盖机械臂7关节、两个独立手指关节9/11和4个桌面装置关节，共13条原保存命令。每次都记录native restore前、native restore后、重发后状态；6次重发均不改变任何读取的物理字段，Python/电机记忆均精确恢复。原接口说明电机命令由下一仿真步处理，重发并非重置关节姿态。

## 实测比较

| 检查 | 配对数量 | 结果 | 实际意义 |
| --- | ---: | --- | --- |
| H2/H3/H4对H1，各24步；L2对L1，24步；L1/L2对H1，共同前12步 | 120控制步配对 | 完整物理字段精确一致，非零差异0 | 相同动作的受控后继不依赖所测试的恢复历史 |
| 上述每组比较的首控制步，内部8个物理tick | 48物理tick配对 | 首tick起全部物理字段与标签精确一致 | 不是等待8tick后才发现趋同，也没有额外预热步 |
| 对应动作前/每控制步/首物理tick的双相机RGB、depth、mask | 792数组配对 | 逐元素差异0 | 同时覆盖恢复起点与执行观察；渲染不推进物理 |
| 旧档案与重新载入的起点图像 | 6数组 | 逐元素差异0 | 跨进程桥接的视觉起点一致 |
| 全部150个控制索引状态，3个接触阈值 | 150组独立标签复核 | 与原保存标签一致 | Holding/桌面支撑未被条件名或期望结果替代 |
| 原生恢复前后body/两手指/接触/配置，及Python记忆 | 6次 | 除分类保存的历史输出外精确一致 | 没有只保存夹爪总宽度或忽略左右手指速度 |

从第1物理tick至24控制步末尾，所有比较中的完整物理记录——包含两侧手指、关节反力、已施加电机力矩、gear返回量、姿态、速度、接触和同进程图形元数据——都完全相同；动作、控制器状态、历史与标签也相同。没有发现轨迹差异起点。

动作前历史读数仍不同。例如H2对H1最大机械臂力矩差18.1803N·m，双指电机力差200.1112N；H3对H1最大机械臂力矩差15.3000N·m。H1动作前的历史输出来自新client初始化，故这些数值不是原失败试采中2.00749N·m差值的再次测量，也不代表瞬间施加了额外力。没有修改这些读数；下一次相同真实动作的第1物理tick便产生完全相同的新读数。

## 原起点为什么已经Holding

这里完全读取原试采已有的准备轨迹，没有重新抓取或改变标签。选定原生红块URDF是尺寸0.07×0.05×0.05m、质量1kg，场景缩放0.8，实际盒体0.056×0.040×0.040m。最低点按原生姿态旋转盒体求得；桌面高度来自闭合前支撑接触点z={g['tabletop_z_m_from_contact_points']:.12f}m。

| 时刻 | 实际观察 | 原因或意义 |
| --- | --- | --- |
| 控制200，下降结束、夹爪仍开 | 红块中心z={g['before_close_center_z_m']:.9f}m，桌面竖直支撑9.8N，Holding=false | 这一时刻仍是受支撑起态 |
| 控制201，闭合第1步 | 双侧指垫开始接触，短时接触/支撑力变化 | 闭合本身是物理操作，不保证物体位置不变 |
| 控制202，闭合第2步 | 桌面力变为0，双侧指垫接触，首次Holding=true | 支撑转移发生在准备阶段，早于命名为lift的动作 |
| 控制220，20步闭合结束/保存分叉起点 | 中心上移{last['center_rise_since_close_start_mm']:.6f}mm；盒体最低点高于桌面{last['box_bottom_clearance_above_table_mm']:.6f}mm | 存在实际亚毫米间隙，并非只因接触阈值改动 |
| 同一保存时刻的受力 | 左/右指垫向上接触分量{last['leftpad_vertical_contact_force_N']:.6f}/{last['rightpad_vertical_contact_force_N']:.6f}N，合计约9.8N；其中摩擦竖直分量{last['pad_friction_vertical_sum_N']:.6f}N | 指垫承担物体重量，桌面不再承重；三阈值与稳定窗口均支持Holding |

原控制器`hold`取当前实际TCP作为闭合目标，而不是继续用前一请求的grasp目标。本根计划grasp z={g['planned_grasp_z_m']:.9f}m，开始闭合时实际TCP/保持目标z={g['close_hold_actual_TCP_z_m']:.9f}m，高{1000*(g['close_hold_actual_TCP_z_m']-g['planned_grasp_z_m']):.6f}mm。固定200N闭合力、接触几何和摩擦共同作用下，原轨迹在闭合阶段已发生微小抬升。位置与受力证据支持这个过程解释，但没有单项干预来把抬升唯一归因于某个几何参数或IK误差。没有声称这些亚毫米变化在当前图像中一定可辨。

所以“准备完成”不能被预设为“仍受桌面支撑”；未来若需要受支撑/已夹持起态的对照，应另行冻结状态资格检查，保留不满足条件的案例。这里不启动新试采、不重命名标签，也不使用本根声称状态与动作联合预测的增益。

## 图例与实际意义

![恢复语义和原始起点受力](figures/01_restore_semantics_and_start.png)

A/B/C横轴为首次真实动作后的物理tick，0为动作前，1为1/240秒后。蓝线代表重复H后恢复，橙线代表第一次短L后恢复，绿线代表第二次短L后恢复，均与H1的同tick值比较；重合线会互相遮挡。纵轴分别为机械臂力矩最大差N·m、双指电机力最大差N、gear原生impulse÷dt返回量差。所有曲线从tick1起为0，说明旧读数被同一步求解刷新；不能从step0差值推断轨迹差异。

D灰线是协议中6组配对的全部body位置最大绝对差，横轴为控制步；各线完全重合在0。纵轴±1e−12m只是为展示零线而设置的显示范围，**不是接受容差**。完整姿态、速度、接触和力读数也精确一致，见CSV，不只比较位置。

E/F读取原来准备轨迹。横轴闭合控制步0对应控制200；紫色竖虚线标识第2步首次Holding。E纵轴是作用于红块的竖直接触力，向上为正，蓝/橙/绿对应桌面/左指垫10/右指垫12；灰色横虚线是物体重力9.8N，不能把某一条指垫线与它单独比较，应看两指合力。F蓝线为块中心相对闭合前的上移，橙线为原生盒体最低点到桌面间隙，单位mm。图中0前后短暂负间隙是接触求解阶段的几何穿透；未把它删除。

## 两个未执行前缀的启动失败也保留

1. `native_restore_semantics_root1_20261018`：分类记录函数对PyBullet的tuple容器执行列表赋值，动作前报错；修正为规范容器转换，检查了数值与输入保真，无物理参数或判据改变。
2. `native_restore_semantics_root1_fix01_20261018`：跨进程档案桥接遇到右侧指垫visualShape的5个无效元数据值：几何类型、3维尺寸和文件名字节。所有演化状态和6组相机数组一致，但该次冻结规则仍停止。没有覆盖此失败。

[对应EGL源码]({V}:1173)声明局部visualShape而未全初始化；缓存命中时跳过初始化几何类型/尺寸/文件名的转换函数，再保存该结构。原生URDF的左右指垫复用同一STL，右指垫缓存条目出现非合法几何枚举与非UTF8文件名字节，与这一路径吻合。最终协议只在跨进程桥接中把body0/link12该条目上述字段标为“API未定义”，且要求两边类型均非法；原值完整保留，并核验URDF/STL hash、所有其他字段和图像。**同进程S恢复及逐步比较没有排除任何visualShape字段。** 这不是放宽动力学阈值，也没有修改渲染器或关闭缓存。

这两个启动均0前缀/0控制步，仅各有2个原生初始化tick。三份协议、日志、失败结果分别保留；最终诊断的源码在执行前冻结，之后无变化。

## 可挽救范围与剩余边界

原生分叉设计可继续采用，但应按接口语义处理门槛：起点严格核验姿态、速度、两指状态、接触、控制器/电机命令及观察；历史已施加力读数另存为历史输出，在真实匹配动作的首个物理步及后续轨迹中核验新读数。不能添加预热步来隐藏初始分岔，不能把读取到的任意差异都归为历史量。若启用articulated warm starting、改变求解器/电机/对象配置，须重新核验相关状态完整性。

本次只证明固定首根、当前二进制与配置、两段预先固定短动作和这些恢复历史的有限精确复现。没有证明任意根、72步完整lift历史、长时稳定性或多技能动作因果效果。原起点已Holding的问题仍限制后续状态条件数据设计。本轮不启动新数据采集/模型训练，不晋级控制器，也不解除原指南720→约6000、多技能、TS/LTLf及独立闭环验收要求。

旧失败试采及更早实验、两个零前缀启动共{a['old_files_verified']:,}个文件hash已验证不变。最终目录：`{E}`。

主要证据：FROZEN_PROTOCOL.json、AUDIT.json、control_comparisons.csv、first_physics_ticks.csv、image_array_comparisons.csv、readback_refresh_by_physics_tick.csv、labels_per_control.csv、original_preparation_stage_analysis.csv、supported_start_geometry.json、PNG/PDF图；原始private状态与当前版本源码在实验目录。报告中的比较次数是相关重复核验，不能当作独立统计样本数。
'''
 (R/'REPORT.md').write_text(report)
 shutil.copy2(__file__,E/'source/write_report.py')
 print('REPORT',R/'REPORT.md')
if __name__=='__main__':main()
