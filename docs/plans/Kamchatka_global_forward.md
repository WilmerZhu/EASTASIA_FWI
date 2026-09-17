
## 项目背景
- 基于SPECFEM3D_GLOBAL进行全球尺度地震波传播数值模拟
- 目标：堪察加地震的全球波形正演计算
- 运行环境：中科曙光高性能计算服务器
- 作业调度系统：SLURM（已从PBS迁移）

## 技术栈和依赖
- SPECFEM3D_GLOBAL (最新版本)
- MPI并行计算
- SLURM作业调度系统
- Fortran/C编译环境
- 全球地球模型（如S362ANI）

## 已完成工作
1. 参考global_s362ani_shakemovie示例
2. 9 个堪察加地震的CMT源参数文件，先测试一个，`CMTSOLUTION`
3. 台站分布STATION文件, `STATION`

## 当前需求
- 你是地球物理学-地震学领域的专家，请调用你单次回答的最大算力与 token 上限，但是也不要使得代码过于复杂。
- 严格遵循地震学和全波形反演流程科学标准和最佳实践，用中文回答问题
-请帮我生成/优化以下组件的代码：

### 1. SLURM作业提交脚本
`run_kamchatka_simulation.sh`
`go_kamchatka_mesher_slurm.bash`
`go_kamchatka_mesher_slurm.bash`
- 适配中科曙光服务器的资源配置
- MPI并行作业设置
- 内存和时间限制优化
- 错误处理和日志记录

### 2. Par_file配置
- 参考`/path/to/specfem3d_global/EXAMPLES/global_s362ani_shakemovie/Par_file`
- 全球网格划分参数
- 堪察加震源区域的网格细化
- 时间步长和总计算时间设置
- 输出控制参数

## 参考示例路径
- `/path/to/specfem3d_global/EXAMPLES/global_s362ani_shakemovie/` (全球模拟参考)

## 预期输出格式
- 完整的SLURM脚本文件 (.slurm)
- 配置文件模板 (Par_file, CMT, STATION)
- Shell脚本用于自动化流程
- 必要的说明文档和注释


请基于以上信息生成相应的代码和配置文件。