# EASTASIA-FWI 智能融合技术路线图与完整开发设计文档

> 文档类型: 主开发文档（SDL 并行深度学习融合）
> 创建日期: 2026-06-11
> 文档版本: v1.0
> 对标文档: .github/instructions/SDL_Development_Design.md
> 目标代码目录: 4_Fusion/ai_fusion/

---

## 1. 项目目标与边界

### 1.1 总目标
在保持 SDL 可解释性优势的前提下，引入 PyTorch 深度学习融合模型，构建双路线融合体系：
- 路线 A: SDL 可解释融合（现有主线）
- 路线 B: 深度学习融合（新增主线）
- 路线 C: SDL + 深度学习混合融合（目标路线）

### 1.2 核心科学目标
1. 在重叠区学习高分辨率结构特征迁移。
2. 在非重叠区实现可控外推增强。
3. 相比 baseline 上采样，显著降低误差并保留结构边缘。
4. 相比纯 SDL，减少过度平滑并提升局部细节恢复。

### 1.3 约束条件
1. 必须遵循 config/base_config.py 的区域定义。
2. 必须支持当前主数据链路（SinoScope1.0, FWEA23, EARA2024, USTClitho2.0, CSES_VM1.0）。
3. 必须保留 SDL 作为强基线与可解释对照。
4. 深度学习结果必须提供不确定性与失败案例分析。

---

## 2. 总体技术路线图

## 2.1 双轨并行 + 混合收敛

Stage 0: 基线与协议统一
- 固化 SDL 基线与评估协议。
- 统一数据切片、深度匹配、训练验证划分。

Stage 1: 轻量监督融合网络
- 先落地 ResUNet 融合模型。
- 完成与 SDL 的同口径对比。

Stage 2: 结构增强与物理约束
- 引入梯度损失、频谱损失、地学约束损失。
- 做权重消融和鲁棒性评估。

Stage 3: 混合融合
- SDL 输出作为深度模型先验输入。
- 深度模型学习 SDL 残差修正。

Stage 4: 生产化
- 全深度自动推理、NetCDF 输出、可视化报告、模型版本管理。

## 2.2 三条可执行路线

路线 A（立即可用）
- SDL 优化拼接策略（average, gaussian, feather, copy）。
- 成本最低，快速形成强基线。

路线 B（新增主线）
- PyTorch ResUNet 融合，监督训练，2D 切片先行。
- 成本中等，细节恢复能力更强。

路线 C（推荐目标）
- SDL 先融合 + DL 残差校正。
- 在可解释性与分辨率恢复之间取得平衡。

---

## 3. 开发阶段与里程碑

## 3.1 Phase A: 数据与实验协议标准化（1 周）

目标
- 建立统一实验协议，避免不同方法评估口径不一致。

交付
1. 统一样本协议
- 训练集: 重叠区域 patch 对。
- 验证集: 重叠区域留出集。
- 测试集: 深度外推 + 空间外推双测试。

2. 统一指标协议
- 误差指标: RMSE, MAE, 相对误差。
- 结构指标: 梯度相关、边缘保持率、频谱一致性。
- 地学指标: 深度连续性、Vp/Vs 一致性。

3. 统一输出协议
- NetCDF 命名、元数据字段、结果图模板。

验收
- SDL 在协议下可稳定复现。
- 所有方法可在同一脚本下自动评估。

## 3.2 Phase B: 深度学习最小可用版本（2 周）

目标
- 建立可运行、可复现、可对比的 PyTorch 融合基线。

模型
- ResUNet 2D（输入 lowres 上采样切片，输出 highres 风格切片）。

输入设计
- 通道 1: lowres upsampled。
- 通道 2: overlap mask。
- 通道 3: depth normalized。
- 可选通道: SDL 输出。

损失函数
- L_total = w1 L1 + w2 L_grad + w3 L_fft。
- 初始建议: w1=1.0, w2=0.2, w3=0.1。

训练策略
- patch 训练，整图验证。
- 深度独立训练与深度共享训练两套实验。

验收
- 在测试深度上显著优于 bicubic baseline。
- 至少不劣于 SDL 在误差与结构指标上的综合分数。

## 3.3 Phase C: 物理约束增强（2 周）

目标
- 降低伪影，提高地学可信度。

新增约束
1. 深度平滑一致性损失。
2. Vp/Vs 耦合约束（联合训练场景）。
3. 非重叠区置信度衰减约束。

新增输出
- uncertainty map。
- OOD 风险热图。

验收
- 误差分布均值接近 0。
- 非重叠区无系统性偏差漂移。

## 3.4 Phase D: SDL + DL 混合融合（2 周）

目标
- 结合 SDL 可解释先验和 DL 细节恢复能力。

方法 1: SDL 先验输入
- 网络输入增加 SDL 变换结果通道。

方法 2: SDL 残差学习
- 网络学习 target - SDL_output 残差。

方法 3: 双分支融合
- 分支 A 学低频（平滑主体）。
- 分支 B 学高频（边缘细节）。
- 门控融合输出。

验收
- 综合评分优于纯 SDL 与纯 DL。
- 在多个目标模型上稳定提升。

## 3.5 Phase E: 工程化与发布（1 周）

目标
- 接入 EASTASIA-FWI 主流程并形成可发布版本。

交付
1. 统一命令行入口。
2. 自动化训练日志与模型注册。
3. 全深度批处理与 NetCDF 输出。
4. 可视化总报告（论文图风格）。

---

## 4. 代码架构设计

## 4.1 目录结构

4_Fusion/ai_fusion/
- __init__.py
- configs.py
- datasets.py
- models/
  - resunet2d.py
  - swin_unet2d.py
  - hybrid_sdl_residual.py
- losses.py
- metrics.py
- trainer.py
- infer.py
- export_netcdf.py
- experiments/
  - exp_resunet_baseline.py
  - exp_hybrid_sdl_dl.py

## 4.2 配置体系

全局配置
- 继续使用 config/base_config.py 作为区域和路径权威。

模块配置
- 新增 AIFusionConfig（参考 SDLConfigV31 风格）。

关键参数
- patch_size_deg, stride_deg
- input_channels
- loss_weights
- depth_mode（independent 或 shared）
- fusion_mode（dl_only, sdl_prior, sdl_residual）

## 4.3 数据接口

统一数据类
- FusionSample
  - x_lowres
  - y_highres
  - mask_overlap
  - depth_km
  - latlon_meta

统一加载器
- 支持 SinoScope, FWEA23, EARA2024, USTClitho, CSES。
- 支持深度精确匹配策略。

---

## 5. 模型设计

## 5.1 ResUNet 基线

输入
- B, C, H, W

编码器
- 4 层下采样，残差块。

解码器
- 4 层上采样，跳连。

输出
- 单通道速度值回归。

特点
- 参数量适中。
- 对小样本稳定。

## 5.2 Swin-UNet 进阶

适用
- 数据量更大、对长程依赖更敏感时。

风险
- 训练成本与调参成本更高。

## 5.3 SDL 混合模型

方案 A: SDL Prior Net
- 输入拼接 lowres + sdl_output + mask。

方案 B: Residual-on-SDL
- 输出学习 residual = target - sdl_output。
- final = sdl_output + residual_pred。

推荐
- 先做 Residual-on-SDL，收敛更快，解释更清晰。

---

## 6. 损失函数与训练目标

基础损失
1. L1 回归损失。
2. 梯度损失（Sobel/Scharr）。
3. 频谱损失（FFT 振幅差）。

物理约束损失
1. 深度连续性损失。
2. Vp/Vs 比例约束损失（联合训练时）。
3. 非重叠区平稳约束损失。

总损失
L = a L1 + b Lgrad + c Lfft + d Lphys

初始权重建议
- a=1.0, b=0.2, c=0.1, d=0.05

---

## 7. 评估体系

## 7.1 定量指标

误差类
- RMSE, MAE, Relative Error。

结构类
- Gradient Correlation。
- Edge Preservation Ratio。
- 高频能量恢复比。

分布类
- Mean bias, Std, P5/P95。

地学类
- 深度连续性评分。
- Vp/Vs 合理性评分。

## 7.2 对照组设计

必须包含
1. Bicubic baseline。
2. SDL 最优版本。
3. DL only。
4. SDL + DL hybrid。

实验矩阵
- 不同深度。
- 不同目标模型（FWEA23 或 EARA2024）。
- 不同拼接与损失策略。

---

## 8. 实验计划与算力预算

## 8.1 实验顺序

第一批（低成本）
1. ResUNet + L1。
2. ResUNet + L1 + Gradient。
3. Residual-on-SDL + L1 + Gradient。

第二批（中成本）
1. 加 FFT loss。
2. 加物理约束。

第三批（高成本）
1. Swin-UNet 或 GAN/扩散对照。

## 8.2 算力建议

本地单卡
- 可完成第一批和部分第二批。

多卡或服务器
- 推荐用于全深度训练和大规模消融。

时间估计
- 第一批: 2-3 天。
- 第二批: 3-5 天。
- 第三批: 1-2 周。

---

## 9. 版本与里程碑

v0.1
- 完成数据与协议标准化。

v0.2
- 完成 ResUNet 基线。

v0.3
- 完成 SDL + DL 混合残差模型。

v0.4
- 完成全深度导出与总报告。

v1.0
- 形成生产可用融合流水线。

---

## 10. 风险与应对

风险 1: 样本不足导致过拟合
- 应对: patch 增广，深度共享训练，正则化。

风险 2: 非重叠区外推失真
- 应对: 置信度图与约束损失。

风险 3: 指标改进但地学不可信
- 应对: 加入地学约束和人工审查流程。

风险 4: 工程复杂度失控
- 应对: 严格按阶段门控推进，不跨阶段堆功能。

---

## 11. 与现有 SDL 文档的对接说明

1. 本文档不替代 SDL 文档，而是新增 AI 融合主线。
2. SDL 文档继续作为路线 A 的权威实现说明。
3. 本文档重点覆盖路线 B 和路线 C。
4. 评估协议与输出规范保持统一，确保横向可比。

---

## 12. 下一步执行清单（立即开工）

第 1 天
1. 创建 4_Fusion/ai_fusion/ 框架。
2. 落地 AIFusionConfig 和 datasets.py。
3. 跑通单深度 ResUNet 基线训练。

第 2-3 天
1. 接入指标模块 metrics.py。
2. 输出与 SDL 并列的对比报告。
3. 加入 Residual-on-SDL 版本。

第 4-7 天
1. 扩展到多深度批处理。
2. 导出 NetCDF 和图组。
3. 形成阶段评审结论并锁定下一阶段。

---

## 13. 附录: 建议命令行接口

训练
- python 4_Fusion/ai_fusion/trainer.py --model resunet --target fwea --param vs
- python 4_Fusion/ai_fusion/trainer.py --model hybrid_residual --target fwea --param vs --use-sdl-prior

推理
- python 4_Fusion/ai_fusion/infer.py --ckpt path/to/model.ckpt --target fwea --param vs

导出
- python 4_Fusion/ai_fusion/export_netcdf.py --input results/ai_fusion --output output/ai_fusion

对比评估
- python 4_Fusion/ai_fusion/metrics.py --compare sdl,dl,hybrid --target fwea --param vs

---

文档结束。
