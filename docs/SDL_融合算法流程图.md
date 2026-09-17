# SDL 融合技术的核心算法动画 (Mermaid)

## 1. 高层流程（简化视图）

```mermaid
graph TD
    A["📊 原始数据<br/>SinoScope1.0 + FWEA23"] --> B["🔍 Phase 1: Patch提取<br/>DirectPatchExtractor"]
    B --> C["📈 Phase 2: 字典学习<br/>D₁, D₂, C训练"]
    C --> D["✨ Phase 3: 完整变换<br/>全域增强"]
    D --> E["🎯 输出: 增强SinoScope<br/>高分辨率特征 + 全覆盖"]
    
    style A fill:#e1f5ff
    style B fill:#fff3e0
    style C fill:#f3e5f5
    style D fill:#e8f5e9
    style E fill:#c8e6c9
```

## 2. Patch 提取细节（Phase 1）

```mermaid
graph TD
    A["SinoScope1.0<br/>1°×1°网格"] --> B["提取patch<br/>6°×6° = 6×6格点"]
    C["FWEA23<br/>0.25°×0.25°网格"] --> D["提取patch<br/>6°×6° = 24×24格点"]
    
    B --> E["Patch₁<br/>L₁=36"]
    D --> F["Patch₂<br/>L₂=576"]
    
    E --> G["配对<br/>同一物理位置"]
    F --> G
    
    G --> H["计算CC<br/>Cross-Correlation"]
    H --> I{CC > 0?}
    I -->|是| J["✓ 保留配对<br/>~8000对"]
    I -->|否| K["✗ 排除异常<br/>~2000对"]
    
    J --> L["分割<br/>训练:验证 = 88:12"]
    
    style A fill:#bbdefb
    style C fill:#c8e6c9
    style E fill:#fff9c4
    style F fill:#fff9c4
    style J fill:#c8e6c9
    style K fill:#ffccbc
    style L fill:#f0f4c3
```

## 3. 字典学习详解（Phase 2）

```mermaid
graph TD
    A["训练集<br/>7000对patch"] --> B["DictionaryLearning"]
    
    B --> C["编码阶段<br/>OMP算法"]
    B --> D["原子学习阶段"]
    
    C --> E["稀疏系数C<br/>7000×20<br/>大部分为0"]
    D --> F["低分辨率字典D₁<br/>20×36"]
    
    E --> G["最小二乘<br/>D₂ = P₂ @ C⁺"]
    F --> G
    
    G --> H["高分辨率字典D₂<br/>20×576"]
    
    E --> I["验证集评估"]
    H --> I
    
    I --> J["验证误差<br/>~8-10%"]
    
    style A fill:#fff9c4
    style C fill:#f3e5f5
    style D fill:#f3e5f5
    style E fill:#ce93d8
    style F fill:#ce93d8
    style H fill:#ce93d8
    style J fill:#c8e6c9
```

## 4. 核心约束：共享稀疏系数

```mermaid
graph TD
    A["物理位置lat,lon"] --> B["Patch对<br/>低分辨率×高分辨率"]
    
    B --> C["低分辨率patch<br/>36个值"]
    B --> D["高分辨率patch<br/>576个值"]
    
    C --> E["编码"]
    D --> F["编码"]
    
    E --> G["稀疏系数c<br/>20个系数<br/>其中5个非零"]
    F --> G
    
    G --> H["约束<br/>同一c编码<br/>两种分辨率<br/>不同原子"]
    
    C --> I["D₁原子集<br/>20个特征模式<br/>低分辨率"]
    D --> J["D₂原子集<br/>20个特征模式<br/>高分辨率"]
    
    H -.-> I
    H -.-> J
    
    style G fill:#ff9800,stroke:#e65100,stroke-width:3px
    style H fill:#ff9800,stroke:#e65100,stroke-width:3px
    style I fill:#ce93d8
    style J fill:#ce93d8
```

## 5. 完整变换流程（Phase 3：核心！）

```mermaid
graph LR
    A["SinoScope原始<br/>patch₁<br/>36个值"] -->|Step1| B["编码<br/>OMP]"]
    B -->|Step2| C["稀疏系数c<br/>20维"]
    C -->|Step3| D["解码<br/>D₂ @ c"]
    D -->|Step4| E["高分辨率<br/>patch预测<br/>576个值"]
    E -->|Step5| F["下采样"]
    F -->|Step6| G["增强patch<br/>36个值<br/>FWEA23风格"]
    G -->|Step7| H["拼接<br/>Gaussian加权"]
    H --> I["✨ 增强的<br/>SinoScope"]
    
    style A fill:#bbdefb
    style B fill:#e1bee7
    style C fill:#ffe0b2
    style D fill:#c8e6c9
    style E fill:#ffccbc
    style F fill:#fff9c4
    style G fill:#b3e5fc
    style I fill:#c8e6c9,stroke:#2e7d32,stroke-width:3px
```

## 6. 为什么能全域增强？

```mermaid
graph TD
    A["FWEA23覆盖区<br/>~1000个patch对"]
    B["学到D₁↔D₂<br/>变换规律"]
    C["SinoScope<br/>覆盖区内"]
    D["SinoScope<br/>覆盖区外<br/>FWEA23无数据"]
    
    A --> B
    B --> E["应用变换<br/>c = OMP_encode<br/>p̂ = D₂ @ c"]
    
    C --> E
    D --> E
    
    E --> F["✅ 全域增强完成<br/>包括FWEA23无覆盖区"]
    
    style A fill:#fff9c4
    style B fill:#ff9800,stroke:#e65100,stroke-width:2px
    style D fill:#ffccbc
    style F fill:#c8e6c9,stroke:#2e7d32,stroke-width:3px
```

## 7. 误差与改善

```mermaid
graph TD
    A["Baseline误差<br/>简单插值"] --> B["相对误差"]
    C["SDL变换"] --> D["相对误差"]
    
    B --> E["11.6%"]
    D --> F["7.6%"]
    
    E --> G["改善<br/>34%"]
    F --> G
    
    H["验证集误差<br/>分布"] --> I["正态分布<br/>μ=0.4%<br/>σ=3%"]
    
    style E fill:#ffccbc
    style F fill:#c8e6c9
    style G fill:#81c784,stroke:#2e7d32,stroke-width:2px
    style I fill:#c8e6c9
```

## 8. 与论文场景对比

```mermaid
graph LR
    subgraph "论文实验"
        A1["CVM-S4.26<br/>0.03°"]
        B1["Z2015<br/>0.01°"]
        C1["比例: 3:1"]
    end
    
    subgraph "东亚配置"
        A2["SinoScope<br/>1°"]
        B2["FWEA23<br/>0.25°"]
        C2["比例: 4:1"]
    end
    
    A1 -.-> A2
    B1 -.-> B2
    C1 -.-> C2
    
    D["✓ 相似性好<br/>✓ 参数可迁移<br/>✓ 结果可参考"]
    
    style D fill:#c8e6c9,stroke:#2e7d32,stroke-width:2px
```

## 9. 关键决策树

```mermaid
graph TD
    Start{选择融合<br/>策略} 
    
    Start -->|无条件融合| A["❌ 简单嵌入<br/>ad-hoc参数<br/>信息丢失<br/>仅重叠区"]
    
    Start -->|数据驱动| B["✅ SDL融合<br/>自动学习<br/>信息融合<br/>全域增强"]
    
    B --> C{需要验证<br/>效果?}
    
    C -->|快速测试| D["运行 Phase 1.5<br/>默认参数K=20, λ=0.1<br/>~1.4h/51层"]
    
    C -->|系统优化| E["运行 Phase H<br/>--grid-search<br/>K×λ网格搜索<br/>~4h/51层"]
    
    D --> F["输出增强<br/>SinoScope"]
    E --> G["保存<br/>optimal_config.json"]
    G --> F
    
    style B fill:#c8e6c9,stroke:#2e7d32,stroke-width:2px
    style F fill:#81c784,stroke:#2e7d32,stroke-width:2px
```

## 10. 参数敏感性

```mermaid
graph LR
    A["K"]
    B["λ"]
    C["CC_threshold"]
    
    A -->|↑| D["特征能力↑<br/>但过度拟合↑"]
    B -->|↑| E["稀疏性↑<br/>但误差↑"]
    C -->|↑| F["样本质量↑<br/>但样本数↓"]
    
    D --> G["平衡点<br/>K=20"]
    E --> G
    F --> H["最优选择<br/>λ=0.1"]
    
    style G fill:#fff9c4,stroke:#f57f17
    style H fill:#fff9c4,stroke:#f57f17
```
