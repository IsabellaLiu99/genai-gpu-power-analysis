# GenAI GPU 功率特征分析

[![数据来源](https://img.shields.io/badge/数据来源-arXiv%3A2604.07345-blue)](https://arxiv.org/abs/2604.07345)
[![GPU](https://img.shields.io/badge/GPU-NVIDIA%20H100-green)](https://www.nvidia.com/en-us/data-center/h100/)
[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

基于 **Vercellino et al. (2026)** 真实测量数据集的全面分析，
聚焦 NVIDIA H100 GPU 在训练、微调、推理三类 AI 工作负载下的功率特征，
面向**电力系统规划**与**AI数据中心供电设计**。

> **引用原论文：**  
> Vercellino et al., "Measurement of Generative AI Workload Power Profiles for Whole-Facility Data Center Infrastructure Planning," [arXiv:2604.07345](https://arxiv.org/abs/2604.07345) (2026).  
> 原始数据集：[NLR Data Catalog](https://data.nlr.gov/submissions/312)

---

## 项目结构

```
genai-gpu-power-analysis/
│
├── full_analysis.py              # 主分析脚本（八阶段全流程）
├── generate_chinese_report.py    # 中文HTML报告生成器
├── requirements.txt              # Python依赖包
│
├── 01_aggregated_datasets/       # 聚合脚本与元数据（Parquet数据文件需自行生成）
│   ├── training/
│   │   ├── metadata.csv          # 作业元数据（模型、节点数、重复次数）
│   │   └── postprocess.py        # 数据聚合脚本
│   ├── inference_offline_llama3_70b/
│   ├── inference_online_finite_llama3_70b/
│   └── inference_online_rate_llama3_70b/
│
├── 02_analysis_scripts/          # 官方 Jupyter Notebook 分析脚本
│   └── *.ipynb
│
├── 03_whole-facility_profiles/   # 全设施仿真分析（DIPLOEE模型）
│   ├── analysis.ipynb
│   └── *.png                     # 仿真结果图
│
└── results/                      # 本项目分析输出
    ├── report_chinese.html       # 中文专业HTML报告（含所有图表）
    ├── executive_summary_chinese.md
    ├── figures/                  # 功率分布、时间尺度、DC放大图
    ├── tables/                   # 统计数据CSV
    ├── checkpoint_analysis/      # Checkpoint事件检测结果
    ├── fft_analysis/             # 频域分析图与数据
    └── training_vs_inference/    # 训练vs推理对比
```

---

## 分析内容（八个阶段）

| 阶段 | 内容 | 关键输出 |
|------|------|---------|
| **Phase 1** | 数据集结构探索 | 文件树、元数据汇总 |
| **Phase 2** | 基础统计分析 | Mean/Max/P99/CV 对比表 |
| **Phase 3** | 多时间尺度分析 | 0.2s/1s/10s/60s 均值对比；斜率统计 |
| **Phase 4** | Checkpoint事件检测 | 161个LLaMA-2 Checkpoint事件；5个SD深度跌落事件 |
| **Phase 5** | 频域分析（FFT） | 功率谱密度；主频识别（10.6s训练周期） |
| **Phase 6** | 训练vs推理对比 | 均值/CV/P99/斜率全指标对比 |
| **Phase 7** | 数据中心放大效应 | 25K/100K GPU/1GW DC 三场景 x 三种相关性 |
| **Phase 8** | 供电工程意义 | UPS/BESS/电网调频/GW级DC规划建议 |

---

## 核心发现摘要

### 功率特征
- **LLaMA-2 LoRA 训练**：均值 ~150 W/GPU，CV 约 33%，峰值接近 H100 TDP（700 W）
- **推理（速率限制模式）**：CV 仅 **2.6%**，接近理想基荷
- **最大功率斜率**：**26,000+ W/s**（16节点整体），折合单GPU约 400 W/s

### Checkpoint机制
- LLaMA-2 LoRA：每 **~13秒** 出现一次功率谷，持续约 **5~7秒**，跌落 **15~17%**
- Stable Diffusion：功率跌落高达 **64~67%**，持续约 **40秒**（EMA双权重写盘）

### 频域特征
- 训练存在明确周期：**10.6秒**（AllReduce通信节拍）、**3.5秒**（前向/反向传播切换）
- 推理为宽带随机谱，无明显主频

### 数据中心放大

| 场景 | 同步模式 | 峰值 | 峰均比 |
|------|---------|------|-------|
| 25,000 H100 | 完全同步 | 18.9 MW | 1.44 |
| 100,000 H100 | 完全同步 | 75.6 MW | 1.44 |
| **1 GW DC** | **完全同步** | **1,266 MW** | **1.44** |
| 1 GW DC | 完全随机 | 879 MW | **1.00** |

> 作业去同步化（Job Staggering）是将峰均比从 1.44 降至 1.00 的最有效软件手段。

---

## 快速开始

### 1. 下载原始数据集
```
从 NLR Data Catalog 下载（约 1 GB）：
https://data.nlr.gov/submissions/312
```

### 2. 安装依赖
```bash
pip install -r requirements.txt
pip install pyarrow
```

### 3. 生成聚合数据（如未下载预生成版本）
```bash
cd 01_aggregated_datasets/training && python postprocess.py
cd ../inference_offline_llama3_70b   && python postprocess.py
cd ../inference_online_finite_llama3_70b && python postprocess.py
cd ../inference_online_rate_llama3_70b   && python postprocess.py
```

### 4. 运行完整分析
```bash
python full_analysis.py
```

### 5. 生成中文报告
```bash
python generate_chinese_report.py
# 输出：results/report_chinese.html
```

---

## 依赖环境

| 包 | 版本 | 用途 |
|----|------|------|
| pandas | 2.3+ | 数据处理 |
| numpy | 2.2+ | 数值计算 |
| scipy | 1.16+ | FFT / 信号处理 |
| matplotlib | 3.10+ | 可视化 |
| pyarrow | 24+ | Parquet 文件读取 |

---

## 数据来源声明

本项目所有分析基于 Vercellino et al. (2026) 发布的开放数据集，
测量值均来自真实硬件（NVIDIA H100 + WattAMeter）。
规模化推演（Phase 7）为解析计算，含建模假设，仅供参考。

如使用本分析，请同时引用原始数据集论文：

```bibtex
@article{vercellino2026genai,
  title   = {Measurement of Generative AI Workload Power Profiles for
             Whole-Facility Data Center Infrastructure Planning},
  author  = {Vercellino et al.},
  journal = {arXiv preprint arXiv:2604.07345},
  year    = {2026}
}
```

---

## License

本项目分析代码采用 [MIT License](LICENSE)。  
原始数据集版权归 Vercellino et al. 及 NLR 所有，请遵守其使用条款。
