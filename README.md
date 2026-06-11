# GenAI GPU Power Profile Analysis

[![Dataset](https://img.shields.io/badge/Dataset-arXiv%3A2604.07345-blue)](https://arxiv.org/abs/2604.07345)
[![GPU](https://img.shields.io/badge/GPU-NVIDIA%20H100-green)](https://www.nvidia.com/en-us/data-center/h100/)
[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

A comprehensive power profile analysis of NVIDIA H100 GPUs running training, fine-tuning, and inference workloads, based on real hardware measurements from **Vercellino et al. (2026)**. Aimed at power systems engineers and data center infrastructure planners.

> **Original dataset:**  
> Vercellino et al., "Measurement of Generative AI Workload Power Profiles for Whole-Facility Data Center Infrastructure Planning," [arXiv:2604.07345](https://arxiv.org/abs/2604.07345) (2026).  
> Data available at: [NLR Data Catalog](https://data.nlr.gov/submissions/312)

---

## What's in This Repo

```
genai-gpu-power-analysis/
│
├── full_analysis.py              # Main analysis script (all 8 phases)
├── generate_chinese_report.py    # Chinese HTML report generator
├── requirements.txt              # Python dependencies
│
├── 01_aggregated_datasets/       # Aggregation scripts and job metadata
│   ├── training/
│   │   ├── metadata.csv          # Job metadata (model, nodes, repeat index)
│   │   └── postprocess.py        # Raw → Parquet aggregation script
│   ├── inference_offline_llama3_70b/
│   ├── inference_online_finite_llama3_70b/
│   └── inference_online_rate_llama3_70b/
│
├── 02_analysis_scripts/          # Official Jupyter notebooks from the paper
│
├── 03_whole-facility_profiles/   # Whole-facility simulation (DIPLOEE model)
│
└── results/                      # Output from full_analysis.py
    ├── report_chinese.html       # Full Chinese HTML report (all figures embedded)
    ├── executive_summary_chinese.md
    ├── figures/                  # Power distribution, timescale, DC scale-up plots
    ├── tables/                   # Summary statistics (CSV)
    ├── checkpoint_analysis/      # Detected checkpoint events + plots
    ├── fft_analysis/             # Power spectral density plots + peak frequencies
    └── training_vs_inference/    # Cross-workload comparison
```

> **Note:** Raw data (~1.92 GB) and aggregated Parquet files (~558 MB) are excluded via `.gitignore`. Download them from the NLR Data Catalog link above and run the postprocess scripts to regenerate.

---

## Analysis Overview

The script `full_analysis.py` runs 8 analysis phases end-to-end:

| Phase | Description | Key Output |
|-------|-------------|------------|
| **1 – Exploration** | File structure scan, metadata summary | Dataset tree, workload classification |
| **2 – Statistics** | Per-workload power statistics | Mean / Max / P99 / CV comparison table |
| **3 – Timescales** | Multi-resolution analysis (0.2 s → 60 s) | Ramp rates, volatility vs. averaging window |
| **4 – Checkpoints** | Automatic power-drop event detection | 161 LLaMA-2 checkpoint events detected |
| **5 – FFT** | Power spectral density analysis | Dominant frequencies per workload |
| **6 – Comparison** | Training vs. fine-tuning vs. inference | Full metric comparison table |
| **7 – Scale-up** | Datacenter-level power aggregation | 25K / 100K GPU / 1 GW DC scenarios |
| **8 – Summary** | Engineering implications | HTML report + executive summary |

---

## Key Findings

### Power Characteristics (real measurements)
| Workload | Mean Power | CV | Notes |
|----------|-----------|-----|-------|
| LLaMA-2 LoRA training (16 nodes) | ~149 W/GPU | 33.2% | Near H100 TDP (700 W) at peak |
| Stable Diffusion training (16 nodes) | ~140 W/GPU | 29.3% | Deeper checkpoint drops |
| Inference – online rate-limited | ~517 W/node | **2.6%** | Near-ideal base load |
| Inference – offline batch | ~706 W/node | 7.1% | High and stable |

### Checkpoint Behavior
- **LLaMA-2 LoRA**: checkpoint every **~13 s**, lasting **5–7 s**, power drops **15–17%** below baseline — 161 events detected across a single 16-node run
- **Stable Diffusion**: only 5 events per run, but each lasts **~40 s** with a **64–67%** power drop (EMA dual-weight checkpointing)

### Frequency Domain
- Training power has discrete spectral peaks at **10.6 s** (AllReduce communication cycle) and **3.5 s** (forward/backward pass boundary)
- Inference power shows a broad continuous spectrum — driven by stochastic request arrivals

### Datacenter Scale-up
| Scenario | Synchronization | Mean | Peak | Peak-to-Average |
|----------|----------------|------|------|----------------|
| 25,000 H100 | Fully sync | 13.1 MW | **18.9 MW** | 1.44 |
| 100,000 H100 | Fully sync | 52.4 MW | **75.6 MW** | 1.44 |
| **1 GW DC** | **Fully sync** | 877 MW | **1,266 MW** | **1.44** |
| 1 GW DC | Fully random | 877 MW | 879 MW | **1.00** |

> Job staggering (de-synchronizing training steps across jobs) is the most effective software-level measure to reduce peak-to-average ratio from 1.44 → 1.00.

---

## Quick Start

### 1. Download the dataset
```
https://data.nlr.gov/submissions/312
Place the extracted folders under the repo root.
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
pip install pyarrow        # required for Parquet support
```

### 3. Generate aggregated Parquet files
```bash
cd 01_aggregated_datasets/training                      && python postprocess.py
cd ../inference_offline_llama3_70b                      && python postprocess.py
cd ../inference_online_finite_llama3_70b                && python postprocess.py
cd ../inference_online_rate_llama3_70b                  && python postprocess.py
```

### 4. Run the full analysis
```bash
python full_analysis.py
# Outputs written to results/
```

### 5. Generate the Chinese HTML report
```bash
python generate_chinese_report.py
# Output: results/report_chinese.html  (self-contained, ~2.8 MB)
```

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| pandas | ≥ 2.3 | Data processing |
| numpy | ≥ 2.2 | Numerical computation |
| scipy | ≥ 1.16 | FFT, signal processing |
| matplotlib | ≥ 3.10 | Visualization |
| pyarrow | ≥ 24 | Parquet I/O |

---

## Citation

If you use this analysis, please cite the original dataset paper:

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

Analysis code in this repository is released under the [MIT License](LICENSE).  
The original dataset is owned by Vercellino et al. and NLR — please comply with their terms of use.
