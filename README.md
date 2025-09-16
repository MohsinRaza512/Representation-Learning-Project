# Representation-Learning-Project

This repository contains the code, data, and results for my project on **fine-tuning and evaluating the Segment Anything Model (SAM-Med)** for medical image segmentation using the **22-Heart dataset**.  
The work investigates how different prompt types (point, box, both) affect segmentation performance on smaller dataset, measured by **Dice coefficient** and **HD95 boundary distance**.

---

## 📂 Repository Structure

```
Representation-Learning-Project/
│
├── My_contribution/                     # Custom scripts and experiments
│   ├── fine_tune_sam_heart.py                   # Fine-tunes SAM-Med on the 22-Heart dataset
│   ├── prompt_type_comparison.py                # Evaluates model with different prompts
│   ├── prompt_type_comparison_debug.py          # Extended/debug evaluation (per-threshold outputs)
│   ├── plot_prompt_type_comparison.py           # Generates summary plots (Dice & HD95)
│   ├── plot_prompt_type_comparison_extended.py  # Extended plots per threshold & per prompt type
│   ├── visualize_prompt_type_comparison.py      # Bar plots for prompt efficiency comparison
│   ├── visualize_prompt_robustness_heatmaps.py  # Heatmaps for prompt robustness
│   └── ...
│
├── data/                                # Dataset and pre-computed embeddings
│   └── precompute_vit_b/test/22_Heart/*.npz
│
├── data_infor_json/                     # Dataset metadata/configuration files
│
├── results/                             # Experimental outputs
│   ├── prompt_type_comparison_heart/
│   │   ├── original_thr*/                # Results from baseline SAM-Med
│   │   ├── finetuned_epoch*/             # Results from fine-tuned SAM-Med (epochs 1–10)
│   │   ├── summary.csv                   # Aggregated averages across all experiments
│   │   ├── plots/                        # Summary plots (Dice & HD95 comparisons)
│   │   └── plots_extended/               # Extended plots (per threshold & per prompt type)
│   └── ...
│
└── README.md                            # Project overview and instructions
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- PyTorch (with CUDA if available)
  Trained on HPC Server with 4x GPU

### Running the Code

### 1. Fine-tune the model
```bash
python My_contribution/fine_tune_sam_heart.py
```

### 2. Evaluate prompt types
- Standard evaluation:
  ```bash
  python My_contribution/prompt_type_comparison.py
  ```
- Extended/debug evaluation (per-threshold CSVs):
  ```bash
  python My_contribution/prompt_type_comparison_debug.py
  ```

### 3. Generate visualizations
- Summary plots (Dice & HD95 averages):
  ```bash
  python My_contribution/plot_prompt_type_comparison.py
  ```
- Extended plots (per-threshold, per-prompt type):
  ```bash
  python My_contribution/plot_prompt_type_comparison_extended.py
  ```
- Heatmaps for robustness:
  ```bash
  python My_contribution/visualize_prompt_robustness_heatmaps.py
  ```
- Prompt efficiency comparison:
  ```bash
  python My_contribution/visualize_prompt_type_comparison.py
  ```

Results (Dice, HD95, and plots) will be saved automatically in the `results/` folder.

---

## 📊 Key Results

- **Box and Both prompts**: Achieved highest Dice scores (~0.75-0.80) after fine-tuning.  
- **Point prompts**: Improved modestly (~0.48), but boundary precision (HD95) remained weaker.  
- **Takeaway**: Fine-tuning improved overlap accuracy but did not consistently improve boundary stability with limited data.

Plots and CSVs are available in:  
- `results/prompt_type_comparison_heart/plots/`  
- `results/prompt_type_comparison_heart/plots_extended/`

---

## 🤝 Contribution

This repository is part of a university research project. External contributions are not expected, but feedback is welcome via GitHub issues.

---

## 📜 License

This project is released for **academic purposes only**.  
Model checkpoints (SAM-Med) are subject to their original licenses.

---
