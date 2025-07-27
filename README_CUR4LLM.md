# CUR4LLM: Activation-aware CUR Decomposition for Compressing Large Language Models

This work explores CUR decomposition as an alternative to SVD for compressing Large Language Models. CUR decomposition provides interpretable compression by selecting actual columns and rows from weight matrices, offering better semantic understanding compared to SVD's linear combinations.

Building upon the ASVD4LLM framework, we introduce **Activation-aware CUR (ACUR)** decomposition that leverages activation distributions to select the most important columns and rows using statistical leverage scores, similar to how ASVD uses activation awareness for SVD.

## Key Advantages of CUR over SVD

1. **Interpretability**: CUR uses actual columns/rows from the original matrix, not linear combinations
2. **Better Semantic Preservation**: Selected columns/rows correspond to meaningful features
3. **Statistical Foundation**: Uses leverage scores from CUR matrix decomposition theory
4. **Activation Awareness**: Incorporates activation distributions like ASVD

## Requirements
- python>=3.10
- pip install -r requirements.txt

## Quick Start

### Direct Usage with Pre-compressed Models

```python3
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Note: Replace with actual CUR4LLM model paths when available
model_id = "your-username/opt-125m-acur90"

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True
)
```

### Running CUR4LLM Compression

Basic usage:
```bash
CUDA_VISIBLE_DEVICES='0' python acur.py --model_id="facebook/opt-125m" --act_aware --alpha 0.5 --n_calib_samples 16 --scaling_method abs_mean --param_ratio_target 0.9 --use_cache
```

Advanced examples:
```bash
# LLaMA-2 7B with 90% parameter ratio
CUDA_VISIBLE_DEVICES='1' python acur.py --model_id="meta-llama/Llama-2-7b-hf" --act_aware --alpha 0.5 --n_calib_samples 32 --scaling_method abs_mean --param_ratio_target 0.9 --use_cache

# With MMLU evaluation
CUDA_VISIBLE_DEVICES='2' python acur.py --model_id="meta-llama/Llama-2-7b-hf" --act_aware --alpha 0.5 --n_calib_samples 32 --scaling_method abs_mean --param_ratio_target 0.95 --use_cache --eval_mmlu

# KV cache compression
CUDA_VISIBLE_DEVICES='3' python acur.py --model_id="meta-llama/Llama-2-13b-hf" --act_aware --alpha 0.5 --n_calib_samples 32 --scaling_method abs_mean --kv_cache_ratio_target 0.9 --use_cache --compress_kv_cache
```

## Command Line Arguments

```
usage: acur.py [-h] [--model_id MODEL_ID] [--ppl_target PPL_TARGET] [--param_ratio_target PARAM_RATIO_TARGET] [--act_aware] [--alpha ALPHA] [--n_calib_samples N_CALIB_SAMPLES] [--calib_dataset {wikitext2,c4,ptb}]
               [--scaling_method {abs_mean,abs_max,fisher}] [--use_cache] [--compress_kv_cache] [--kv_cache_ratio_target KV_CACHE_RATIO_TARGET]

Key options:
  --model_id MODEL_ID       Pretrained model ID
  --param_ratio_target      Target parameter ratio (0.9 = 90% compression)
  --act_aware              Use activation-aware CUR (ACUR)
  --alpha ALPHA            Hyper-parameter alpha for ACUR (default: 0.5)
  --n_calib_samples        Number of calibration samples (default: 32)
  --scaling_method         Scaling method: abs_mean, abs_max, fisher
  --compress_kv_cache      Compress KV cache using ACUR
  --use_cache              Use cached calibration results
```

## How CUR4LLM Works

### Statistical Leverage Scores
CUR4LLM computes statistical leverage scores to identify the most important columns and rows:

```
π_j = (1/k) * Σ(v_ξj)² for ξ=1 to k
```

Where `v_ξj` are elements of the top-k right singular vectors.

### Activation-Aware Selection
Similar to ASVD4LLM, we incorporate activation distributions:

1. **Calibration**: Collect activation statistics from calibration data
2. **Scaling**: Apply scaling based on activation magnitudes
3. **Selection**: Use leverage scores on scaled matrices to select columns/rows
4. **Decomposition**: Construct CUR = C * U * R decomposition

### CUR vs SVD Comparison

| Aspect | SVD | CUR |
|--------|-----|-----|
| Interpretability | Linear combinations | Actual columns/rows |
| Semantic Meaning | Abstract | Concrete features |
| Selection Method | Mathematical optimum | Statistical leverage |
| Activation Awareness | Matrix scaling | Leverage-weighted selection |

### Algorithm Overview

1. **Input**: Linear layer weight matrix W [m×n]
2. **Activation Scaling**: W_scaled = W * activation_scaling^α  
3. **Leverage Computation**: Compute column/row leverage scores
4. **Selection**: Sample c columns and r rows based on leverage scores
5. **Decomposition**: Compute U = C⁺ * W_subset * R⁺
6. **Reconstruction**: W ≈ C * U * R

## Creating Huggingface Repositories

Build a CUR4LLM model repository:

```bash
CUDA_VISIBLE_DEVICES='0' python huggingface_repos/build_acur_repo.py --model_id="facebook/opt-125m" --act_aware --alpha 0.5 --n_calib_samples 32 --scaling_method abs_mean --param_ratio_target 0.9 --use_cache
```

This creates a repository in `huggingface_repos/opt-125m-acur90/` that can be used directly:

```python3
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "huggingface_repos/opt-125m-acur90"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True
)
```

## Files Modified from ASVD4LLM

### Key Changes:
1. **`modules/svd_linear.py` → `modules/cur_linear.py`**: Core CUR implementation
2. **`asvd.py` → `acur.py`**: Main entry point adapted for CUR
3. **`binary_search.py`**: Updated to use CURLinear instead of SVDLinear
4. **`sensitivity.py`**: CUR-specific sensitivity analysis
5. **`quantization.py`**: Support for quantizing CUR components
6. **New HuggingFace configs**: `configuration_acur_*.py` and `modeling_acur_*.py`

### Architecture Differences:
- **SVD**: W ≈ U * Σ * V^T (3 matrices, all dense)
- **CUR**: W ≈ C * U * R (3 matrices, C and R are selected columns/rows)

## Experimental Results Structure

The framework provides the same evaluation capabilities as ASVD4LLM:

- **Perplexity**: WikiText2, PTB, C4 datasets
- **Downstream Tasks**: MMLU, LongBench, etc.
- **Memory Usage**: Track compression ratios
- **Quality Metrics**: Compare against baselines

## Cache Management

CUR4LLM uses separate cache files to avoid conflicts with ASVD4LLM:
- Sensitivity cache: `*_sensitivity_cur_*.pt`
- Calibration cache: Same as ASVD4LLM (compatible)

## Research Applications

This implementation enables research comparing:
1. **CUR vs SVD**: Interpretability vs compression efficiency
2. **Activation Awareness**: Impact on column/row selection
3. **Statistical Leverage**: Theoretical vs practical performance
4. **Semantic Preservation**: Downstream task performance

## Citation

If you use CUR4LLM in your research, please cite both the original ASVD work and relevant CUR papers:

```bibtex
@misc{yuan2023asvd,
    title={ASVD: Activation-aware Singular Value Decomposition for Compressing Large Language Models}, 
    author={Zhihang Yuan and Yuzhang Shang and Yue Song and Qiang Wu and Yan Yan and Guangyu Sun},
    year={2023},
    eprint={2312.05821},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}

@article{mahoney2009cur,
    title={CUR matrix decompositions for improved data analysis},
    author={Mahoney, Michael W and Drineas, Petros},
    journal={Proceedings of the national academy of sciences},
    volume={106},
    number={3},
    pages={697--702},
    year={2009},
    publisher={National Acad Sciences}
}
```

## Troubleshooting

### Common Issues:
1. **CUDA Memory**: CUR may use more memory during decomposition due to leverage score computation
2. **Numerical Stability**: Falls back to uniform selection if leverage computation fails
3. **Cache Conflicts**: Uses separate cache files with "cur" prefix

### Performance Tips:
1. Use `--use_cache` to avoid recomputation
2. Start with smaller models for testing
3. Monitor GPU memory usage during decomposition
4. Use appropriate `rank_align` values for memory efficiency