# ASVD4LLM to CUR4LLM Migration Guide

This guide provides exact steps to convert your ASVD4LLM installation to CUR4LLM.

## Step-by-Step Migration

### Step 1: Files to Delete
```bash
# Remove original SVD-specific files
rm modules/svd_linear.py
rm asvd.py
rm huggingface_repos/build_asvd_repo.py
```

### Step 2: Files to Replace Completely

**Replace the entire content** of these files with the CUR4LLM versions provided:

1. **Create `modules/cur_linear.py`** - Use the complete CUR linear implementation
2. **Create `acur.py`** - Use the complete ACUR main script
3. **Update `binary_search.py`** - Replace with CUR-compatible version
4. **Update `sensitivity.py`** - Replace with CUR-compatible version
5. **Update `quantization.py`** - Replace with CUR-compatible version

### Step 3: Files to Add

Create these new files for HuggingFace integration:

1. **`huggingface_repos/build_acur_repo.py`**
2. **`huggingface_repos/configuration_acur_llama.py`**
3. **`huggingface_repos/modeling_acur_llama.py`**
4. **`huggingface_repos/configuration_acur_opt.py`**
5. **`huggingface_repos/modeling_acur_opt.py`**

### Step 4: Files to Keep Unchanged

These files remain exactly the same:
- `datautils.py`
- `act_aware_utils.py` 
- `evaluate_utils.py`
- `tools/` directory (all files)
- `requirements.txt`
- `.gitignore`
- `LICENSE`

### Step 5: Create New Experiment Scripts

```bash
# Create new experiment directory structure
mkdir -p experiments/cur/
```

Create **`experiments/cur_experiments.sh`** with the provided CUR experiment commands.

### Step 6: Update Import Statements

If you have any custom scripts that import from the old modules, update them:

```python
# OLD (ASVD4LLM)
from modules.svd_linear import SVDLinear, GradSVDLinear
from asvd import main

# NEW (CUR4LLM) 
from modules.cur_linear import CURLinear, GradCURLinear
from acur import main
```

### Step 7: Verify Installation

Test your installation:

```bash
# Basic functionality test
python -c "from modules.cur_linear import CURLinear; print('CUR4LLM installed successfully')"

# Run a simple experiment
CUDA_VISIBLE_DEVICES='0' python acur.py --model_id="facebook/opt-125m" --act_aware --alpha 0.5 --n_calib_samples 16 --scaling_method abs_mean --param_ratio_target 0.9 --use_cache
```

## Directory Structure After Migration

```
CUR4LLM/
├── modules/
│   └── cur_linear.py              # NEW: CUR implementation
├── huggingface_repos/
│   ├── build_acur_repo.py         # NEW: CUR repo builder
│   ├── configuration_acur_llama.py # NEW: LLAMA config
│   ├── modeling_acur_llama.py     # NEW: LLAMA modeling
│   ├── configuration_acur_opt.py  # NEW: OPT config
│   └── modeling_acur_opt.py       # NEW: OPT modeling
├── experiments/
│   └── cur_experiments.sh         # NEW: CUR experiments
├── tools/                         # UNCHANGED
├── acur.py                        # NEW: Main entry point
├── binary_search.py               # UPDATED: CUR-compatible
├── sensitivity.py                 # UPDATED: CUR-compatible
├── quantization.py                # UPDATED: CUR-compatible
├── datautils.py                   # UNCHANGED
├── act_aware_utils.py             # UNCHANGED
├── evaluate_utils.py              # UNCHANGED
├── requirements.txt               # UNCHANGED
├── README_CUR4LLM.md              # NEW: CUR documentation
└── MIGRATION_GUIDE.md             # NEW: This guide
```

## Key Differences in Usage

### Command Line Interface

```bash
# OLD (ASVD4LLM)
python asvd.py --model_id="facebook/opt-1.3b" --act_aware --alpha 0.5 --param_ratio_target 0.9

# NEW (CUR4LLM)
python acur.py --model_id="facebook/opt-1.3b" --act_aware --alpha 0.5 --param_ratio_target 0.9
```

### Repository Building

```bash
# OLD (ASVD4LLM)
python huggingface_repos/build_asvd_repo.py --model_id="facebook/opt-125m" --param_ratio_target 0.9

# NEW (CUR4LLM)
python huggingface_repos/build_acur_repo.py --model_id="facebook/opt-125m" --param_ratio_target 0.9
```

### Model Loading

```python
# OLD (ASVD4LLM)
model_id = "username/opt-125m-asvd90"

# NEW (CUR4LLM)
model_id = "username/opt-125m-acur90"
```

## Cache Compatibility

CUR4LLM uses separate cache files to avoid conflicts:

- **Calibration caches**: Compatible with ASVD4LLM (same format)
- **Sensitivity caches**: New format with `_cur_` identifier

Your existing calibration caches will work, but sensitivity caches will be regenerated.

## Performance Considerations

### Memory Usage
- CUR may use slightly more memory during decomposition (leverage score computation)
- Runtime memory is similar to ASVD due to same parameter ratios

### Computation Time
- Initial decomposition might be slower (leverage score computation)
- Inference speed is comparable to ASVD

### Quality Expectations
- CUR provides different compression characteristics than SVD
- May perform better on interpretable features
- Results may vary depending on the model architecture

## Troubleshooting Migration

### Issue: Import Errors
```bash
# Fix: Ensure all new files are in place
ls modules/cur_linear.py  # Should exist
python -c "from modules.cur_linear import CURLinear"  # Should work
```

### Issue: Cache Conflicts
```bash
# Fix: Clear cache if needed
rm -rf cache/*_sensitivity_*.pt
# Calibration caches can be kept
```

### Issue: CUDA Memory
```bash
# Fix: Reduce calibration samples temporarily
python acur.py --n_calib_samples 16 --param_ratio_target 0.9 [other args]
```

### Issue: Numerical Instability
```bash
# Fix: Use rank alignment
python acur.py --rank_align 8 --param_ratio_target 0.9 [other args]
```

## Validation Checklist

After migration, verify:

- [ ] `acur.py` runs without errors
- [ ] CUR models can be created and saved
- [ ] Evaluation metrics work correctly
- [ ] HuggingFace repositories build successfully
- [ ] Quantization works with CUR models
- [ ] Results are reasonable (PPL not drastically worse)

## Research Comparison

To compare CUR vs SVD performance:

1. **Keep ASVD4LLM backup**: Maintain original for comparison
2. **Run parallel experiments**: Same hyperparameters, different methods
3. **Compare metrics**: PPL, downstream tasks, interpretability
4. **Document differences**: Parameter efficiency, semantic preservation

## Getting Help

If you encounter issues during migration:

1. Check that all files are correctly placed
2. Verify CUDA/PyTorch compatibility
3. Test with smaller models first
4. Compare outputs with ASVD4LLM baseline
5. Check cache permissions and disk space

## Final Verification Command

Run this command to ensure everything works:

```bash
# Complete test
CUDA_VISIBLE_DEVICES='0' python acur.py \
    --model_id="facebook/opt-125m" \
    --act_aware \
    --alpha 0.5 \
    --n_calib_samples 16 \
    --scaling_method abs_mean \
    --param_ratio_target 0.9 \
    --use_cache \
    --eval_ppl wikitext2

# If this completes successfully, your migration is complete!
```