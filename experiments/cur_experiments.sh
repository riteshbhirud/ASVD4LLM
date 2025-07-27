#!/bin/bash

# CUR4LLM Experiments
# Replace SVD experiments with CUR experiments

# Basic CUR experiments
CUDA_VISIBLE_DEVICES='0' python acur.py --model_id="facebook/opt-1.3b" --act_aware --alpha 0.5 --n_calib_samples 32 --scaling_method abs_mean --param_ratio_target 0.9 --use_cache
CUDA_VISIBLE_DEVICES='1' python acur.py --model_id="facebook/opt-1.3b" --act_aware --alpha 0.5 --n_calib_samples 32 --scaling_method abs_mean --param_ratio_target 0.85 --use_cache
CUDA_VISIBLE_DEVICES='2' python acur.py --model_id="facebook/opt-1.3b" --act_aware --alpha 0.5 --n_calib_samples 32 --scaling_method abs_mean --param_ratio_target 0.8 --use_cache

# Gemma experiments with CUR
CUDA_VISIBLE_DEVICES='2' python acur.py --model_id="models/gemma-2-9b" --act_aware --alpha 0.5 --n_calib_samples 32 --scaling_method abs_mean --param_ratio_target 0.9 --use_cache
CUDA_VISIBLE_DEVICES='3' python acur.py --model_id="models/gemma-2-2b" --act_aware --alpha 0.5 --n_calib_samples 32 --scaling_method abs_mean --param_ratio_target 0.9 --use_cache

# LLaMA experiments with CUR
CUDA_VISIBLE_DEVICES='0' python acur.py --model_id="models/Llama-2-7b-hf" --act_aware --alpha 0.5 --n_calib_samples 32 --scaling_method abs_mean --param_ratio_target 0.95 --use_cache --eval_mmlu
CUDA_VISIBLE_DEVICES='1' python acur.py --model_id="models/Llama-2-7b-hf" --act_aware --alpha 0.5 --n_calib_samples 32 --scaling_method abs_mean --param_ratio_target 0.9 --use_cache --eval_mmlu
CUDA_VISIBLE_DEVICES='2' python acur.py --model_id="models/Llama-2-7b-hf" --act_aware --alpha 0.5 --n_calib_samples 32 --scaling_method abs_mean --param_ratio_target 0.85 --use_cache --eval_mmlu

# Comparison between different calibration datasets
CUDA_VISIBLE_DEVICES='2' python acur.py --model_id="models/Llama-2-7b-hf" --act_aware --alpha 0.5 --n_calib_samples 32 --scaling_method abs_mean --param_ratio_target 0.95 --use_cache --calib_dataset selfgen --seed 42
CUDA_VISIBLE_DEVICES='2' python acur.py --model_id="models/Llama-2-7b-hf" --act_aware --alpha 0.5 --n_calib_samples 32 --scaling_method abs_mean --param_ratio_target 0.9 --use_cache --calib_dataset selfgen --seed 42
CUDA_VISIBLE_DEVICES='2' python acur.py --model_id="models/Llama-2-7b-hf" --act_aware --alpha 0.5 --n_calib_samples 32 --scaling_method abs_mean --param_ratio_target 0.85 --use_cache --calib_dataset selfgen --seed 42

CUDA_VISIBLE_DEVICES='3' python acur.py --model_id="models/Llama-2-13b-hf" --act_aware --alpha 0.5 --n_calib_samples 32 --scaling_method abs_mean --param_ratio_target 0.95 --use_cache --calib_dataset selfgen --seed 42
CUDA_VISIBLE_DEVICES='3' python acur.py --model_id="models/Llama-2-13b-hf" --act_aware --alpha 0.5 --n_calib_samples 32 --scaling_method abs_mean --param_ratio_target 0.9 --use_cache --calib_dataset selfgen --seed 42
CUDA_VISIBLE_DEVICES='3' python acur.py --model_id="models/Llama-2-13b-hf" --act_aware --alpha 0.5 --n_calib_samples 32 --scaling_method abs_mean --param_ratio_target 0.85 --use_cache --calib_dataset selfgen --seed 42

CUDA_VISIBLE_DEVICES='2' python acur.py --model_id="models/Llama-2-7b-hf" --act_aware --alpha 0.5 --n_calib_samples 32 --scaling_method abs_mean --param_ratio_target 0.95 --use_cache --calib_dataset c4 --seed 42
CUDA_VISIBLE_DEVICES='2' python acur.py --model_id="models/Llama-2-7b-hf" --act_aware --alpha 0.5 --n_calib_samples 32 --scaling_method abs_mean --param_ratio_target 0.9 --use_cache --calib_dataset c4 --seed 42
CUDA_VISIBLE_DEVICES='2' python acur.py --model_id="models/Llama-2-7b-hf" --act_aware --alpha 0.5 --n_calib_samples 32 --scaling_method abs_mean --param_ratio_target 0.85 --use_cache --calib_dataset c4 --seed 42

CUDA_VISIBLE_DEVICES='3' python acur.py --model_id="models/Llama-2-13b-hf" --act_aware --alpha 0.5 --n_calib_samples 32 --scaling_method abs_mean --param_ratio_target 0.95 --use_cache --calib_dataset c4 --seed 42
CUDA_VISIBLE_DEVICES='3' python acur.py --model_id="models/Llama-2-13b-hf" --act_aware --alpha 0.5 --n_calib_samples 32 --scaling_method abs_mean --param_ratio_target 0.9 --use_cache --calib_dataset c4 --seed 42
CUDA_VISIBLE_DEVICES='3' python acur.py --model_id="models/Llama-2-13b-hf" --act_aware --alpha 0.5 --n_calib_samples 32 --scaling_method abs_mean --param_ratio_target 0.85 --use_cache --calib_dataset c4 --seed 42

# KV cache compression with CUR  
CUDA_VISIBLE_DEVICES='0' python acur.py --model_id="models/Llama-2-7b-hf" --act_aware --alpha 0.5 --n_calib_samples 32 --scaling_method abs_mean --kv_cache_ratio_target 0.9 --use_cache --compress_kv_cache --eval_tasks small_longbench &
CUDA_VISIBLE_DEVICES='1' python acur.py --model_id="models/Llama-2-7b-hf" --act_aware --alpha 0.5 --n_calib_samples 32 --scaling_method abs_mean --kv_cache_ratio_target 0.8 --use_cache --compress_kv_cache --eval_tasks small_longbench &
CUDA_VISIBLE_DEVICES='2' python acur.py --model_id="models/Llama-2-7b-hf" --act_aware --alpha 0.5 --n_calib_samples 32 --scaling_method abs_mean --kv_cache_ratio_target 0.7 --use_cache --compress_kv_cache --eval_tasks small_longbench &
CUDA_VISIBLE_DEVICES='3' python acur.py --model_id="models/Llama-2-7b-hf" --act_aware --alpha 0.5 --n_calib_samples 32 --scaling_method abs_mean --kv_cache_ratio_target 0.6 --use_cache --compress_kv_cache --eval_tasks small_longbench &
CUDA_VISIBLE_DEVICES='4' python acur.py --model_id="models/Llama-2-7b-hf" --act_aware --alpha 0.5 --n_calib_samples 32 --scaling_method abs_mean --kv_cache_ratio_target 0.5 --use_cache --compress_kv_cache --eval_tasks small_longbench &
CUDA_VISIBLE_DEVICES='5' python acur.py --model_id="models/Llama-2-7b-hf" --act_aware --alpha 0.5 --n_calib_samples 32 --scaling_method abs_mean --kv_cache_ratio_target 0.4 --use_cache --compress_kv_cache --eval_tasks small_longbench &

# Quantization experiments with CUR
CUDA_VISIBLE_DEVICES='2' python acur.py --model_id="models/Llama-2-7b-hf" --act_aware --alpha 0.5 --n_calib_samples 32 --scaling_method abs_mean --param_ratio_target 1 --use_cache --weight_quant awq_int8 --rank_align=128
CUDA_VISIBLE_DEVICES='2' python acur.py --model_id="models/Llama-2-7b-hf" --act_aware --alpha 0.5 --n_calib_samples 32 --scaling_method abs_mean --param_ratio_target 0.95 --use_cache --weight_quant awq_int8 --rank_align=128
CUDA_VISIBLE_DEVICES='2' python acur.py --model_id="models/Llama-2-7b-hf" --act_aware --alpha 0.5 --n_calib_samples 32 --scaling_method abs_mean --param_ratio_target 0.9 --use_cache --weight_quant awq_int8 --rank_align=128
CUDA_VISIBLE_DEVICES='2' python acur.py --model_id="models/Llama-2-7b-hf" --act_aware --alpha 0.5 --n_calib_samples 32 --scaling_method abs_mean --param_ratio_target 0.85 --use_cache --weight_quant awq_int8 --rank_align=128