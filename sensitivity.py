import os
import torch
import torch.nn as nn
from modules.cur_linear import CURLinear
from evaluate_utils import evaluate_model, evaluate_perplexity
from tqdm import tqdm
import numpy as np


@torch.no_grad()
def calib_sensitivity_ppl(model, calib_loader, args, use_cache=True):
    model_id = model.config._name_or_path
    cache_file = f"cache/{model_id.replace('/','_')}_sensitivity_cur_{args.scaling_method}_{args.alpha}_{args.n_calib_samples}_{args.calib_dataset}.pt"
    if os.path.exists(cache_file) and use_cache:
        sensitivity_dict = torch.load(cache_file, map_location="cpu")
        return sensitivity_dict
    model.eval()

    full_name_dict = {module: name for name, module in model.named_modules()}
    linear_info = {}
    modules = [model]
    while len(modules) > 0:
        submodule = modules.pop()
        for name, raw_linear in submodule.named_children():
            if isinstance(raw_linear, nn.Linear):
                full_name = full_name_dict[raw_linear]
                linear_info[raw_linear] = {
                    "father": submodule,
                    "name": name,
                    "full_name": full_name,
                }
            else:
                modules.append(raw_linear)

    sensitivity_dict = {}
    if args.compress_kv_cache:
        param_ratio_candidates = [0.1 * i for i in range(1, 20)]
    else:
        param_ratio_candidates = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    input_ids = torch.cat([_["input_ids"] for _ in calib_loader], 0)
    print(f"input_ids.shape={input_ids.shape}")
    pbar = tqdm(total=len(linear_info) * len(param_ratio_candidates))
    for raw_linear, info in linear_info.items():
        sensitivity_dict[info["full_name"]] = {}
        for param_ratio in param_ratio_candidates:
            cur_linear = CURLinear.from_linear(
                raw_linear,
                param_ratio=param_ratio,
                alpha=args.alpha,
                act_aware=True,
                rank_align=args.rank_align,
            )
            setattr(info["father"], info["name"], cur_linear)

            ppl = evaluate_perplexity(model, input_ids, args.n_calib_samples)
            sensitivity_dict[info["full_name"]][param_ratio] = ppl
            print(f"{info['full_name']} {param_ratio} {ppl}")
            pbar.update(1)
        setattr(info["father"], info["name"], raw_linear)
    torch.save(sensitivity_dict, cache_file)
    return sensitivity_dict


@torch.no_grad()
def calib_sensitivity_stable_rank(model, calib_loader, args, use_cache=True):
    model_id = model.config._name_or_path
    cache_file = f"cache/{model_id.replace('/','_')}_sensitivity_cur_stable_rank_{args.scaling_method}_{args.alpha}_{args.n_calib_samples}_{args.calib_dataset}.pt"
    if os.path.exists(cache_file) and use_cache:
        sensitivity_dict = torch.load(cache_file, map_location="cpu")
        return sensitivity_dict
    model.eval()

    full_name_dict = {module: name for name, module in model.named_modules()}
    linear_info = {}
    modules = [model]
    while len(modules) > 0:
        submodule = modules.pop()
        for name, raw_linear in submodule.named_children():
            if isinstance(raw_linear, nn.Linear):
                full_name = full_name_dict[raw_linear]
                linear_info[raw_linear] = {
                    "father": submodule,
                    "name": name,
                    "full_name": full_name,
                }
            else:
                modules.append(raw_linear)

    sensitivity_dict = {}
    param_ratio_candidates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    input_ids = torch.cat([_["input_ids"] for _ in calib_loader], 0)
    print(f"input_ids.shape={input_ids.shape}")
    pbar = tqdm(total=len(linear_info) * len(param_ratio_candidates))
    for raw_linear, info in linear_info.items():
        sensitivity_dict[info["full_name"]] = {}

        # For CUR, we compute a modified stable rank that considers column/row selection
        w = raw_linear.weight
        if hasattr(raw_linear, "scaling_diag_matrix"):
            w = w * raw_linear.scaling_diag_matrix.view(1, -1) ** args.alpha
        if hasattr(raw_linear, "fisher_info"):
            w = w * raw_linear.fisher_info.view(1, -1) ** args.alpha
            
        w_fro = torch.norm(w, p="fro") ** 2
        try:
            _, singular_values, _ = torch.svd_lowrank(w.float(), q=min(w.shape)//4)
            spectral_norm = torch.max(singular_values)
            w_spec = spectral_norm**2
            sr = (w_fro / w_spec) ** 0.5
        except:
            sr = 1.0  # Fallback value

        for param_ratio in param_ratio_candidates:
            # For CUR, we modify the stable rank score to account for the selection process
            # Lower stable rank indicates better compressibility with CUR
            sensitivity_dict[info["full_name"]][param_ratio] = -sr * (param_ratio**0.1)
            pbar.update(1)
    torch.save(sensitivity_dict, cache_file)
    return sensitivity_dict