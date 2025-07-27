# code from https://github.com/IST-DASLab/gptq

import math
import time

import torch
import torch.nn as nn
import transformers
import numpy as np
import torch
import torch.nn as nn
from modules.cur_linear import CURLinear

DEBUG = False


def quantize(x, scale, zero, maxq):
    if maxq < 0:
        return (x > scale / 2).float() * scale + (x < zero / 2).float() * zero
    q = torch.clamp(torch.round(x / scale) + zero, 0, maxq)
    return scale * (q - zero)


class Quantizer(nn.Module):
    def __init__(self, shape=1):
        super(Quantizer, self).__init__()
        self.register_buffer("maxq", torch.tensor(0))
        self.register_buffer("scale", torch.zeros(shape))
        self.register_buffer("zero", torch.zeros(shape))

    def configure(
        self,
        bits,
        perchannel=False,
        sym=True,
        mse=False,
        norm=2.4,
        grid=100,
        maxshrink=0.8,
        trits=False,
    ):
        self.maxq = torch.tensor(2**bits - 1)
        self.perchannel = perchannel
        self.sym = sym
        self.mse = mse
        self.norm = norm
        self.grid = grid
        self.maxshrink = maxshrink
        if trits:
            self.maxq = torch.tensor(-1)

    def find_params(self, x, weight=False):
        dev = x.device
        self.maxq = self.maxq.to(dev)

        shape = x.shape
        if self.perchannel:
            if weight:
                x = x.flatten(1)
            else:
                if len(shape) == 4:
                    x = x.permute([1, 0, 2, 3])
                    x = x.flatten(1)
                if len(shape) == 3:
                    x = x.reshape((-1, shape[-1])).t()
                if len(shape) == 2:
                    x = x.t()
        else:
            x = x.flatten().unsqueeze(0)

        tmp = torch.zeros(x.shape[0], device=dev)
        xmin = torch.minimum(x.min(1)[0], tmp)
        xmax = torch.maximum(x.max(1)[0], tmp)

        if self.sym:
            xmax = torch.maximum(torch.abs(xmin), xmax)
            tmp = xmin < 0
            if torch.any(tmp):
                xmin[tmp] = -xmax[tmp]
        tmp = (xmin == 0) & (xmax == 0)
        xmin[tmp] = -1
        xmax[tmp] = +1

        if self.maxq < 0:
            self.scale = xmax
            self.zero = xmin
        else:
            self.scale = (xmax - xmin) / self.maxq
            if self.sym:
                self.zero = torch.full_like(self.scale, (self.maxq + 1) / 2)
            else:
                self.zero = torch.round(-xmin / self.scale)

        if self.mse:
            best = torch.full([x.shape[0]], float("inf"), device=dev)
            for i in range(int(self.maxshrink * self.grid)):
                p = 1 - i / self.grid
                xmin1 = p * xmin
                xmax1 = p * xmax
                scale1 = (xmax1 - xmin1) / self.maxq
                zero1 = torch.round(-xmin1 / scale1) if not self.sym else self.zero
                q = quantize(x, scale1.unsqueeze(1), zero1.unsqueeze(1), self.maxq)
                q -= x
                q.abs_()
                q.pow_(self.norm)
                err = torch.sum(q, 1)
                tmp = err < best
                if torch.any(tmp):
                    best[tmp] = err[tmp]
                    self.scale[tmp] = scale1[tmp]
                    self.zero[tmp] = zero1[tmp]
        if not self.perchannel:
            if weight:
                tmp = shape[0]
            else:
                tmp = shape[1] if len(shape) != 3 else shape[2]
            self.scale = self.scale.repeat(tmp)
            self.zero = self.zero.repeat(tmp)

        if weight:
            shape = [-1] + [1] * (len(shape) - 1)
            self.scale = self.scale.reshape(shape)
            self.zero = self.zero.reshape(shape)
            return
        if len(shape) == 4:
            self.scale = self.scale.reshape((1, -1, 1, 1))
            self.zero = self.zero.reshape((1, -1, 1, 1))
        if len(shape) == 3:
            self.scale = self.scale.reshape((1, 1, -1))
            self.zero = self.zero.reshape((1, 1, -1))
        if len(shape) == 2:
            self.scale = self.scale.unsqueeze(0)
            self.zero = self.zero.unsqueeze(0)

    def quantize(self, x):
        if self.ready():
            return quantize(x, self.scale, self.zero, self.maxq)
        return x

    def enabled(self):
        return self.maxq > 0

    def ready(self):
        return torch.all(self.scale != 0)


def find_layers(module, layers=[nn.Conv2d, nn.Linear], name=""):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(child, layers=layers, name=name + "." + name1 if name != "" else name1))
    return res


@torch.no_grad()
def rtn_quant_sequential(model, wbits):
    print("Starting RTN quantization for CUR components...")

    if "opt" in model.config._name_or_path:
        layers = model.model.decoder.layers
    elif "llama" in model.config._name_or_path:
        layers = model.model.layers
    
    for i in range(len(layers)):
        layer = layers[i].to(model.device)
        subset = find_layers(layer)
        for name in subset:
            module = subset[name]
            if isinstance(module, CURLinear):
                # Quantize each CUR component separately
                # C component
                quantizer_C = Quantizer()
                quantizer_C.configure(wbits, perchannel=True, sym=False, mse=False)
                quantizer_C.find_params(module.C.data.float(), weight=True)
                module.C.data = quantizer_C.quantize(module.C.data.float()).to(module.C.data.dtype)
                
                # U component  
                quantizer_U = Quantizer()
                quantizer_U.configure(wbits, perchannel=True, sym=False, mse=False)
                quantizer_U.find_params(module.U.data.float(), weight=True)
                module.U.data = quantizer_U.quantize(module.U.data.float()).to(module.U.data.dtype)
                
                # R component
                quantizer_R = Quantizer()
                quantizer_R.configure(wbits, perchannel=True, sym=False, mse=False)
                quantizer_R.find_params(module.R.data.float(), weight=True)
                module.R.data = quantizer_R.quantize(module.R.data.float()).to(module.R.data.dtype)
                
                print(f"Quantizing CUR components in {name} finished")
            else:
                # Standard linear layer quantization
                quantizer = Quantizer()
                quantizer.configure(wbits, perchannel=True, sym=False, mse=False)
                quantizer.find_params(module.weight.data.float(), weight=True)
                wq = quantizer.quantize(module.weight.data.float())
                module.weight.data = wq.to(module.weight.data.dtype)
                print(f"Quantizing {name} finished")
        del layer
        torch.cuda.empty_cache()


def awq_quant_sequential(model, tokenizer, wbits):
    from awq.models import LlamaAWQForCausalLM

    device = model.device

    class ACURLlamaAWQForCausalLM(LlamaAWQForCausalLM):
        @staticmethod
        def get_layers_for_scaling(module, input_feat, module_kwargs):
            layers = []
            input_names = []

            def curlinear_process(module, inp_name, module2inspect=None, kwargs=None):
                module2inspect = None
                kwargs = {}
                if isinstance(module, CURLinear):
                    # For CUR, we need to handle the decomposed structure
                    # We'll treat it as multiple linear operations: R -> U -> C
                    layers.append(
                        dict(
                            prev_op=module.R,  # First operation in the chain
                            layers=[module.C],  # Final operation
                            inp=input_feat[inp_name + ".C"],
                            module2inspect=module2inspect,
                            kwargs=kwargs,
                        )
                    )
                    input_names.append(inp_name + ".R")
                    return module.R
                else:
                    input_names.append(inp_name)
                    return module

            # Attention input
            layers.append(
                dict(
                    prev_op=module.input_layernorm,
                    layers=[
                        curlinear_process(module.self_attn.k_proj, "self_attn.k_proj", module.self_attn, module_kwargs),
                        curlinear_process(module.self_attn.q_proj, "self_attn.q_proj", module.self_attn, module_kwargs),
                        curlinear_process(module.self_attn.v_proj, "self_attn.v_proj", module.self_attn, module_kwargs),
                    ],
                    inp=input_feat[input_names[-1]],
                    module2inspect=module.self_attn,
                    kwargs=module_kwargs,
                )
            )

            # Attention out
            layers.append(
                dict(
                    prev_op=(
                        module.self_attn.v_proj.C
                        if isinstance(module.self_attn.v_proj, CURLinear)
                        else module.self_attn.v_proj
                    ),
                    layers=[curlinear_process(module.self_attn.o_proj, "self_attn.o_proj")],
                    inp=input_feat[input_names[-1]],
                )
            )

            # Linear 1
            layers.append(
                dict(
                    prev_op=module.post_attention_layernorm,
                    layers=[
                        curlinear_process(module.mlp.gate_proj, "mlp.gate_proj", module.mlp),
                        curlinear_process(module.mlp.up_proj, "mlp.up_proj", module.mlp),
                    ],
                    inp=input_feat[input_names[-1]],
                    module2inspect=module.mlp,
                )
            )

            # Linear 2
            layers.append(
                dict(
                    prev_op=(
                        module.mlp.up_proj.C if isinstance(module.mlp.up_proj, CURLinear) else module.mlp.up_proj
                    ),
                    layers=[curlinear_process(module.mlp.down_proj, "mlp.down_proj")],
                    inp=input_feat[input_names[-1]],
                )
            )

            return layers

    quant_config = {"zero_point": True, "q_group_size": 128, "w_bit": wbits, "version": "GEMM"}

    # Load model
    qmodel = ACURLlamaAWQForCausalLM(model, "llama", False, model.config, quant_config, None)

    # Quantize
    qmodel.quantize(tokenizer, quant_config=quant_config)
    qmodel.to(device)
    qmodel.device = device
    qmodel.lm_head = lambda x: x

    # If CUDA error, change q_linear=q_linear_module.from_linear to q_linear=linear_layer
    # in awq/quantize/quantizer.py AwqQuantizer._apply_quant function

    return qmodel