import math
from pathlib import Path
import torch
from torch import nn
from collections import defaultdict
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import (
    load_model_and_tokenizer,
    get_input_device,
    load_c4_calibration,
    save_model_and_tokenizer,
    iter_batches,
    get_decoder_layers,
)

# Configurable Pruning Ratios
MODEL_ID = "Qwen/Qwen3.5-0.8B"
OUTPUT_DIR = "models/qwen-width-pruned"
ATTN_HEAD_PRUNE_RATIO = 0.25   # Prune 25% of attention heads (leaves 6 out of 8 heads)
MLP_PRUNE_RATIO = 0.20         # Prune 20% of MLP intermediate dimension (leaves 2864 out of 3584)

N_CALIBRATION_SAMPLES = 4      # Set to 128 for reliable pruning, or 4 for a quick smoke test
SEQUENCE_LENGTH = 64           # Set to 2048 for full context, or 64 for a quick smoke test
RANDOM_SEED = 0
BATCH_SIZE = 1
TORCH_DTYPE = "auto"
DEVICE_MAP = "auto"
TRUST_REMOTE_CODE = True       # Qwen3.5 text model needs trust_remote_code=True for hybrid layers

# --- Activation Stats Collector ---

class ActivationStatsCollector:
    """Accumulates activation norms for attention heads and MLP intermediate dimensions."""
    def __init__(self, model):
        self.model = model
        # layer_idx -> sum of squared activations for query heads [num_attention_heads]
        self.attn_stats = defaultdict(lambda: None)
        # layer_idx -> sum of squared activations for MLP intermediate dimensions [intermediate_size]
        self.mlp_stats = defaultdict(lambda: None)
        self.handles = []

    def register_hooks(self):
        layers = get_decoder_layers(self.model)
        for layer_idx, layer in enumerate(layers):
            # 1. Standard self_attn layer (Qwen3_5Attention)
            if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "o_proj"):
                o_proj = layer.self_attn.o_proj
                handle = o_proj.register_forward_hook(
                    self.make_attn_hook(layer_idx, layer.self_attn)
                )
                self.handles.append(handle)
            
            # 2. MLP layer (Qwen3_5MLP)
            if hasattr(layer, "mlp") and hasattr(layer.mlp, "down_proj"):
                down_proj = layer.mlp.down_proj
                handle = down_proj.register_forward_hook(
                    self.make_mlp_hook(layer_idx)
                )
                self.handles.append(handle)

    def make_attn_hook(self, layer_idx, self_attn):
        num_heads = self_attn.config.num_attention_heads
        head_dim = self_attn.head_dim
        def hook(module, inputs, output):
            # Input to o_proj is the concatenated outputs of the heads.
            # Shape: [batch, seq, num_attention_heads * head_dim]
            x = inputs[0].detach().float()
            x = x.reshape(-1, num_heads, head_dim)
            # Sum squares over the token (batch * seq) and head_dim dimensions
            scores = torch.sum(x.pow(2), dim=(0, 2)).cpu()
            if self.attn_stats[layer_idx] is None:
                self.attn_stats[layer_idx] = scores
            else:
                self.attn_stats[layer_idx] += scores
        return hook

    def make_mlp_hook(self, layer_idx):
        def hook(module, inputs, output):
            # Input to down_proj is the intermediate activations.
            # Shape: [batch, seq, intermediate_size]
            x = inputs[0].detach().float()
            x = x.reshape(-1, x.shape[-1])
            # Sum squares over the token dimension
            scores = torch.sum(x.pow(2), dim=0).cpu()
            if self.mlp_stats[layer_idx] is None:
                self.mlp_stats[layer_idx] = scores
            else:
                self.mlp_stats[layer_idx] += scores
        return hook

    def remove_hooks(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []

# --- GQA Compatibility Resolvers ---

def get_valid_gqa_targets(num_heads, num_kv_heads, prune_ratio):
    """Calculate valid GQA target head numbers that are divisible and <= original."""
    target_heads = max(1, int(round(num_heads * (1.0 - prune_ratio))))
    target_kv_heads = max(1, int(round(num_kv_heads * (1.0 - prune_ratio))))
    
    if target_heads % target_kv_heads == 0:
        return target_heads, target_kv_heads
        
    best_h, best_kv = None, None
    min_dist = float('inf')
    for h in range(1, num_heads + 1):
        for kv in range(1, num_kv_heads + 1):
            if h % kv == 0:
                dist = abs(h - target_heads) + abs(kv - target_kv_heads)
                if dist < min_dist:
                    min_dist = dist
                    best_h = h
                    best_kv = kv
    return best_h, best_kv

def select_heads_to_keep(scores, num_heads, num_kv_heads, target_heads, target_kv_heads):
    """Select the most important query heads and key-value heads while preserving symmetry."""
    orig_group_size = num_heads // num_kv_heads
    new_group_size = target_heads // target_kv_heads
    
    # Calculate group importance scores
    kv_scores = []
    for k in range(num_kv_heads):
        group_query_heads = range(k * orig_group_size, (k + 1) * orig_group_size)
        group_score = sum(scores[h].item() for h in group_query_heads)
        kv_scores.append((k, group_score))
        
    # Keep target_kv_heads KV groups with the highest aggregated scores
    kv_scores.sort(key=lambda x: x[1], reverse=True)
    kept_kv_indices = [kv_scores[i][0] for i in range(target_kv_heads)]
    kept_kv_indices.sort()
    
    # Within each kept KV group, keep the top new_group_size query heads
    kept_query_indices = []
    for old_k_idx in kept_kv_indices:
        group_query_heads = list(range(old_k_idx * orig_group_size, (old_k_idx + 1) * orig_group_size))
        group_query_heads.sort(key=lambda h: scores[h].item(), reverse=True)
        selected_heads = group_query_heads[:new_group_size]
        selected_heads.sort()
        kept_query_indices.extend(selected_heads)
        
    return kept_query_indices, kept_kv_indices

# --- Layer Weight Slicing ---

def prune_attn_layer(self_attn, kept_q_indices, kept_kv_indices):
    """Replace linear modules in self_attn with sliced ones."""
    head_dim = self_attn.head_dim
    device = self_attn.q_proj.weight.device
    dtype = self_attn.q_proj.weight.dtype

    # 1. q_proj (output features: num_attention_heads * head_dim * 2)
    # The output contains both standard queries and gates concatenated.
    q_indices = []
    for h in kept_q_indices:
        q_indices.extend(list(range(h * head_dim * 2, (h + 1) * head_dim * 2)))
    q_proj = self_attn.q_proj
    new_q_weight = q_proj.weight.data[q_indices, :]
    new_q_proj = nn.Linear(q_proj.in_features, len(q_indices), bias=(q_proj.bias is not None)).to(device=device, dtype=dtype)
    new_q_proj.weight.data.copy_(new_q_weight)
    if q_proj.bias is not None:
        new_q_proj.bias.data.copy_(q_proj.bias.data[q_indices])
    self_attn.q_proj = new_q_proj

    # 2. k_proj (output features: num_key_value_heads * head_dim)
    k_indices = []
    for k in kept_kv_indices:
        k_indices.extend(list(range(k * head_dim, (k + 1) * head_dim)))
    k_proj = self_attn.k_proj
    new_k_weight = k_proj.weight.data[k_indices, :]
    new_k_proj = nn.Linear(k_proj.in_features, len(k_indices), bias=(k_proj.bias is not None)).to(device=device, dtype=dtype)
    new_k_proj.weight.data.copy_(new_k_weight)
    if k_proj.bias is not None:
        new_k_proj.bias.data.copy_(k_proj.bias.data[k_indices])
    self_attn.k_proj = new_k_proj

    # 3. v_proj (output features: num_key_value_heads * head_dim)
    v_proj = self_attn.v_proj
    new_v_weight = v_proj.weight.data[k_indices, :]
    new_v_proj = nn.Linear(v_proj.in_features, len(k_indices), bias=(v_proj.bias is not None)).to(device=device, dtype=dtype)
    new_v_proj.weight.data.copy_(new_v_weight)
    if v_proj.bias is not None:
        new_v_proj.bias.data.copy_(v_proj.bias.data[k_indices])
    self_attn.v_proj = new_v_proj

    # 4. o_proj (input features: num_attention_heads * head_dim)
    o_indices = []
    for h in kept_q_indices:
        o_indices.extend(list(range(h * head_dim, (h + 1) * head_dim)))
    o_proj = self_attn.o_proj
    new_o_weight = o_proj.weight.data[:, o_indices]
    new_o_proj = nn.Linear(len(o_indices), o_proj.out_features, bias=(o_proj.bias is not None)).to(device=device, dtype=dtype)
    new_o_proj.weight.data.copy_(new_o_weight)
    if o_proj.bias is not None:
        new_o_proj.bias.data.copy_(o_proj.bias.data)
    self_attn.o_proj = new_o_proj

    # Update groups count
    self_attn.num_key_value_groups = len(kept_q_indices) // len(kept_kv_indices)

def prune_mlp_layer(mlp, kept_mlp_indices):
    """Replace linear modules in mlp with sliced ones."""
    device = mlp.gate_proj.weight.device
    dtype = mlp.gate_proj.weight.dtype
    
    # 1. gate_proj (output features: intermediate_size)
    gate_proj = mlp.gate_proj
    new_gate_weight = gate_proj.weight.data[kept_mlp_indices, :]
    new_gate_proj = nn.Linear(gate_proj.in_features, len(kept_mlp_indices), bias=(gate_proj.bias is not None)).to(device=device, dtype=dtype)
    new_gate_proj.weight.data.copy_(new_gate_weight)
    if gate_proj.bias is not None:
        new_gate_proj.bias.data.copy_(gate_proj.bias.data[kept_mlp_indices])
    mlp.gate_proj = new_gate_proj

    # 2. up_proj (output features: intermediate_size)
    up_proj = mlp.up_proj
    new_up_weight = up_proj.weight.data[kept_mlp_indices, :]
    new_up_proj = nn.Linear(up_proj.in_features, len(kept_mlp_indices), bias=(up_proj.bias is not None)).to(device=device, dtype=dtype)
    new_up_proj.weight.data.copy_(new_up_weight)
    if up_proj.bias is not None:
        new_up_proj.bias.data.copy_(up_proj.bias.data[kept_mlp_indices])
    mlp.up_proj = new_up_proj

    # 3. down_proj (input features: intermediate_size)
    down_proj = mlp.down_proj
    new_down_weight = down_proj.weight.data[:, kept_mlp_indices]
    new_down_proj = nn.Linear(len(kept_mlp_indices), down_proj.out_features, bias=(down_proj.bias is not None)).to(device=device, dtype=dtype)
    new_down_proj.weight.data.copy_(new_down_weight)
    if down_proj.bias is not None:
        new_down_proj.bias.data.copy_(down_proj.bias.data)
    mlp.down_proj = new_down_proj

# --- Main Pruning Driver ---

def apply_width_pruning(model, samples):
    """Executes the entire activation collection and width pruning pipeline."""
    # Step 1: Collect activation statistics
    collector = ActivationStatsCollector(model)
    collector.register_hooks()
    
    device = get_input_device(model)
    print("Collecting activation statistics on calibration dataset...")
    with torch.no_grad():
        for i, batch in enumerate(iter_batches(samples, BATCH_SIZE)):
            batch = batch.to(device)
            # Run forward pass of the model
            model(batch)
            print(f"  Processed batch {i+1}")
            
    collector.remove_hooks()
    
    # Step 2: Determine target shapes
    config = model.config.text_config if hasattr(model.config, "text_config") else model.config
    target_heads, target_kv_heads = get_valid_gqa_targets(
        config.num_attention_heads,
        config.num_key_value_heads,
        ATTN_HEAD_PRUNE_RATIO
    )
    
    target_intermediate_size = int(round(config.intermediate_size * (1.0 - MLP_PRUNE_RATIO) / 8) * 8)
    target_intermediate_size = max(8, target_intermediate_size)
    
    print(f"\nTarget Attention Query Heads: {target_heads} (original {config.num_attention_heads})")
    print(f"Target Attention KV Heads: {target_kv_heads} (original {config.num_key_value_heads})")
    print(f"Target MLP Intermediate Size: {target_intermediate_size} (original {config.intermediate_size})")
    
    # Step 3: Physically prune layers
    layers = get_decoder_layers(model)
    for layer_idx, layer in enumerate(layers):
        # 1. Prune standard attention heads
        if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "o_proj"):
            scores = collector.attn_stats[layer_idx]
            if scores is not None:
                kept_q, kept_kv = select_heads_to_keep(
                    scores,
                    config.num_attention_heads,
                    config.num_key_value_heads,
                    target_heads,
                    target_kv_heads
                )
                print(f"Layer {layer_idx:02d} standard attention: keeping Q-heads {kept_q}, KV-heads {kept_kv}")
                prune_attn_layer(layer.self_attn, kept_q, kept_kv)
                
        # 2. Prune MLP intermediate dimension
        if hasattr(layer, "mlp") and hasattr(layer.mlp, "down_proj"):
            scores = collector.mlp_stats[layer_idx]
            if scores is not None:
                sorted_indices = torch.argsort(scores, descending=True)
                kept_mlp = sorted_indices[:target_intermediate_size].tolist()
                kept_mlp.sort() # keep original relative order
                prune_mlp_layer(layer.mlp, kept_mlp)
                
    # Step 4: Update global configuration
    if hasattr(model.config, "text_config") and model.config.text_config is not None:
        model.config.text_config.num_attention_heads = target_heads
        model.config.text_config.num_key_value_heads = target_kv_heads
        model.config.text_config.intermediate_size = target_intermediate_size
        
    model.config.num_attention_heads = target_heads
    model.config.num_key_value_heads = target_kv_heads
    model.config.intermediate_size = target_intermediate_size
    
    print("\nModel successfully pruned!")

# --- Main Entry Point ---

def main():
    model, tokenizer = load_model_and_tokenizer(
        MODEL_ID,
        torch_dtype=TORCH_DTYPE,
        device_map=DEVICE_MAP,
        trust_remote_code=TRUST_REMOTE_CODE,
    )
    
    # Run a pre-pruning test generation
    print("\nPre-pruning test generation:")
    device = get_input_device(model)
    inputs = tokenizer("Pruning language models is", return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=10)
    print(f"Generated: {tokenizer.decode(outputs[0], skip_special_tokens=True)}")

    # Load calibration data
    samples = load_c4_calibration(
        tokenizer,
        n_samples=N_CALIBRATION_SAMPLES,
        sequence_length=SEQUENCE_LENGTH,
        seed=RANDOM_SEED,
    )
    
    # Prune
    apply_width_pruning(model, samples)
    
    # Post-pruning test generation
    print("\nPost-pruning test generation:")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=10)
    print(f"Generated: {tokenizer.decode(outputs[0], skip_special_tokens=True)}")
    
    # Save the pruned model and tokenizer
    save_model_and_tokenizer(model, tokenizer, OUTPUT_DIR)

if __name__ == "__main__":
    main()
