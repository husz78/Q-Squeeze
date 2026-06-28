import math
from collections import defaultdict

import torch
from torch import nn
from transformers.masking_utils import create_causal_mask

from utils import (
    get_decoder_layers,
    get_input_device,
    get_text_model,
    iter_batches,
    load_hybrid_calibration,
    load_model_and_tokenizer,
    print_vram_usage,
    save_model_and_tokenizer,
)


# Edit these constants while experimenting.
# MODEL_ID = "Qwen/Qwen3.5-0.8B"
MODEL_ID = "Qwen/Qwen3.5-4B"
OUTPUT_DIR = "models/qwen3.5-4b-wanda-20"
SPARSITY = 0.20
# N_CALIBRATION_SAMPLES = 4  # increase on entropy/colab to 128.
# SEQUENCE_LENGTH = 64  # increase to 2048 (as in original Wanda paper).
N_CALIBRATION_SAMPLES = 128
SEQUENCE_LENGTH = 2048
RANDOM_SEED = 0
BATCH_SIZE = 1
TORCH_DTYPE = "auto"
DEVICE_MAP = "auto"
TRUST_REMOTE_CODE = False

# It prints the model structure and target layers
# without downloading C4 or changing weights.
PRINT_ONLY = False


def find_target_linears(layer):
    """Return every nn.Linear inside one decoder block.

    In Qwen3.5 this includes both normal attention layers, e.g. self_attn.q_proj,
    and Gated DeltaNet / linear-attention layers, e.g. linear_attn.in_proj_qkv.
    We prune all of them, because otherwise we would leave a large
    part of the Qwen3.5 architecture untouched.
    """
    targets = {}
    for name, module in layer.named_modules():
        if isinstance(module, nn.Linear):
            targets[name] = module
    return targets


def is_wanda_target(full_name, module):
    """Return True if a full model module name points to a pruned decoder Linear.

    We prune every nn.Linear inside model.model.layers.*. This excludes embeddings,
    final norms, and possible output heads outside the repeated decoder stack.
    """
    return isinstance(module, nn.Linear) and full_name.startswith("model.layers.")


def print_model_modules(model):
    """Print the full module tree and mark modules selected for Wanda.

    This is our architecture sanity check. It prints all modules, not only Linear
    layers. A [WANDA] marker means the module is an nn.Linear inside a decoder layer,
    so it will be pruned.
    """
    print("\nFull model module tree:\n")

    for name, module in model.named_modules():
        depth = name.count(".")
        indent = "  " * depth
        display_name = name or "<root>"
        marker = " [WANDA]" if is_wanda_target(name, module) else ""
        weight = ""
        if isinstance(module, nn.Linear):
            weight = f" weight={tuple(module.weight.shape)}"
        print(f"{indent}{display_name}: {module.__class__.__name__}{marker}{weight}")


def make_position_ids(batch_size, sequence_length, device):
    """Create Qwen3.5 position ids in the same shape used by its forward method.

    Qwen3.5 expands positions to shape [4, batch, sequence]. The first slice is used
    as text_position_ids for masks/attention. The remaining three slices are used for
    rotary embeddings. This mirrors the official Qwen3.5 forward method.
    """
    position_ids = torch.arange(sequence_length, device=device)
    return position_ids.view(1, 1, -1).expand(4, batch_size, -1)


class ActivationStats:
    """Accumulate activation statistics needed by Wanda.

    For a Linear layer with input X and weight W, Wanda scores each weight as:

        score_ij = abs(W_ij) * ||X_j||_2

    So for each selected Linear layer, we store sum(X_j^2) across all calibration
    tokens. Later sqrt(sum(X_j^2)) gives the L2 norm ||X_j||_2.
    """

    def __init__(self):
        self.sumsq = defaultdict(lambda: None)

    def add(self, name, activations):
        # A Linear layer receives activations shaped [batch, seq, hidden].
        # Wanda treats batch and sequence positions as one big token axis:
        # [batch, seq, hidden] -> [batch * seq, hidden].
        activations = activations.detach().float().reshape(-1, activations.shape[-1])

        # Sum over tokens, leaving one value per hidden/input channel.
        chunk = torch.sum(activations.pow(2), dim=0).cpu()
        if self.sumsq[name] is None:
            self.sumsq[name] = chunk
        else:
            self.sumsq[name] += chunk

    def l2_norm(self, name):
        """Return ||X_j||_2 for each input channel j of a named Linear layer."""
        return torch.sqrt(self.sumsq[name])


def prepare_manual_forward_states(model, samples):
    """Prepare the initial hidden states for a layer-by-layer manual forward pass.

    The normal Qwen3.5 forward pass does roughly:

        input_ids -> token embeddings -> masks/position embeddings -> decoder layers

    We do the first part once here. After that, apply_wanda() can run one decoder
    layer at a time, prune it, then pass updated hidden states to the next layer.

    Important naming:
        states = list of calibration batches.
        state  = runtime data for one calibration batch.

    If N_CALIBRATION_SAMPLES=4 and BATCH_SIZE=1, then states has 4 items.
    If N_CALIBRATION_SAMPLES=4 and BATCH_SIZE=4, then states has 1 item.
    """
    text_model = get_text_model(model)
    device = get_input_device(model)
    states = []

    with torch.no_grad():
        for input_ids in iter_batches(samples, BATCH_SIZE):
            # This loop creates one state per calibration batch.
            # input_ids has shape [batch, sequence_length].
            input_ids = input_ids.to(device)

            # Our calibration spans have no padding, so every token is valid.
            attention_mask = torch.ones_like(input_ids, device=device)

            # Convert token IDs to vectors. This is the first hidden_states value
            # that will flow through layer 0, then layer 1, and so on.
            hidden_states = text_model.embed_tokens(input_ids)

            # Qwen3.5 needs position metadata in addition to token vectors.
            position_ids = make_position_ids(
                batch_size=hidden_states.shape[0],
                sequence_length=hidden_states.shape[1],
                device=hidden_states.device,
            )
            text_position_ids = position_ids[0]
            rotary_position_ids = position_ids[1:]

            causal_mask = create_causal_mask(
                config=text_model.config,
                inputs_embeds=hidden_states,
                attention_mask=attention_mask,
                past_key_values=None,
                position_ids=text_position_ids,
            )
            linear_attn_mask = text_model._update_linear_attn_mask(
                attention_mask, past_key_values=None
            )
            position_embeddings = text_model.rotary_emb(
                hidden_states, rotary_position_ids
            )

            # This dictionary is one batch's "travel pack" through the decoder stack.
            # Only hidden_states changes after each pruned layer. Masks and positions
            # stay fixed because token order and padding do not change.
            states.append(
                {
                    "hidden_states": hidden_states,
                    "position_embeddings": position_embeddings,
                    "text_position_ids": text_position_ids,
                    "causal_mask": causal_mask,
                    "linear_attn_mask": linear_attn_mask,
                }
            )

    return states


def get_layer_attention_mask(model, layer_idx, state):
    """Choose the correct attention mask for one Qwen3.5 decoder layer.

    Qwen3.5 alternates between linear-attention/DeltaNet layers and full-attention
    layers. Linear-attention layers use linear_attn_mask; full-attention layers use
    causal_mask.
    """
    text_model = get_text_model(model)
    if text_model.config.layer_types[layer_idx] == "linear_attention":
        return state["linear_attn_mask"]
    return state["causal_mask"]


def run_one_decoder_layer(model, layer_idx, layer, state):
    """Run exactly one decoder layer on one calibration batch state."""
    return layer(
        state["hidden_states"],
        position_embeddings=state["position_embeddings"],
        attention_mask=get_layer_attention_mask(model, layer_idx, state),
        position_ids=state["text_position_ids"],
        past_key_values=None,
        use_cache=False,
    )


def collect_layer_activation_stats(model, layer_idx, layer, target_linears, states):
    """Capture Linear-layer inputs while running only one decoder layer.

    PyTorch forward hooks let us observe a module while the normal model forward pass
    is running. We attach a hook to each target Linear layer. Every time that Linear
    runs, the hook receives its input activation X and adds X^2 to ActivationStats.
    """
    stats = ActivationStats()
    handles = []

    for name, module in target_linears.items():
        # key=name freezes the current loop value of name inside the lambda.
        # Without it, Python closures would make all hooks use the last name.
        handle = module.register_forward_hook(
            lambda module, inputs, output, key=name: stats.add(key, inputs[0])
        )
        handles.append(handle)

    try:
        with torch.no_grad():
            for state in states:
                # Run the current layer once for each calibration batch.
                # Forward hooks attached above collect Linear inputs into stats.
                # We ignore the output here because this pass is only for measuring
                # activations before pruning the current layer.
                run_one_decoder_layer(model, layer_idx, layer, state)
    finally:
        for handle in handles:
            handle.remove()

    return stats


def update_states_after_pruning(model, layer_idx, layer, states):
    """Recompute current layer after pruning and store its output for the next layer.

    We collect stats using the layer's pre-pruning weights. Then we zero low-score
    weights. To make the next decoder layer receive updated activations, we run the
    pruned layer once and replace state["hidden_states"] with its output.
    """
    with torch.no_grad():
        for state in states:
            # Now we DO keep the output. This output was produced by the pruned
            # current layer, so it becomes the input hidden_states for the next layer.
            state["hidden_states"] = run_one_decoder_layer(
                model, layer_idx, layer, state
            )


@torch.no_grad()
def prune_linear_with_wanda(linear, activation_l2):
    """Apply Wanda to one Linear matrix.

    For weight W with shape [out_features, in_features], score each weight as
    abs(W_ij) * ||X_j||_2. Then prune the lowest scores separately in each row.
    """
    weight = linear.weight
    out_features, in_features = weight.shape

    # Wanda prunes the same fraction in every output row. For example, if a row has
    # 1024 input weights and SPARSITY=0.20, we zero 204 weights from that row.
    n_prune = int(in_features * SPARSITY)
    if n_prune == 0:
        return 0

    # Put activation norms next to the weights and compute scores in float32 even if
    # the model itself is loaded in float16/bfloat16/auto.
    activation_l2 = activation_l2.to(device=weight.device, dtype=torch.float32)
    scores = weight.detach().float().abs() * activation_l2.reshape(1, -1)

    # For each output row, choose the column indices with the smallest Wanda scores.
    prune_indices = torch.topk(scores, k=n_prune, dim=1, largest=False).indices

    # True keeps a weight, False zeroes it.
    mask = torch.ones_like(weight, dtype=torch.bool)
    mask.scatter_(dim=1, index=prune_indices, value=False)
    weight.mul_(mask.to(dtype=weight.dtype))
    return out_features * n_prune


def count_target_sparsity(model):
    """Count zeros only in decoder Linear matrices that Wanda targets."""
    zeros = 0
    total = 0
    for layer in get_decoder_layers(model):
        for linear in find_target_linears(layer).values():
            weight = linear.weight.detach()
            zeros += torch.count_nonzero(weight == 0).item()
            total += weight.numel()
    return zeros, total, zeros / total if total else math.nan


def apply_wanda(model, samples):
    """Prune the model block by block with a manual layer-by-layer forward pass.

    After one block is pruned, later blocks see activations produced by the already
    pruned earlier blocks. This matches the sequential procedure described in Wanda.

    High-level flow:

        1. Embed calibration tokens once.
        2. For decoder layer 0:
           collect stats -> prune layer 0 -> update hidden states.
        3. For decoder layer 1:
           use updated hidden states -> collect stats -> prune -> update.
        4. Continue to the final layer.

    So it is not one call to model.forward(); it is one manual pass through the decoder
    stack, with a second local run of each layer after pruning to propagate changes.
    """
    # Build one state per calibration batch. Each state starts at the embedding output,
    # before decoder layer 0 has run.
    states = prepare_manual_forward_states(model, samples)
    total_pruned = 0
    for layer_idx, layer in enumerate(get_decoder_layers(model)):
        target_linears = find_target_linears(layer)
        if not target_linears:
            continue

        print(
            f"\nLayer {layer_idx:02d}: collecting activations "
            f"for {len(target_linears)} Linear modules..."
        )
        stats = collect_layer_activation_stats(
            model, layer_idx, layer, target_linears, states
        )

        # Now that we know ||X_j||_2 for this layer's Linear inputs, we can compute
        # Wanda scores and zero the lowest-scoring weights.
        for name, linear in target_linears.items():
            pruned = prune_linear_with_wanda(linear, stats.l2_norm(name))
            total_pruned += pruned
            print(f"  pruned {pruned:,} weights in {name}")

        # Push every calibration batch through the newly pruned layer so the next
        # decoder layer receives updated activations (as in original paper).
        update_states_after_pruning(model, layer_idx, layer, states)
        if layer_idx % 4 == 0:
            print_vram_usage(f"after layer {layer_idx:02d}")

    return total_pruned


def main():
    model, tokenizer = load_model_and_tokenizer(
        MODEL_ID,
        torch_dtype=TORCH_DTYPE,
        device_map=DEVICE_MAP,
        trust_remote_code=TRUST_REMOTE_CODE,
    )
    print_vram_usage("after model load")
    print_model_modules(model)

    if PRINT_ONLY:
        print("PRINT_ONLY=True, so we stop after printing modules.")
        return

    samples = load_hybrid_calibration(
        tokenizer,
        n_samples=N_CALIBRATION_SAMPLES,
        sequence_length=SEQUENCE_LENGTH,
        seed=RANDOM_SEED,
    )
    print_vram_usage("after calibration load")
    before = count_target_sparsity(model)
    print(f"\nTarget sparsity before: {before[0]:,}/{before[1]:,} = {before[2]:.2%}")

    total_pruned = apply_wanda(model, samples)

    after = count_target_sparsity(model)
    print(f"\nRequested row sparsity: {SPARSITY:.2%}")
    print(f"Newly pruned weights: {total_pruned:,}")
    print(f"Target sparsity after: {after[0]:,}/{after[1]:,} = {after[2]:.2%}")

    print_vram_usage("before saving")
    save_model_and_tokenizer(model, tokenizer, OUTPUT_DIR)
    print_vram_usage("after saving")


if __name__ == "__main__":
    main()
