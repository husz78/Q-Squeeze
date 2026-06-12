# Q-Squeeze

This repository contains pruning implementations for Qwen/Qwen3.5-style causal language models, supporting both unstructured and structured pruning techniques:

1. **Wanda (Unstructured Pruning)**: Prunes individual weights in linear layers based on weight magnitudes and input activation norms:
   $$S_{ij} = |W_{ij}| \times \|X_j\|_2$$
   It zeros out the lowest scoring weights under a target sparsity constraint without changing the physical shape of the tensors.
2. **Structured Width Pruning**: Physically prunes entire attention heads (under GQA constraints) and MLP intermediate dimensions to reduce parameters, save VRAM, and speed up inference.

---

## Repository Structure

- `wanda.py`: Wanda-specific scoring, hooks, masks, and unstructured pruning logic.
- `width_pruning/layer_pruning.py`: Structured width pruning logic, including activation collection, GQA alignment, and physical weight slicing.
- `utils.py`: Shared utilities for model/tokenizer loading, C4 calibration data sampling, batching, and model saving.


## What to Download

Install Python dependencies:

```bash
pip install -r requirements.txt
```

The script downloads these through Hugging Face when first run:

- the target model, for example your Qwen 3.5 8B checkpoint;
- tokenizer files for that model;
- C4 English calibration data from `allenai/c4`.

For gated/private models, log in first:

```bash
huggingface-cli login
```

## Calibration Data

We follow the Wanda/SparseGPT setup: 128 calibration sequences sampled from the C4 training set.
By default, `wanda.py` samples from the first C4 training shard:

```text
allenai/c4: en/c4-train.00000-of-01024.json.gz
```

Use the model context length you want to test. For a first local run, `2048` is cheaper; on the final
cluster run, use the context length agreed for the report. If the first-shard download causes problems
on the cluster, use `--calib-source streaming-c4` as a practical fallback and document the difference.

## Dry Run

Edit the constants at the top of `wanda.py`. Keep this setting for the first run:

```python
PRINT_ONLY = True
```

Then run:

```bash
python wanda.py
```

This loads the model and prints every decoder layer with its Linear matrices. Every printed matrix is
selected for Wanda pruning.

## Prune

After checking the printed layers, set:

```python
PRINT_ONLY = False
```

For a local smoke test, keep:

```python
MODEL_ID = "Qwen/Qwen3.5-0.8B"
OUTPUT_DIR = "models/qwen-wanda-20"
SPARSITY = 0.20
N_CALIBRATION_SAMPLES = 128
SEQUENCE_LENGTH = 2048
TORCH_DTYPE = "auto"
DEVICE_MAP = "auto"
```

Then run:

```bash
python wanda.py
```

For the final run, replace `MODEL_ID` with the Qwen 3.5 8B checkpoint and choose a matching `OUTPUT_DIR`.
Repeat for `0.10`, `0.20`, `0.30`, `0.40`, and `0.50` sparsity to build the report curve.

## Evaluation

Start with perplexity or a small `lm-eval` task before full MMLU/GSM8K:

```bash
lm_eval --model hf \
  --model_args pretrained=models/qwen3.5-8b-wanda-20 \
  --tasks wikitext \
  --device cuda:0 \
  --batch_size auto
```

Then run the project tasks:

```bash
lm_eval --model hf \
  --model_args pretrained=models/qwen3.5-8b-wanda-20 \
  --tasks gsm8k,mmlu \
  --device cuda:0 \
  --batch_size auto
```

## Structured Width Pruning

In addition to Wanda, this repository supports structured width pruning of Multi-Head Attention query/key/value heads and MLP intermediate dimensions in `width_pruning/layer_pruning.py`.

Unlike unstructured pruning, structured width pruning physically slices the weight matrices, resulting in smaller files, reduced VRAM footprint, and faster inference on any standard CPU or GPU.

### How it works:
1. **Activation-based Scoring**: Forward hooks record the L2 activation norms of query heads (inputs of `o_proj`) and MLP neurons (inputs of `down_proj`) over a C4 calibration dataset.
2. **GQA Alignment**: Automatically resolves the target number of query/KV heads under GQA divisibility constraints and ranks/selects the most active heads.
3. **Physical Slicing**: Replaces the layer modules with physically smaller `nn.Linear` layers and updates the global Hugging Face configuration.

### How to Run:
Configure `ATTN_HEAD_PRUNE_RATIO` (e.g. `0.25`) and `MLP_PRUNE_RATIO` (e.g. `0.20`) at the top of `width_pruning/layer_pruning.py`, then run:
```bash
python width_pruning/layer_pruning.py
```
The pruned model will be saved in `models/qwen-width-pruned` and can be loaded out-of-the-box using `AutoModelForCausalLM.from_pretrained()`.

