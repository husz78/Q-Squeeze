# Q-Squeeze Wanda

This repo contains a first Wanda implementation for Qwen/Qwen3.5-style causal language models.
Wanda prunes each linear layer with the score from the paper:

```text
S_ij = |W_ij| * ||X_j||_2
```

Scores are compared per output row, and the lowest `sparsity` fraction in each row is set to zero.
The current script prunes every `torch.nn.Linear` inside each decoder layer, including Qwen3.5
linear-attention / Gated DeltaNet projections and MLP projections.

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
