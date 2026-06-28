from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model_and_tokenizer(
    model_id,
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=False,
):
    """Load a Hugging Face causal LM and its tokenizer for pruning experiments."""
    print(f"Loading tokenizer: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, trust_remote_code=trust_remote_code
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model: {model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=trust_remote_code,
    )

    model.eval()
    model.config.use_cache = False
    return model, tokenizer


def get_input_device(model):
    """Return the device where input_ids should be placed."""
    return next(model.parameters()).device


def get_decoder_layers(model):
    """Find the repeated decoder/language-model blocks in common causal LMs."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        return model.gpt_neox.layers
    raise ValueError("Could not find decoder layers for this model.")


def get_text_model(model):
    """Return the inner text model from a CausalLM wrapper."""
    if hasattr(model, "model"):
        return model.model
    raise ValueError("Expected the loaded model to have an inner .model text module.")


def load_c4_calibration(
    tokenizer,
    n_samples,
    sequence_length,
    seed=0,
):
    """Sample fixed-length C4 token sequences for calibration.

    Wanda/SparseGPT-style calibration uses unlabeled text only. We use the first C4
    English training shard and cut random contiguous spans from random documents.
    """
    print(
        "Loading C4 calibration data from allenai/c4 first train shard "
        f"(samples={n_samples}, sequence_length={sequence_length})..."
    )
    dataset = load_dataset(
        "allenai/c4",
        data_files={"train": "en/c4-train.00000-of-01024.json.gz"},
        split="train",
    )

    generator = torch.Generator()
    generator.manual_seed(seed)

    samples = []
    attempts = 0
    max_attempts = n_samples * 100
    while len(samples) < n_samples and attempts < max_attempts:
        attempts += 1
        row_idx = torch.randint(0, len(dataset), (1,), generator=generator).item()
        text = dataset[row_idx]["text"]
        input_ids = tokenizer(
            text, return_tensors="pt", add_special_tokens=False
        ).input_ids

        if input_ids.shape[1] < sequence_length:
            continue

        max_start = input_ids.shape[1] - sequence_length
        start = torch.randint(0, max_start + 1, (1,), generator=generator).item()
        samples.append(input_ids[:, start : start + sequence_length])
        print(f"  collected calibration sample {len(samples)}/{n_samples}")

    if len(samples) != n_samples:
        raise RuntimeError(f"Collected {len(samples)} samples, expected {n_samples}.")

    print(
        f"Collected {len(samples)} calibration sequences of length {sequence_length}."
    )
    return samples


def load_hybrid_calibration(
    tokenizer,
    n_samples,
    sequence_length,
    seed=0,
):
    """Sample a 50-50 mix of C4 general English text and GSM8K math word problems.

    GSM8K samples are constructed by concatenating multiple random QA pairs until
    the sequence length is met. C4 samples are cut from random documents.
    """
    print(
        f"Loading 50-50 Hybrid (C4 + GSM8K) calibration data "
        f"(samples={n_samples}, sequence_length={sequence_length})..."
    )

    n_c4 = n_samples // 2
    n_gsm = n_samples - n_c4

    generator = torch.Generator()
    generator.manual_seed(seed)

    # 1. Load C4 dataset
    c4_dataset = load_dataset(
        "allenai/c4",
        data_files={"train": "en/c4-train.00000-of-01024.json.gz"},
        split="train",
    )

    c4_samples = []
    attempts = 0
    max_attempts = n_c4 * 100
    while len(c4_samples) < n_c4 and attempts < max_attempts:
        attempts += 1
        row_idx = torch.randint(0, len(c4_dataset), (1,), generator=generator).item()
        text = c4_dataset[row_idx]["text"]
        input_ids = tokenizer(
            text, return_tensors="pt", add_special_tokens=False
        ).input_ids

        if input_ids.shape[1] < sequence_length:
            continue

        max_start = input_ids.shape[1] - sequence_length
        start = torch.randint(0, max_start + 1, (1,), generator=generator).item()
        c4_samples.append(input_ids[:, start : start + sequence_length])
        print(f"  collected C4 calibration sample {len(c4_samples)}/{n_c4}")

    if len(c4_samples) != n_c4:
        raise RuntimeError(f"Collected {len(c4_samples)} C4 samples, expected {n_c4}.")

    # 2. Load GSM8K dataset
    gsm_dataset = load_dataset("openai/gsm8k", "main", split="train")

    gsm_samples = []
    attempts = 0
    max_attempts = n_gsm * 100
    while len(gsm_samples) < n_gsm and attempts < max_attempts:
        attempts += 1
        current_text = ""
        while True:
            row_idx = torch.randint(
                0, len(gsm_dataset), (1,), generator=generator
            ).item()
            row = gsm_dataset[row_idx]
            current_text += f"Question: {row['question']}\nAnswer: {row['answer']}\n\n"

            input_ids = tokenizer(
                current_text, return_tensors="pt", add_special_tokens=False
            ).input_ids

            if input_ids.shape[1] >= sequence_length:
                max_start = input_ids.shape[1] - sequence_length
                start = torch.randint(
                    0, max_start + 1, (1,), generator=generator
                ).item()
                gsm_samples.append(input_ids[:, start : start + sequence_length])
                print(
                    f"  collected GSM8K calibration sample {len(gsm_samples)}/{n_gsm}"
                )
                break

    if len(gsm_samples) != n_gsm:
        raise RuntimeError(
            f"Collected {len(gsm_samples)} GSM8K samples, expected {n_gsm}."
        )

    # Combine and shuffle
    all_samples = c4_samples + gsm_samples
    indices = torch.randperm(len(all_samples), generator=generator).tolist()
    shuffled_samples = [all_samples[i] for i in indices]

    print(
        f"Collected {len(shuffled_samples)} hybrid calibration sequences of length {sequence_length}."
    )
    return shuffled_samples


def iter_batches(samples, batch_size):
    """Join token sequences into small batches."""
    for start in range(0, len(samples), batch_size):
        yield torch.cat(samples[start : start + batch_size], dim=0)


def save_model_and_tokenizer(model, tokenizer, output_dir):
    """Save model and tokenizer in normal Hugging Face format."""
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving model to {path}")
    model.save_pretrained(path, safe_serialization=True, max_shard_size="2GB")
    tokenizer.save_pretrained(path)


def print_vram_usage(step_name=""):
    """Get memory in bytes and convert to Gigabytes (GiB)."""
    allocated = torch.cuda.memory_allocated() / (1024**3)
    reserved = torch.cuda.memory_reserved() / (1024**3)
    max_allocated = torch.cuda.max_memory_allocated() / (1024**3)

    print(f"\n--- [VRAM Status: {step_name}] ---")
    print(f"  Currently allocated by tensors: {allocated:.2f} GiB")
    print(f"  Reserved by PyTorch (cache):     {reserved:.2f} GiB")
    print(f"  Historical peak (Max Peak):      {max_allocated:.2f} GiB")
    print("-" * 30 + "\n")
