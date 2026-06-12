import torch
import time 

from torch import nn
import torch.nn.functional as F

from utils import (
    load_model_and_tokenizer,
    load_c4_calibration
)

MODEL_ID = "Qwen/Qwen3.5-4B"
# OUTPUT_DIR = "models/qwen-wanda-smoke"
N_CALIBRATION_SAMPLES = 128  # increase on entropy/colab to 128.
SEQUENCE_LENGTH = 1024  # increase to 2048 (as in original Wanda paper).
RANDOM_SEED = 0
BATCH_SIZE = 1
CHUNK_SIZE = 1
TORCH_DTYPE = "auto"
DEVICE_MAP = "auto"
TRUST_REMOTE_CODE = False

# It prints the model structure and target layers
# without downloading C4 or changing weights.
PRINT_ONLY = False


def calculate_all_blocks_BI(model, samples: list[torch.Tensor], chunk_size: int = 16) -> list[float]:
    """Calculates the Block Influence (BI) for all decoder blocks in the model.

    Args:
        model: The language model.
        samples (list[torch.Tensor]): A list of tokenized input samples for calibration.

    Returns:
        list[float]: A list of BI scores for each block.
    """
    device = next(model.parameters()).device

    tokens_batch = torch.cat(samples, dim=0).to(device)

    with torch.no_grad():
        prev_hidden_states = model.model.embed_tokens(tokens_batch)

        seq_length = tokens_batch.size(1)
        position_ids = torch.arange(0, seq_length, dtype=torch.long, device=device).unsqueeze(0)

        # Extract cosines and sines for RoPE from Qwen's dedicated module
        # (1, 2048, 128) - seq_lenght = 2048, head_dim = 128
        position_embeddings = model.model.rotary_emb(prev_hidden_states, position_ids)

    results = []

    for idx, block in enumerate(get_blocks(model)):
        print(f"Przetwarzam blok {idx}...")

        new_hidden_states = evaluate_block(block, prev_hidden_states, position_embeddings, chunk_size=chunk_size)

        BI = calculate_BI(prev_hidden_states, new_hidden_states)

        results.append(BI)

        prev_hidden_states = new_hidden_states

    return results

def calculate_BI(input_states: torch.Tensor, output_states: torch.Tensor) -> float:
    """Calculates the Block Influence (BI) according to the ShortGPT formula.

    Args:
        input_states (torch.Tensor): The input hidden states to a block.
        output_states (torch.Tensor): The output hidden states from a block.

    Returns:
        float: The calculated Block Influence (BI) score.
    """
    print_vram_usage("Zaczynamy calculate BI")
    with torch.no_grad():
        input_flat = input_states.view(-1, input_states.size(-1)).to(torch.float32)
        print_vram_usage("Po input flat (calculate BI)")
        output_flat = output_states.view(-1, output_states.size(-1)).to(torch.float32)
        print_vram_usage("Po output flat (calculate BI)")

        cos_sim = F.cosine_similarity(input_flat, output_flat, dim=-1)

        BI = 1.0 - cos_sim.mean().item()

    return BI

def get_blocks(model) -> nn.ModuleList:
    """Retrieves the decoder layers (blocks) from the model.

    Args:
        model: The language model.

    Returns:
        torch.nn.ModuleList: A list of decoder layers.

    Raises:
        ValueError: If decoder layers are not found in the model.
    """
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    raise ValueError("Decoder layers not found.")

def evaluate_block(block, hidden_states: torch.Tensor, position_embeddings: tuple, chunk_size) -> torch.Tensor:
    total_size = hidden_states.size(0)
    
    # Empty tensor for memory optimization
    final_output = torch.empty_like(hidden_states)
    with torch.no_grad():
        for start_idx in range(0, total_size, chunk_size):
            print_vram_usage(f"Chunk start: {start_idx}")
            end_idx = min(start_idx + chunk_size, total_size)
            
            input_chunk = hidden_states[start_idx:end_idx]
            
            out_chunk = block(input_chunk, position_embeddings=position_embeddings)
            
            final_output[start_idx:end_idx] = out_chunk

    return final_output


def save_results(results: list[float]):
    with open("BI_results.txt", "w", encoding="utf-8") as plik:
        for wynik in results:
            plik.write(f"{wynik:.6f}\n")


def print_vram_usage(step_name=""):
    # Pobieramy pamięć w bajtach i przeliczamy na Gigabajty (GiB)
    allocated = torch.cuda.memory_allocated() / (1024 ** 3)
    reserved = torch.cuda.memory_reserved() / (1024 ** 3)
    max_allocated = torch.cuda.max_memory_allocated() / (1024 ** 3)
    
    print(f"\n--- [VRAM Status: {step_name}] ---")
    print(f"  Aktualnie alokowane przez tensory: {allocated:.2f} GiB")
    print(f"  Zarezerwowane przez PyTorch (cache): {reserved:.2f} GiB")
    print(f"  Historyczny szczyt (Max Peak):       {max_allocated:.2f} GiB")
    print("-" * 30 + "\n")

def main():
    print_vram_usage("Start skryptu")
    model, tokenizer = load_model_and_tokenizer(
        MODEL_ID,
        torch_dtype=TORCH_DTYPE,
        device_map=DEVICE_MAP,
        trust_remote_code=TRUST_REMOTE_CODE,
    )
    print_vram_usage("Po załadowaniu modelu")

    print(f"Model jest na urządzeniu: {next(model.parameters()).device}")

    if PRINT_ONLY:
        print("PRINT_ONLY=True, so we stop after printing modules.")
        return

    samples = load_c4_calibration(
        tokenizer,
        n_samples=N_CALIBRATION_SAMPLES,
        sequence_length=SEQUENCE_LENGTH,
        seed=RANDOM_SEED,
    )

    print_vram_usage("Po załadowaniu danych C4")

    start = time.perf_counter()

    BI_results = calculate_all_blocks_BI(
        model=model,
        samples=samples,
        chunk_size=CHUNK_SIZE
    )

    end = time.perf_counter()

    print(BI_results)
    print(f"Czas wykonania: {end - start:.2f} sekund")

    save_results(BI_results)

    # save_model_and_tokenizer(model, tokenizer, OUTPUT_DIR)


if __name__ == "__main__":
    main()