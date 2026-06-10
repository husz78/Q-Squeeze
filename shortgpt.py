import torch

from torch import nn
import torch.nn.functional as F

from utils import (
    load_model_and_tokenizer,
    load_c4_calibration
)

MODEL_ID = "Qwen/Qwen3.5-0.8B"
# OUTPUT_DIR = "models/qwen-wanda-smoke"
N_CALIBRATION_SAMPLES = 128  # increase on entropy/colab to 128.
SEQUENCE_LENGTH = 2048  # increase to 2048 (as in original Wanda paper).
RANDOM_SEED = 0
BATCH_SIZE = 1
CHUNK_SIZE = 16
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
    with torch.no_grad():
        input_flat = input_states.view(-1, input_states.size(-1)).to(torch.float32)
        output_flat = output_states.view(-1, output_states.size(-1)).to(torch.float32)

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
    """Passes hidden states through a single Qwen block in smaller chunks
    to avoid CUDA Out of Memory errors.

    Args:
        block: The Qwen decoder layer to evaluate.
        hidden_states (torch.Tensor): The input hidden states to the block.
        position_embeddings (tuple): The positional embeddings required by the Qwen block.
        chunk_size (int): The size of chunks to process to avoid CUDA Out of Memory errors.

    Returns:
        torch.Tensor: The output hidden states from the block.
    """
    total_samples = hidden_states.size(0)
    output_chunks = []

    # Split the large batch into chunks
    for i in range(0, total_samples, chunk_size):
        chunk_hidden = hidden_states[i : i + chunk_size]

        with torch.no_grad():
            outputs = block(chunk_hidden, position_embeddings=position_embeddings)
            output_chunks.append(outputs)

    # Concatenate the results of the small chunks back into one large tensor
    return torch.cat(output_chunks, dim=0)

def save_results(results: list[float]):
    with open("BI_results.txt", "w", encoding="utf-8") as plik:
        for wynik in results:
            plik.write(f"{wynik:.6f}\n")

def main():
    model, tokenizer = load_model_and_tokenizer(
        MODEL_ID,
        torch_dtype=TORCH_DTYPE,
        device_map=DEVICE_MAP,
        trust_remote_code=TRUST_REMOTE_CODE,
    )

    if PRINT_ONLY:
        print("PRINT_ONLY=True, so we stop after printing modules.")
        return

    samples = load_c4_calibration(
        tokenizer,
        n_samples=N_CALIBRATION_SAMPLES,
        sequence_length=SEQUENCE_LENGTH,
        seed=RANDOM_SEED,
    )

    BI_results = calculate_all_blocks_BI(
        model=model,
        samples=samples,
        chunk_size=CHUNK_SIZE
    )

    print(BI_results)

    save_results(BI_results)

    # save_model_and_tokenizer(model, tokenizer, OUTPUT_DIR)


if __name__ == "__main__":
    main()