results = {
    "Qwen3.5-4B": [0.86328125, 0.07421875, 0.171875, 0.13671875, 0.10546875, 0.12890625, 0.1640625, 0.1328125, 0.1015625, 0.08984375, 0.08984375, 0.0859375, 0.0703125, 0.05859375, 0.07421875, 0.07421875, 0.06640625, 0.07421875, 0.1171875, 0.12890625, 0.08203125, 0.0703125, 0.08203125, 0.0703125, 0.05859375, 0.046875, 0.0546875, 0.0703125, 0.05078125, 0.05078125, 0.06640625, 0.3203125],
    "Qwen3.5-0.8B": [0.8837890625, 0.171875, 0.1953125, 0.1640625, 0.14453125, 0.125, 0.12890625, 0.109375, 0.07421875, 0.06640625, 0.08203125, 0.078125, 0.06640625, 0.07421875, 0.1171875, 0.12890625, 0.0859375, 0.0625, 0.06640625, 0.07421875, 0.0546875, 0.0390625, 0.06640625, 0.34375]
}

import numpy as np
import matplotlib.pyplot as plt

def get_attention_layers(n):
    # Qwen3.5-4B hybrid: co 4 warstwa attention
    return set(range(3, n, 4))


def plot_bi(values, title, save_path):

    values = np.array(values)
    n = len(values)
    x = np.arange(n)

    att_layers = get_attention_layers(n)

    plt.figure(figsize=(12, 4.5), dpi=300)

    # gradient bars
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, n))

    bars = plt.bar(x, values, color=colors, edgecolor="none")

    # highlight attention blocks (subtle background bands)
    for i in range(n):
        if i in att_layers:
            plt.axvspan(i - 0.5, i + 0.5, color="red", alpha=0.08)

    # emphasize first & last layer
    bars[0].set_edgecolor("black")
    bars[0].set_linewidth(1.2)

    bars[-1].set_edgecolor("black")
    bars[-1].set_linewidth(1.2)

    plt.title(title, fontsize=13)
    plt.xlabel("Block Index", fontsize=11)
    plt.ylabel("BI Score", fontsize=11)

    # reduce x-axis clutter
    step = 2 if n > 20 else 1
    plt.xticks(x[::step])

    plt.grid(axis="y", linestyle="--", alpha=0.3)

    plt.tight_layout()

    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


# run
plot_bi(results["Qwen3.5-4B"], "Qwen3.5-4B BI per Block", "qwen4B_bi.pdf")
plot_bi(results["Qwen3.5-0.8B"], "Qwen3.5-0.8B BI per Block", "qwen0_8B_bi.pdf")