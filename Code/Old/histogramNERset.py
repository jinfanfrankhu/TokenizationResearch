import os
import matplotlib.pyplot as plt
from collections import defaultdict

def count_labels_from_conll(files):
    label_counts = defaultdict(int)

    for file_path in files:
        if not os.path.exists(file_path):
            print(f"Warning: File {file_path} not found.")
            continue

        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("<"):  # skip empty or XML-style lines
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    label = parts[-1]
                    label_counts[label] += 1
    return label_counts

def plot_sorted_histogram(label_counts, lang):
    sorted_items = sorted(label_counts.items(), key=lambda x: x[1], reverse=True)
    labels, counts = zip(*sorted_items)

    plt.figure(figsize=(10, len(labels) * 0.35))
    plt.barh(labels, counts, color='darkcyan')
    plt.xlabel("Frequency (Support)")
    plt.title(f"{lang} NER Label Distribution")
    plt.gca().invert_yaxis()
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    plt.tight_layout()

    plt.savefig(output_folder + fr"\{lang}_NER_label_distribution.png", dpi=300)

# === Example usage ===

turkish_files = [
    r"C:\Users\jinfa\Desktop\Research Dr. Mani\Evaluation\Turkishtest.conll",
    r"C:\Users\jinfa\Desktop\Research Dr. Mani\Evaluation\Turkishtrain.conll"
]

finnish_files = [
    r"C:\Users\jinfa\Desktop\Research Dr. Mani\Evaluation\Finnishtest.conll",
    r"C:\Users\jinfa\Desktop\Research Dr. Mani\Evaluation\Finnishtrain.conll"
]

output_folder = rf"C:\Users\jinfa\Desktop\Research Dr. Mani\NERSetHistograms"

turkish_counts = count_labels_from_conll(turkish_files)
plot_sorted_histogram(turkish_counts, "Turkish")

finnish_counts = count_labels_from_conll(finnish_files)
plot_sorted_histogram(finnish_counts, "Finnish")
