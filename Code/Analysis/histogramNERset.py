import os
import matplotlib.pyplot as plt
from collections import defaultdict
from Code.metasettings import LANGS

output_folder = r"C:\Users\jinfa\Desktop\Research Dr. Mani\NER Histograms"

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
    os.makedirs(output_folder, exist_ok=True)

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
if __name__ == "__main__":
    for lang in LANGS:
        print(f"Processing {lang} NER files...")
        files = [rf"C:\Users\jinfa\Desktop\Research Dr. Mani\NERSets\{lang}.conll"]

        label_counts = count_labels_from_conll(files)
        plot_sorted_histogram(label_counts, lang)