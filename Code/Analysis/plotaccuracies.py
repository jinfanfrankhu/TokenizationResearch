import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from adjustText import adjust_text 
from Code.metasettings import RUNNUMBER, LANGS, STRATEGIES

# Function to load classification data from a JSON file
def load_classification_data(file_path):
    if not os.path.exists(file_path):
        print(f"Warning: File {file_path} not found. Skipping.")
        return None
    with open(file_path, "r") as f:
        data = json.load(f)
    return data

# Function to save plots
def save_plot(fig, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)  # Ensure directory exists
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)  # Close figure to free memory

# Function to plot and save class-wise F1-scores
def plot_f1_scores(data, lang, strat, save_dir):
    classification_report = data["classification_report"]
    
    labels = [label for label in classification_report.keys() if label not in ["accuracy", "macro avg", "weighted avg"]]
    f1_scores = [classification_report[label]["f1-score"] for label in labels]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(labels, f1_scores, color='blue')
    ax.set_xlabel("F1-Score")
    ax.set_ylabel("Class Labels")
    ax.set_title(f"{lang} {strat} Class-wise F1-Scores")
    ax.set_xlim(0, 1)
    ax.invert_yaxis()

    save_plot(fig, os.path.join(save_dir, "f1_scores.png"))

# Function to plot and save precision vs recall scatter plot
def plot_precision_vs_recall(data, lang, strat, save_dir):
    classification_report = data["classification_report"]

    # Only include labels that are not summary averages and that have non-zero precision or recall
    labels = []
    precision = []
    recall = []
    for label, scores in classification_report.items():
        if label in ["accuracy", "macro avg", "weighted avg"]:
            continue
        if scores["precision"] > 0 or scores["recall"] > 0:
            labels.append(label)
            precision.append(scores["precision"])
            recall.append(scores["recall"])

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(precision, recall, color='red', s=100, alpha=0.7)

    texts = []
    for i, label in enumerate(labels):
        texts.append(ax.text(precision[i], recall[i], label, fontsize=9))
    ax.scatter(0, 0, color="black", s=100, alpha=0.7)
    ax.text(0.02, 0.02, "Failed Classes", fontsize=9)

    adjust_text(texts, arrowprops=dict(arrowstyle="->", color='gray', lw=0.5))

    ax.set_xlabel("Precision")
    ax.set_ylabel("Recall")
    ax.set_title(f"{lang} {strat} Precision vs Recall")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True)

    save_plot(fig, os.path.join(save_dir, "precision_recall.png"))

# Function to plot and save confusion matrix heatmap
def plot_confusion_matrix(data, lang, strat, save_dir):
    confusion_matrix = np.array(data["confusion_matrix"])
    labels = data["labels"]

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted Labels")
    ax.set_ylabel("True Labels")
    ax.set_title(f"{lang} {strat} Confusion Matrix Heatmap")

    save_plot(fig, os.path.join(save_dir, "confusion_matrix.png"))

# Compare across strategies by tag
def plot_tag_comparison_across_strategies(lang, all_data, save_dir_base):
    """
    For a given language, creates one bar plot per NER tag, showing F1-scores across strategies.

    Parameters:
        lang (str): The language to process.
        all_data (dict): Dictionary of {strategy: loaded_json_data}.
        save_dir_base (str): Base directory to save plots.
    """
    # Collect all unique labels (NER tags) across all strategies
    all_labels = set()
    for data in all_data.values():
        all_labels.update([
            label for label in data["classification_report"].keys()
            if label not in ["accuracy", "macro avg", "weighted avg"]
        ])

    for label in sorted(all_labels):
        f1_scores = []
        used_strategies = []

        for strategy, data in all_data.items():
            report = data["classification_report"]
            if label in report:
                f1 = report[label]["f1-score"]
                f1_scores.append(f1)
                used_strategies.append(strategy)

        if not f1_scores:
            continue  # Skip this label if it's missing from all strategies

        # Sort strategies by F1-score
        sorted_pairs = sorted(zip(f1_scores, used_strategies), reverse=True)
        sorted_scores, sorted_strategies = zip(*sorted_pairs)

        # Plot
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.barh(sorted_strategies, sorted_scores, color='green')
        ax.set_xlabel("F1-Score")
        ax.set_title(f"{lang} - {label} F1 Score by Strategy")
        ax.set_xlim(0, 1)
        ax.invert_yaxis()

        tag_safe = label.replace("/", "_").replace("\\", "_").replace(" ", "_")
        os.makedirs(save_dir_base, exist_ok=True)
        save_plot(fig, os.path.join(save_dir_base, f"{tag_safe}_f1_by_strategy.png"))

if __name__ == "__main__":
    verbose = 1
    for lang in LANGS:
        all_data = {}
        for strategy in STRATEGIES:
            file_path = fr"C:\Users\jinfa\Desktop\Research Dr. Mani\{lang} Run {RUNNUMBER}\{lang} Evaluation\{lang}_{strategy}_NER_results.json"
            data = load_classification_data(file_path)
            if data:
                all_data[strategy] = data
                save_dir = os.path.join(rf"C:\Users\jinfa\Desktop\Research Dr. Mani\{lang} Run {RUNNUMBER}\{lang} Evaluation\Plots", strategy)
                plot_f1_scores(data, lang, strategy, save_dir)
                plot_precision_vs_recall(data, lang, strategy, save_dir)
                # plot_confusion_matrix(data, lang, strategy, save_dir)
                if verbose:
                    print(f"Saved plots for {lang} - {strategy} in {save_dir}")
        # Now call the new function to compare F1 scores per tag
        plot_tag_comparison_across_strategies(lang, all_data, rf"C:\Users\jinfa\Desktop\Research Dr. Mani\{lang} Run {RUNNUMBER}\{lang} Evaluation\F1 By Strategy")

