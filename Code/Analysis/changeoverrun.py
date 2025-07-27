from Code.metasettings import RUNNUMBER, LANGS, STRATEGIES
import os
import json
import csv
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def compare_classification_reports(old_json_path, new_json_path):
    with open(old_json_path, 'r', encoding='utf-8') as f:
        old_data = json.load(f)
    with open(new_json_path, 'r', encoding='utf-8') as f:
        new_data = json.load(f)

    old_report = old_data["classification_report"]
    new_report = new_data["classification_report"]

    entity_changes = {}

    for tag in old_report:
        # Skip overall metrics like "accuracy", "macro avg", "weighted avg"
        if tag in ["accuracy", "macro avg", "weighted avg"]:
            continue

        old_scores = old_report[tag]
        new_scores = new_report.get(tag, {"precision": 0.0, "recall": 0.0, "f1-score": 0.0, "support": 0.0})

        entity_changes[tag] = {
            "precision_change": new_scores["precision"] - old_scores["precision"],
            "recall_change": new_scores["recall"] - old_scores["recall"],
            "f1_change": new_scores["f1-score"] - old_scores["f1-score"]
        }

    return entity_changes

def save_changes_to_csv(changes, output_path):
    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['Entity', 'Precision Change', 'Recall Change', 'F1 Change']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for entity, scores in changes.items():
            writer.writerow({
                'Entity': entity,
                'Precision Change': scores['precision_change'],
                'Recall Change': scores['recall_change'],
                'F1 Change': scores['f1_change']
            })

def generate_language_specific_plots(df, lang, output_base_dir):
    lang_df = df[df["Language"] == lang]

    # 1. F1 Change by Strategy (Bar Plot)
    plt.figure(figsize=(8,5))
    avg_f1_strategy = lang_df.groupby("Strategy")["F1 Change"].mean().sort_values(ascending=False)
    plt.bar(avg_f1_strategy.index, avg_f1_strategy.values)
    plt.title(f"{lang}: Avg F1 Change by Strategy")
    plt.xlabel("Strategy")
    plt.ylabel("F1 Change")
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_base_dir, f"{lang}_f1_change_by_strategy.png"), dpi=300)
    plt.close()

    # 2. Heatmap: F1 Change by Entity × Strategy
    plt.figure(figsize=(10,8))
    heatmap_data = lang_df.pivot_table(index="Entity", columns="Strategy", values="F1 Change", aggfunc="mean")
    sns.heatmap(heatmap_data, annot=True, fmt=".3f", cmap="coolwarm", linewidths=0.5)
    plt.title(f"{lang}: F1 Change Heatmap (Entity x Strategy)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_base_dir, f"{lang}_f1_change_heatmap.png"), dpi=300)
    plt.close()

    # 3. Top 5 Improved Entities
    plt.figure(figsize=(8,5))
    top5 = lang_df.groupby("Entity")["F1 Change"].mean().sort_values(ascending=False).head(5)
    plt.barh(top5.index[::-1], top5.values[::-1])
    plt.title(f"{lang}: Top 5 Improved Entities")
    plt.xlabel("F1 Change")
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_base_dir, f"{lang}_top5_entities_improved.png"), dpi=300)
    plt.close()

    # 4. Top 5 Degraded Entities
    plt.figure(figsize=(8,5))
    bottom5 = lang_df.groupby("Entity")["F1 Change"].mean().sort_values(ascending=True).head(5)
    plt.barh(bottom5.index[::-1], bottom5.values[::-1])
    plt.title(f"{lang}: Top 5 Degraded Entities")
    plt.xlabel("F1 Change")
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_base_dir, f"{lang}_top5_entities_degraded.png"), dpi=300)
    plt.close()

    print(f"Saved language-specific plots to {output_base_dir}")

if __name__ == "__main__":
    all_changes = []
    for lang in LANGS:
        for strategy in STRATEGIES:
            # Check if old file exists
            if os.path.exists(f"C:\\Users\\jinfa\\Desktop\\Research Dr. Mani\\{lang} Run {RUNNUMBER}\\{lang} Evaluation\\{lang}_{strategy}_NER_results.json") and os.path.exists(f"C:\\Users\\jinfa\\Desktop\\Research Dr. Mani\\{lang} Run {RUNNUMBER-1}\\{lang} Evaluation\\{lang}_{strategy}_POS_results.json"):
                old_json_path = f"C:\\Users\\jinfa\\Desktop\\Research Dr. Mani\\{lang} Run {RUNNUMBER-1}\\{lang} Evaluation\\{lang}_{strategy}_POS_results.json"
                new_json_path = f"C:\\Users\\jinfa\\Desktop\\Research Dr. Mani\\{lang} Run {RUNNUMBER}\\{lang} Evaluation\\{lang}_{strategy}_NER_results.json"
            else:
                continue

            changes = compare_classification_reports(old_json_path, new_json_path)
            output_csv_path = f"C:\\Users\\jinfa\\Desktop\\Research Dr. Mani\\{lang} Run {RUNNUMBER}\\{lang} Changes\\{strategy}.csv"
            os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
            save_changes_to_csv(changes, output_csv_path)

            df = pd.read_csv(output_csv_path)
            df["Language"] = lang
            df["Strategy"] = strategy
            all_changes.append(df)

    full_df = pd.concat(all_changes, ignore_index=True)

    for lang in LANGS:
        output_base_dir = f"C:\\Users\\jinfa\\Desktop\\Research Dr. Mani\\{lang} Run {RUNNUMBER}\\{lang} Changes\\Plots"
        os.makedirs(output_base_dir, exist_ok=True)
        generate_language_specific_plots(full_df, lang, output_base_dir)
