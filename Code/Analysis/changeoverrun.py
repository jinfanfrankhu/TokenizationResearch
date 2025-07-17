from Code.metasettings import RUNNUMBER, LANGS, STRATEGIES
import os
import json
import csv
import pandas as pd

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

all_changes = []
for lang in LANGS:
    for strategy in STRATEGIES:
        # Check if old file exists
        if os.path.exists(f"C:\\Users\\jinfa\\Desktop\\Research Dr. Mani\\{lang} Run {RUNNUMBER}\\{lang} Evaluation\\{lang}_{strategy}_POS_results.json") and os.path.exists(f"C:\\Users\\jinfa\\Desktop\\Research Dr. Mani\\{lang} Run {RUNNUMBER-1}\\{lang} Evaluation\\{lang}_{strategy}_POS_results.json"):
            old_json_path = f"C:\\Users\\jinfa\\Desktop\\Research Dr. Mani\\{lang} Run {RUNNUMBER-1}\\{lang} Evaluation\\{lang}_{strategy}_POS_results.json"
            new_json_path = f"C:\\Users\\jinfa\\Desktop\\Research Dr. Mani\\{lang} Run {RUNNUMBER}\\{lang} Evaluation\\{lang}_{strategy}_POS_results.json"
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
# Mean change per strategy
summary = full_df.groupby("Strategy")[["Precision Change", "Recall Change", "F1 Change"]].mean()
print(summary)


        
            