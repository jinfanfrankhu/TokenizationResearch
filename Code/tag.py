import numpy as np
import os
import json
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import KFold
from gensim.models import Word2Vec
import time
import youtokentome as yttm
from collections import defaultdict

from Code.tokenizetexts import get_tokenizer
from Code.metasettings import LANGS, STRATEGIES, RUNNUMBER

# Load Word2Vec model
def load_word2vec():
    model_file = os.path.join(model_path, f"{lang}_{strategy}_word2vec.model")
    print(f"Loading Word2Vec model: {model_file}")
    return Word2Vec.load(model_file)

# Parse CoNLL files
def load_conll_data(file_path):
    sentences = []
    pos_tags = []
    sentence = []
    tags = []

    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                parts = line.split("\t")
                if len(parts) == 2:
                    word, tag = parts
                    sentence.append(word)
                    tags.append(tag)
            else:
                if sentence:
                    sentences.append(sentence)
                    pos_tags.append(tags)
                    sentence = []
                    tags = []

    return sentences, pos_tags

# Propagate tags through tokenizer
def propagate_tags(words, tags, tokenizer):
    new_tokens = []
    new_tags = []

    for word, tag in zip(words, tags):
        subtokens = tokenizer(word)
        new_tokens.extend(subtokens)
        new_tags.extend([tag] * len(subtokens))

    return new_tokens, new_tags

# Convert words to embeddings
def words_to_embeddings(sentences, pos_tags, w2v_model, tokenizer):
    X = []
    y = []

    for words, tags in zip(sentences, pos_tags):
        tokens, aligned_tags = propagate_tags(words, tags, tokenizer)

        for token, tag in zip(tokens, aligned_tags):
            if token in w2v_model.wv:
                X.append(w2v_model.wv[token])
            else:
                X.append(np.zeros(w2v_model.vector_size))
            y.append(tag)

    return np.array(X), np.array(y)

# Train and evaluate Logistic Regression model
def train_logistic_regression(X_train, y_train, X_test, y_test):
    print("Encoding POS tags...")
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)

    print("Training Logistic Regression...")
    start_time = time.time()
    model = LogisticRegression(max_iter=500, solver="saga", verbose=1, n_jobs=-1)
    model.fit(X_train, y_train_encoded)
    end_time = time.time()

    train_duration = float(end_time - start_time)
    n_epochs = int(model.n_iter_.max()) if hasattr(model, "n_iter_") else "N/A"

    print(f"Training completed in {train_duration:.2f} seconds with {n_epochs} epochs")

    print("Evaluating model...")
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test_encoded, y_pred)
    class_report = classification_report(
        y_test_encoded, y_pred, target_names=label_encoder.classes_, output_dict=True
    )
    conf_matrix = confusion_matrix(y_test_encoded, y_pred)

    print(f"Accuracy: {accuracy:.4f}")
    print(classification_report(y_test_encoded, y_pred, target_names=label_encoder.classes_))

    return accuracy, class_report, conf_matrix, label_encoder.classes_, train_duration, n_epochs

# Main logic
if __name__ == "__main__":
    for lang in LANGS:
        for strategy in STRATEGIES:
            model_path = fr"C:\Users\jinfa\Desktop\Research Dr. Mani\{lang} Run {RUNNUMBER}\{lang} Word2Vec"
            conll_file = fr"C:\Users\jinfa\Desktop\Research Dr. Mani\NERSets\{lang}.conll"
            output_stats_path = fr"C:\Users\jinfa\Desktop\Research Dr. Mani\{lang} Run {RUNNUMBER}\{lang} Evaluation\{lang}_{strategy}_POS_results.json"

            w2v_model = load_word2vec()
            sentences, tags = load_conll_data(conll_file)
            tokenizer = get_tokenizer(strategy, lang, RUNNUMBER)

            kf = KFold(n_splits=5, shuffle=True, random_state=42)

            merged_report = defaultdict(lambda: {"precision": 0, "recall": 0, "f1-score": 0, "support": 0})
            total_accuracy = 0.0
            total_duration = 0.0
            total_epochs = 0

            for fold, (train_idx, test_idx) in enumerate(kf.split(sentences)):
                print(f"\n--- Fold {fold+1} ---")
                train_sent = [sentences[i] for i in train_idx]
                train_tags = [tags[i] for i in train_idx]
                test_sent = [sentences[i] for i in test_idx]
                test_tags = [tags[i] for i in test_idx]

                X_train, y_train = words_to_embeddings(train_sent, train_tags, w2v_model, tokenizer)
                X_test, y_test = words_to_embeddings(test_sent, test_tags, w2v_model, tokenizer)

                acc, report, conf_matrix, classes, duration, epochs = train_logistic_regression(X_train, y_train, X_test, y_test)
                total_accuracy += acc
                total_duration += duration
                total_epochs += epochs if isinstance(epochs, int) else 0

                for label, scores in report.items():
                    if label in ("accuracy", "macro avg", "weighted avg"):
                        continue
                    for metric in ["precision", "recall", "f1-score"]:
                        merged_report[label][metric] += scores[metric] * scores["support"]
                    merged_report[label]["support"] += scores["support"]

            # Average and finalize
            final_report = {}
            for label, metrics in merged_report.items():
                support = metrics["support"]
                final_report[label] = {
                    "precision": metrics["precision"] / support,
                    "recall": metrics["recall"] / support,
                    "f1-score": metrics["f1-score"] / support,
                    "support": support
                }

            results = {
                "average_accuracy": total_accuracy / 5,
                "classification_report": final_report,
                "training_duration_seconds": total_duration,
                "average_epochs": total_epochs // 5 if total_epochs > 0 else "N/A",
                "labels": list(final_report.keys())
            }

            os.makedirs(os.path.dirname(output_stats_path), exist_ok=True)
            with open(output_stats_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=4)

            print(f"\nResults saved to {output_stats_path}")
