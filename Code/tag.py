import numpy as np
import os
import json
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import KFold
from gensim.models import Word2Vec
import time
import youtokentome as yttm
from collections import defaultdict
import pickle
import matplotlib.pyplot as plt
from contextlib import redirect_stdout
import io
import re

from Code.tokenizetexts import get_tokenizer
from Code.metasettings import LANGS, STRATEGIES, RUNNUMBER, TEST_STRATEGIES, TEST_LANGS

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["test", "full"], default="full")
args = parser.parse_args()

TEST_MODE = args.mode == "test"

# Load Word2Vec model
def load_word2vec():
    model_file = os.path.join(model_path, f"{lang}_{strategy}_word2vec.model")
    print(f"Loading Word2Vec model: {model_file}")
    return Word2Vec.load(model_file)

# Parse CoNLL files
def load_conll_data(file_path):
    sentences = []
    ner_tags = []
    sentence = []
    tags = []

    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            #print("Processing line:" + line.strip())
            line = line.strip()
            if line:
                parts = line.split()
                if len(parts) == 2:
                    word, tag = parts
                    sentence.append(word)
                    tags.append(tag)
            else:
                if sentence:
                    sentences.append(sentence)
                    ner_tags.append(tags)
                    sentence = []
                    tags = []

    # Patch: handle no trailing blank line
    if sentence:
        sentences.append(sentence)
        ner_tags.append(tags)

    return sentences, ner_tags

# Propagate tags through tokenizer
def propagate_tags(words, tags, tokenizer):
    new_tokens = []
    new_tags = []

    for word, tag in zip(words, tags):
        subtokens = tokenizer({"text": [word]})["tokens"][0]
        new_tokens.extend(subtokens)
        new_tags.extend([tag] * len(subtokens))

    return new_tokens, new_tags

# Convert words to embeddings
def words_to_embeddings(sentences, ner_tags, w2v_model, tokenizer):
    X = []
    y = []

    for words, tags in zip(sentences, ner_tags):
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
    changes = []

    print("Encoding NER tags...")
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)

    print("Training Logistic Regression...")
    start_time = time.time()
    captured_output = io.StringIO()
    
    model = LogisticRegression(
        max_iter=1 if TEST_MODE else 500,
        solver="saga",
        verbose=1,
        n_jobs=-1
    )
    
    # Redirect stdout to capture the verbose output
    with redirect_stdout(captured_output):
        model.fit(X_train, y_train_encoded)
    
    # Parse the captured output to extract epoch and change information
    output_lines = captured_output.getvalue().split('\n')
    for line in output_lines:
        # Look for lines containing epoch and change information
        # The format is typically: "-- Epoch 1, change: 0.123456"
        match = re.search(r'Epoch\s*:?\s*(\d+)[^\d]+change\s*:?\s*([\d.eE+-]+)', line)
        if match:
            change = float(match.group(2))
            changes.append(change)
            print("Appended change:", change)
    
    end_time = time.time()
    train_duration = float(end_time - start_time)
    n_epochs = int(model.n_iter_.max()) if hasattr(model, "n_iter_") else "N/A"

    print(f"Training completed in {train_duration:.2f} seconds with {n_epochs} epochs")

    print("Evaluating model...")
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test_encoded, y_pred)
    class_report = classification_report(
        y_test_encoded,
        y_pred,
        labels=np.arange(len(label_encoder.classes_)),
        target_names=label_encoder.classes_,
        output_dict=True,
        zero_division=0  # optional: suppress divide-by-zero warnings
    )
    conf_matrix = confusion_matrix(y_test_encoded, y_pred)

    print(f"Accuracy: {accuracy:.4f}")
    print(classification_report(
        y_test_encoded,
        y_pred,
        labels=np.arange(len(label_encoder.classes_)),
        target_names=label_encoder.classes_,
        zero_division=0
    ))
    return accuracy, class_report, conf_matrix, label_encoder.classes_, train_duration, n_epochs, changes

# Train and evaluate SGD model
def train_SGD_regression(X_train, y_train, X_test, y_test, patience=5, delta=1e-4, max_epochs=500):
    print("Encoding NER tags...")
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)

    print("Training SGD Regression with early stopping...")
    model = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        learning_rate="optimal",
        n_jobs=-1,
        random_state=42
    )

    best_loss = float("inf")
    epochs_without_improvement = 0
    losses = []
    start_time = time.time()

    for epoch in range(1 if TEST_MODE else max_epochs):
        model.partial_fit(X_train, y_train_encoded, classes=np.unique(y_train_encoded))
        y_pred_proba = model.predict_proba(X_train)
        train_loss = -np.mean(np.log(y_pred_proba[np.arange(len(y_train_encoded)), y_train_encoded] + 1e-15))
        losses.append(train_loss)

        print(f"Epoch {epoch + 1}, Loss: {train_loss:.6f}")

        if train_loss < best_loss - delta:
            best_loss = train_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs.")
            break

    end_time = time.time()
    train_duration = float(end_time - start_time)
    n_epochs = epoch + 1

    print("Evaluating model...")
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test_encoded, y_pred)
    class_report = classification_report(
        y_test_encoded,
        y_pred,
        labels=np.arange(len(label_encoder.classes_)),
        target_names=label_encoder.classes_,
        output_dict=True,
        zero_division=0
    )
    conf_matrix = confusion_matrix(y_test_encoded, y_pred)

    print(f"Accuracy: {accuracy:.4f}")
    print(classification_report(
        y_test_encoded,
        y_pred,
        labels=np.arange(len(label_encoder.classes_)),
        target_names=label_encoder.classes_,
        zero_division=0
    ))

    return accuracy, class_report, conf_matrix, label_encoder.classes_, train_duration, n_epochs, losses
    
def plot_changes_over_epochs(change_list, save_dir, filename="epoch_vs_change.png", title=None):
    """
    Plots a line graph of change values over training epochs.

    Args:
        change_list (List[float]): Change values per epoch.
        save_dir (str): Directory where the plot image will be saved.
        filename (str): Output filename for the saved plot image.
        title (str): Optional plot title. Defaults to "Epoch vs Change".
    """
    if not change_list:
        print("Warning: Empty change list passed to plot_changes_over_epochs().")
        return

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(change_list) + 1), change_list, marker='o', markersize=2, linestyle='-', linewidth=1, color='green')
    plt.xlabel("Epoch")
    plt.ylabel("Change")
    plt.title(title or f"Epoch vs Change (Epochs 1–{len(change_list)})")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    print(f"Epoch-change plot saved to: {save_path}")

def cache_tokenized_sentences(lang, strategy, runnumber, sentences, tags, tokenizer):
    cache_dir = os.path.join("cache", f"{lang}_{strategy}_run{runnumber}")
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, "tokenized.pkl")

    if os.path.exists(cache_file):
        print("Loading cached tokenized sentences...")
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    print("Tokenizing and caching sentences...")
    tokenized_data = []
    for words, tags_seq in zip(sentences, tags):
        tokens, aligned_tags = propagate_tags(words, tags_seq, tokenizer)
        tokenized_data.append((tokens, aligned_tags))

    with open(cache_file, "wb") as f:
        pickle.dump(tokenized_data, f)
    
    return tokenized_data

def cache_embeddings(lang, strategy, runnumber, tokenized_data, w2v_model):
    cache_dir = os.path.join("cache", f"{lang}_{strategy}_run{runnumber}")
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, "embeddings.pkl")

    if os.path.exists(cache_file):
        print("Loading cached embeddings...")
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    print("Computing and caching embeddings...")
    all_embeddings = []
    all_tags = []

    for tokens, tags_seq in tokenized_data:
        # Handle empty sentences
        if not tokens:
            print(f"Warning: Empty sentence found, skipping...")
            continue
            
        sentence_embeddings = []
        for token in tokens:
            if token in w2v_model.wv:
                sentence_embeddings.append(w2v_model.wv[token])
            else:
                sentence_embeddings.append(np.zeros(w2v_model.vector_size))
        
        # Ensure consistent 2D shape: (num_tokens, embedding_dim)
        embeddings_array = np.array(sentence_embeddings)
        if embeddings_array.ndim == 1 and len(sentence_embeddings) == 1:
            # Single token case: reshape from (embedding_dim,) to (1, embedding_dim)
            embeddings_array = embeddings_array.reshape(1, -1)
        elif embeddings_array.ndim == 0:
            # Empty case: create proper empty 2D array
            embeddings_array = np.empty((0, w2v_model.vector_size))
            
        all_embeddings.append(embeddings_array)
        all_tags.append(np.array(tags_seq))

    cached_data = (all_embeddings, all_tags)
    with open(cache_file, "wb") as f:
        pickle.dump(cached_data, f)

    return cached_data

def safe_concatenate_embeddings(embeddings_list, tags_list, indices):
    """Safely concatenate embeddings handling dimension mismatches"""
    valid_embeddings = []
    valid_tags = []
    
    for i in indices:
        emb = embeddings_list[i]
        tag = tags_list[i]
        
        # Skip empty sentences
        if emb.size == 0:
            continue
            
        # Ensure 2D shape
        if emb.ndim == 1:
            emb = emb.reshape(1, -1)
        elif emb.ndim == 0:
            continue  # Skip scalar/empty
            
        valid_embeddings.append(emb)
        valid_tags.append(tag)
    
    if not valid_embeddings:
        # Find embedding dimension from any non-empty embedding in the list
        embedding_dim = None
        for emb in embeddings_list:
            if emb.size > 0:
                embedding_dim = emb.shape[-1]
                break
        
        # Fallback if all embeddings are empty (shouldn't happen in practice)
        if embedding_dim is None:
            embedding_dim = 100  # Default Word2Vec dimension
            
        return np.empty((0, embedding_dim)), np.array([])
    
    return np.concatenate(valid_embeddings), np.concatenate(valid_tags)

if __name__ == "__main__":
    langs_to_run = TEST_LANGS if TEST_MODE else LANGS
    for lang in langs_to_run:
        strategies_to_run = TEST_STRATEGIES if TEST_MODE else STRATEGIES

        # Load once per language
        model_path_base = fr"C:\Users\jinfa\Desktop\Research Dr. Mani\{lang} Run {RUNNUMBER}"
        conll_file = fr"C:\Users\jinfa\Desktop\Research Dr. Mani\NERSets\{lang}.conll"
        sentences, tags = load_conll_data(conll_file)

        if len(sentences) < 5:
            print(f"Skipping {lang} due to insufficient data ({len(sentences)} sentences).")
            continue

        label_encoder = LabelEncoder()
        label_encoder.fit([tag for seq in tags for tag in seq])
        all_labels = label_encoder.classes_

        # Shared folds
        kf = KFold(n_splits=3, shuffle=True, random_state=42)
        fold_indices = list(kf.split(sentences))

        for strategy in strategies_to_run:
            model_path = os.path.join(model_path_base, f"{lang} Word2Vec")
            output_stats_path = os.path.join(model_path_base, f"{lang} Evaluation", f"{lang}_{strategy}_NER_results.json")

            w2v_model = load_word2vec()
            tokenizer = get_tokenizer(strategy, lang, RUNNUMBER)
            tokenized_data = cache_tokenized_sentences(lang, strategy, RUNNUMBER, sentences, tags, tokenizer)
            embeddings_by_sentence, tags_by_sentence = cache_embeddings(lang, strategy, RUNNUMBER, tokenized_data, w2v_model)
            merged_report = defaultdict(lambda: {"precision": 0, "recall": 0, "f1-score": 0, "support": 0})

            total_accuracy = 0.0
            total_duration = 0.0
            total_epochs = 0

            for fold, (train_idx, test_idx) in enumerate(fold_indices):
                print(f"\n--- Fold {fold+1} ---")

                valid_train_idx = [i for i in train_idx if i < len(embeddings_by_sentence)]
                valid_test_idx = [i for i in test_idx if i < len(embeddings_by_sentence)]
                
                X_train, y_train = safe_concatenate_embeddings(embeddings_by_sentence, tags_by_sentence, valid_train_idx)
                X_test, y_test = safe_concatenate_embeddings(embeddings_by_sentence, tags_by_sentence, valid_test_idx)

                if strategy.startswith("BPE"):
                    acc, report, conf_matrix, classes, duration, epochs, losses = train_SGD_regression(X_train, y_train, X_test, y_test)
                else:
                    acc, report, conf_matrix, classes, duration, epochs, losses = train_logistic_regression(X_train, y_train, X_test, y_test)

                total_accuracy += acc
                total_duration += duration
                total_epochs += epochs if isinstance(epochs, int) else 0

                plot_changes_over_epochs(losses, f"C:\\Users\\jinfa\\Desktop\\Research Dr. Mani\\{lang} Run {RUNNUMBER}\\{lang} Evaluation\\Plots\\{strategy}\\losses", filename=f"fold {fold+1}.png")

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
                    "precision": metrics["precision"] / support if support > 0 else 0.0,
                    "recall": metrics["recall"] / support if support > 0 else 0.0,
                    "f1-score": metrics["f1-score"] / support if support > 0 else 0.0,
                    "support": support
                }

            results = {
                "average_accuracy": total_accuracy / 3,
                "classification_report": final_report,
                "training_duration_seconds": total_duration,
                "average_epochs": total_epochs // 3 if total_epochs > 0 else "N/A",
                "labels": list(final_report.keys())
            }

            os.makedirs(os.path.dirname(output_stats_path), exist_ok=True)
            with open(output_stats_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=4)

            print(f"\nResults saved to {output_stats_path}")
