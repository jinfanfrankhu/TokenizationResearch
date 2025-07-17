import numpy as np
import os
import json
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from gensim.models import Word2Vec
import time
import youtokentome as yttm

from metasettings import LANGS, STRATEGIES, RUNNUMBER

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
            if line:  # Token and tag are present
                parts = line.split("\t")
                if len(parts) == 2:
                    word, tag = parts
                    sentence.append(word)
                    tags.append(tag)
            else:  # Sentence boundary
                if sentence:
                    sentences.append(sentence)
                    pos_tags.append(tags)
                    sentence = []
                    tags = []

    return sentences, pos_tags

def get_tokenizer(strategy, lang, runnumber):
    if strategy == "Word":
        return lambda s: s.split()

    elif strategy == "Char":
        return list

    elif strategy == "Bigrams":
        return lambda s: [s[i:i+2] for i in range(len(s) - 1)]

    elif strategy == "Trigrams":
        return lambda s: [s[i:i+3] for i in range(len(s) - 2)]

    elif strategy.startswith("BPE"):
        vocabsize = int(strategy[3:-1])
        model_path = rf"C:\Users\jinfa\Desktop\Research Dr. Mani\{lang} Run {runnumber}\{lang} Tokenized\{strategy}\bpe_tokenizer_{vocabsize}.model"
        tokenizer = yttm.BPE(model=model_path)

        return lambda s: tokenizer.encode([s], output_type=yttm.OutputType.SUBWORD)[0]

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

# Propagate tags through tokenizer
def propagate_tags(words, tags, tokenizer):
    """
    Given original words and their tags, apply the tokenizer
    and propagate each tag to the tokens it produces.
    """
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

    # Compute performance metrics
    accuracy = accuracy_score(y_test_encoded, y_pred)
    class_report = classification_report(y_test_encoded, y_pred, target_names=label_encoder.classes_, output_dict=True)
    conf_matrix = confusion_matrix(y_test_encoded, y_pred)

    # Print results
    print(f"Accuracy: {accuracy:.4f}")
    print(classification_report(y_test_encoded, y_pred, target_names=label_encoder.classes_))

    # Save results to file
    save_results(accuracy, class_report, conf_matrix, label_encoder.classes_, train_duration, n_epochs)

# Save evaluation results to a file
def save_results(accuracy, class_report, conf_matrix, class_labels, train_duration, n_epochs):
    results = {
        "accuracy": accuracy,
        "classification_report": class_report,
        "confusion_matrix": conf_matrix.tolist(),
        "labels": class_labels.tolist(),
        "training_duration_seconds": train_duration,
        "epochs": n_epochs
    }

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_stats_path), exist_ok=True)

    with open(output_stats_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    print(f"Results saved to {output_stats_path}")

for lang in LANGS:
    for strategy in STRATEGIES:
        model_path = fr"C:\Users\jinfa\Desktop\Research Dr. Mani\{lang} Run {RUNNUMBER}\{lang} Word2Vec"
        train_file = fr"C:\Users\jinfa\Desktop\Research Dr. Mani\NERSets\{lang}train.conll"
        test_file = fr"C:\Users\jinfa\Desktop\Research Dr. Mani\NERSets\{lang}test.conll"
        
        output_stats_path = fr"C:\Users\jinfa\Desktop\Research Dr. Mani\{lang} Run {RUNNUMBER}\{lang} Evaluation\{lang}_{strategy}_POS_results.json"

        # Load Word2Vec model
        w2v_model = load_word2vec()

        # Load and process dataset
        print("Loading training data...")
        train_sentences, train_pos_tags = load_conll_data(train_file)
        tokenizer = get_tokenizer(strategy, lang, RUNNUMBER)
        X_train, y_train = words_to_embeddings(train_sentences, train_pos_tags, w2v_model, tokenizer)

        print("Loading testing data...")
        test_sentences, test_pos_tags = load_conll_data(test_file)
        X_test, y_test = words_to_embeddings(test_sentences, test_pos_tags, w2v_model, tokenizer)

        # Train and evaluate
        train_logistic_regression(X_train, y_train, X_test, y_test)
