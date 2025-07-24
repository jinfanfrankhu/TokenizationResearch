import os
from datasets import load_dataset
from gensim.models import Word2Vec
import youtokentome as yttm


def get_tokenizer(strategy, lang, runnumber):
    """
    Returns a function that tokenizes text based on the selected strategy.
    """
    if strategy == "Word":
        def tokenize(examples):
            return {"tokens": [text.split() for text in examples["text"]]}

    elif strategy == "Char":
        def tokenize(examples):
            return {"tokens": [list(text) for text in examples["text"]]}

    elif strategy == "Bigrams":
        def tokenize(examples):
            tokenized = []
            for text in examples["text"]:
                tokenized.append([text[i:i+2] for i in range(len(text)-1)])
            return {"tokens": tokenized}

    elif strategy == "Trigrams":
        def tokenize(examples):
            tokenized = []
            for text in examples["text"]:
                tokenized.append([text[i:i+3] for i in range(len(text)-2)])
            return {"tokens": tokenized}
        
    elif strategy == "4grams":
        def tokenize(examples):
            tokenized = []
            for text in examples["text"]:
                tokenized.append([text[i:i+4] for i in range(len(text)-3)])
            return {"tokens": tokenized}

    elif strategy == "5grams":
        def tokenize(examples):
            tokenized = []
            for text in examples["text"]:
                tokenized.append([text[i:i+5] for i in range(len(text)-4)])
            return {"tokens": tokenized}
        
    elif strategy[0:3] == "BPE":
        vocabsize = int(strategy[3:-1])
        # Train and load BPE tokenizer
        input_path = rf"C:\Users\jinfa\Desktop\Research Dr. Mani\{lang} Run {runnumber}\{lang}10k"
        output_path = rf"C:\Users\jinfa\Desktop\Research Dr. Mani\{lang} Run {runnumber}\{lang} Tokenized\{strategy}"
        train_file_path = os.path.join(output_path, "train_corpus.txt")

        with open(train_file_path, "w", encoding="utf-8") as f:
            for file in os.listdir(input_path):
                if file.startswith("article_") and file.endswith(".txt"):
                    with open(os.path.join(input_path, file), "r", encoding="utf-8") as infile:
                        for line in infile:
                            f.write(line.strip() + "\n")

        model_path = os.path.join(output_path, f"bpe_tokenizer_{vocabsize}.model")
        yttm.BPE.train(data=train_file_path, vocab_size=vocabsize * 1000, model=model_path)
        tokenizer = yttm.BPE(model=model_path)

        def tokenize(examples):
            return {"tokens": tokenizer.encode(examples["text"], output_type=yttm.OutputType.SUBWORD)}

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return tokenize


def tokenize_and_train(lang, strategy, runnumber):
    print(f"Tokenizing and training: Lang = {lang}, Strategy = {strategy}")
    input_path = rf"C:\Users\jinfa\Desktop\Research Dr. Mani\{lang} Run {runnumber}\{lang}10k"
    output_path = rf"C:\Users\jinfa\Desktop\Research Dr. Mani\{lang} Run {runnumber}\{lang} Tokenized\{strategy}"
    os.makedirs(output_path, exist_ok=True)

    #print(f"Loading dataset from {input_path}...")
    dataset = load_dataset("text", data_files={"train": rf"{input_path}\article_*.txt"})
    tokenizer = get_tokenizer(strategy, lang, runnumber)
    tokenized = dataset.map(tokenizer, batched=True)

    # Save tokenized dataset
    tokenized.save_to_disk(output_path)

    # Train Word2Vec
    sentences = tokenized["train"]["tokens"]
    model = Word2Vec(sentences=sentences, vector_size=100, window=5, min_count=1, workers=4, sg=1, seed=42)

    # Save the model
    model_save_path = rf"C:\Users\jinfa\Desktop\Research Dr. Mani\{lang} Run {runnumber}\{lang} Word2Vec"
    os.makedirs(model_save_path, exist_ok=True)
    model_file = os.path.join(model_save_path, f"{lang}_{strategy}_word2vec.model")
    model.save(model_file)
    print(f"Saved model to {model_file}")


if __name__ == "__main__":
    from Code.metasettings import LANGS, STRATEGIES, RUNNUMBER

    for lang in LANGS:
        for strategy in STRATEGIES:
            tokenize_and_train(lang, strategy, RUNNUMBER)
