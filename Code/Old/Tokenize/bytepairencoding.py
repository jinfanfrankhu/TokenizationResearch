import youtokentome as yttm
from datasets import load_dataset
import os

langs = ["ZFinnish"]
vocabsizes = [5, 10, 25, 50]
def do_it(lang, vocabsize):
    input_path = rf"C:\Users\jinfa\Desktop\Research Dr. Mani\{lang}10k"
    output_path = rf"C:\Users\jinfa\Desktop\Research Dr. Mani\{lang} Tokenized\SUBWORDCORRECTEDBPE{vocabsize}k"

    # Ensure output directory exists
    os.makedirs(output_path, exist_ok=True)

    # Combine all input files into a single training file for YTTM
    train_file_path = os.path.join(output_path, "train_corpus.txt")
    with open(train_file_path, "w", encoding="utf-8") as train_file:
        for file_name in os.listdir(input_path):
            if file_name.startswith("article_") and file_name.endswith(".txt"):
                with open(os.path.join(input_path, file_name), "r", encoding="utf-8") as f:
                    for line in f:
                        train_file.write(line.strip() + "\n")

    # Train a BPE tokenizer using YTTM
    tokenizer_model_path = os.path.join(output_path, "bpe_tokenizer.model")
    yttm.BPE.train(data=train_file_path, vocab_size=vocabsize * 1000, model=tokenizer_model_path)

    # Load the trained tokenizer
    tokenizer = yttm.BPE(model=tokenizer_model_path)

    # Load the dataset
    dataset = load_dataset(
        "text", 
        data_files={"train": rf"{input_path}\article_*.txt"}
    )

    # Tokenize the dataset using the BPE tokenizer
    def bpe_tokenize(examples):
        """
        Tokenizes each text in examples["text"] using the BPE tokenizer.
        """
        tokenized_texts = tokenizer.encode(examples["text"], output_type=yttm.OutputType.SUBWORD)
        return {"tokens": tokenized_texts}

    # Apply the tokenizer to the dataset
    tokenized_dataset = dataset.map(bpe_tokenize, batched=True)

    # Save the tokenized dataset
    tokenized_dataset.save_to_disk(output_path)

    print(f"BPE tokenized dataset saved to {output_path}")

for lang in langs:
    for vocabsize in vocabsizes:
        do_it(lang, vocabsize)