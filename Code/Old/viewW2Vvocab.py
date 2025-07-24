import youtokentome as yttm
import os

def export_vocab(model_path, output_path):
    # Load the trained BPE model
    tokenizer = yttm.BPE(model=model_path)

    # Get the vocabulary (list of subword strings)
    vocab = tokenizer.vocab()

    # Write to output file
    with open(output_path, "w", encoding="utf-8") as f:
        for i, token in enumerate(vocab):
            f.write(f"{i}\t{token}\n")

    print(f"✅ Saved vocab to {output_path}")


if __name__ == "__main__":
    from Code.metasettings import LANGS, STRATEGIES, RUNNUMBER
    
    for lang in LANGS:
        for strategy in STRATEGIES:
            if not (strategy.startswith("BPE")):
                continue
            model_path = rf"C:\Users\jinfa\Desktop\Research Dr. Mani\{lang} Run {RUNNUMBER}\{lang} Tokenized\{strategy}\bpe_tokenizer_{strategy[3:-1]}.model"
            output_path = rf"C:\Users\jinfa\Desktop\Research Dr. Mani\{lang} Run {RUNNUMBER}\{lang} Tokenized\{strategy}\bpe_vocab.txt"

            export_vocab(model_path, output_path)


    
