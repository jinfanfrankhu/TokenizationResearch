from datasets import load_dataset

lang = "ZFinnish"
# Load your .txt files
dataset = load_dataset(
    "text", 
    data_files={
        "train": rf"C:\Users\jinfa\Desktop\Research Dr. Mani\{lang}10k\article_*.txt"
    }
)

def char_bigrams_tokenize(examples):
    """
    Converts each text in examples["text"] into a list of overlapping
    2-character tokens, including whitespace.
    For "a pple" -> ["a ", " p", "pp", "pl", "le"].
    """
    tokenized_texts = []
    for text in examples["text"]:
        # Keep whitespace
        bigrams = []
        for i in range(len(text) - 1):
            bigrams.append(text[i : i + 2])
        
        tokenized_texts.append(bigrams)
    
    return {"tokens": tokenized_texts}

tokenized_dataset = dataset.map(char_bigrams_tokenize, batched=True)

# Choose a save path that includes the "Finnish Tokenized" subfolder
save_path = rf"C:\Users\jinfa\Desktop\Research Dr. Mani\{lang} Tokenized\Bigrams"

tokenized_dataset.save_to_disk(save_path)
