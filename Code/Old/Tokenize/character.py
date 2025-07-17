from datasets import load_dataset

lang = "ZFinnish"
# Load your .txt files
dataset = load_dataset(
    "text", 
    data_files={
        "train": rf"C:\Users\jinfa\Desktop\Research Dr. Mani\{lang}10k\article_*.txt"
    }
)

def char_tokenize(examples):
    """
    Converts each text in examples["text"] into a list of individual 
    characters, including all whitespace characters.
    For "a b\nc" -> ["a", " ", "b", "\n", "c"]
    """
    tokenized_texts = []
    for text in examples["text"]:
        # Keep all characters, including whitespace
        tokens = list(text)
        tokenized_texts.append(tokens)
    
    return {"tokens": tokenized_texts}


tokenized_dataset = dataset.map(char_tokenize, batched=True)

# Choose a save path that includes the "Finnish Tokenized" subfolder
save_path = rf"C:\Users\jinfa\Desktop\Research Dr. Mani\{lang} Tokenized\Char"

tokenized_dataset.save_to_disk(save_path)
