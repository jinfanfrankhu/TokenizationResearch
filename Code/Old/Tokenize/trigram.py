from datasets import load_dataset

lang = "ZTurkish"
# Load your .txt files
dataset = load_dataset(
    "text", 
    data_files={
        "train": rf"C:\Users\jinfa\Desktop\Research Dr. Mani\{lang}10k\article_*.txt"
    }
)

def char_trigrams_tokenize(examples):
    """
    Converts each text in examples["text"] into a list of overlapping
    3-character tokens, with whitespace.
    For "apple" -> ["app", "ppl", "ple"].
    """
    tokenized_texts = []
    for text in examples["text"]:
        # Build trigrams in a sliding window
        trigrams = []
        for i in range(len(text) - 2):
            trigrams.append(text[i : i + 3])
        
        tokenized_texts.append(trigrams)
    
    return {"tokens": tokenized_texts}


tokenized_dataset = dataset.map(char_trigrams_tokenize, batched=True)

# Choose a save path that includes the "Finnish Tokenized" subfolder
save_path = rf"C:\Users\jinfa\Desktop\Research Dr. Mani\{lang} Tokenized\Trigrams"

tokenized_dataset.save_to_disk(save_path)
