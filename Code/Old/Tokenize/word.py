from datasets import load_dataset

lang = "ZFinnish"
# Load your .txt files
dataset = load_dataset(
    "text", 
    data_files={
        "train": fr"C:\Users\jinfa\Desktop\Research Dr. Mani\{lang}10k\article_*.txt"
    }
)

def simple_tokenize(examples):
    # Whitespace-split
    return {"tokens": [text.split() for text in examples["text"]]}

tokenized_dataset = dataset.map(simple_tokenize, batched=True)

# Choose a save path that includes the "Finnish Tokenized" subfolder
save_path = rf"C:\Users\jinfa\Desktop\Research Dr. Mani\{lang} Tokenized\Word"

tokenized_dataset.save_to_disk(save_path)
