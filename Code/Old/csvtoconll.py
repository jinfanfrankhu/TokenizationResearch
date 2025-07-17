import os
import random

# File paths
input_file = r"C:\Users\jinfa\OneDrive\Desktop\Research Dr. Mani\Evaluation\wikipedia.test.csv"  # Update with actual path
train_file = r"C:\Users\jinfa\OneDrive\Desktop\Research Dr. Mani\Evaluation\Finnishtrain.conll"
test_file = r"C:\Users\jinfa\OneDrive\Desktop\Research Dr. Mani\Evaluation\Finnishtest.conll"

# Read and clean the data
with open(input_file, "r", encoding="utf-8") as file:
    lines = file.readlines()

# Remove section headers
clean_lines = [line.strip() for line in lines if line.strip() not in ("<HEADLINE>", "<BODY>")]

# Group into sentences (split by empty lines)
sentences = []
sentence = []

for line in clean_lines:
    if line == "":  # Empty line means sentence boundary
        if sentence:
            sentences.append(sentence)
            sentence = []
    else:
        parts = line.split("\t")  # Split by tab
        if len(parts) >= 2:  # Ensure there's at least a word and a tag
            word, tag = parts[:2]  # Keep only first two columns
            sentence.append(f"{word}\t{tag}")

# Add last sentence if not already added
if sentence:
    sentences.append(sentence)

# Shuffle and split (80% Train, 20% Test)
random.seed(42)  # Ensures reproducibility
random.shuffle(sentences)

split_idx = int(len(sentences) * 0.9)
train_sentences = sentences[:split_idx]
test_sentences = sentences[split_idx:]

# Function to write to .conllu file
def save_to_conllu(sentences, file_path):
    with open(file_path, "w", encoding="utf-8") as f:
        for sentence in sentences:
            for line in sentence:
                f.write(line + "\n")
            f.write("\n")  # Separate sentences

    print(f"Saved: {file_path}")

# Save train and test sets
save_to_conllu(train_sentences, train_file)
save_to_conllu(test_sentences, test_file)

print("Conversion complete!")
