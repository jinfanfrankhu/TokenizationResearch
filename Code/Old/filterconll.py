from Code.metasettings import LANGS
import sys
import os
import shutil

def filter_all_o(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    output_sentences = []
    current_sentence = []
    current_tags = []

    for line in lines:
        if line.strip() == '':
            if current_sentence:
                # Check if sentence has at least one non-O tag
                if any(tag != 'O' for tag in current_tags):
                    output_sentences.append('\n'.join(current_sentence))
                # Reset for next sentence
                current_sentence = []
                current_tags = []
            continue

        parts = line.strip().split()
        if len(parts) != 2:
            continue  # Skip malformed lines

        word, tag = parts
        current_sentence.append(line.strip())
        current_tags.append(tag)

    # Handle last sentence if no final newline
    if current_sentence and any(tag != 'O' for tag in current_tags):
        output_sentences.append('\n'.join(current_sentence))

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n\n'.join(output_sentences))

if __name__ == "__main__":
    for lang in LANGS:
        path = rf"C:\Users\jinfa\Desktop\Research Dr. Mani\NERSets\{lang}.conll"
        tmp_path = rf"C:\Users\jinfa\Desktop\Research Dr. Mani\NERSets\{lang}.tmp"

        filter_all_o(path, tmp_path)

        # Replace original with the filtered version
        shutil.move(tmp_path, path)
