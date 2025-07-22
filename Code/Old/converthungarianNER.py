import os

def extract_ner(input_lines):
    sentence = []
    sentence_tags = []

    for line in input_lines:
        if line.strip() == '':
            if sentence and any(tag != 'O' for tag in sentence_tags):
                # Only add the sentence if it has at least one non-O tag
                yield sentence
            # Reset for next sentence
            sentence = []
            sentence_tags = []
            continue

        if line.startswith('#'):
            continue  # Skip comments

        parts = line.strip().split('\t')
        if len(parts) < 6:
            continue

        form = parts[0]
        ner_tag = parts[5]

        sentence.append(f"{form} {ner_tag}")
        sentence_tags.append(ner_tag)

    # Catch last sentence if file doesn't end with newline
    if sentence and any(tag != 'O' for tag in sentence_tags):
        yield sentence

if __name__ == "__main__":
    base_dir = r"C:\Users\jinfa\Desktop\Research Dr. Mani\NERSets"
    paths = ["wikipedia", "wikipediadevel", "wikipediatrain"]

    output_sentences = []

    for folder in paths:
        folder_path = os.path.join(base_dir, folder)

        for filename in os.listdir(folder_path):
            if filename.endswith(".conllup"):
                file_path = os.path.join(folder_path, filename)

                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()

                for sentence in extract_ner(lines):
                    output_sentences.append('\n'.join(sentence))

    output_path = os.path.join(base_dir, "Hungarian.conll")

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n\n'.join(output_sentences))
