import json
import random
import os
import re

def sanitize_filename(name):
    """Replace invalid filename characters with an underscore."""
    return re.sub(r'[<>:"/\\|?*]', '_', name)

def clean_text_content(text):
    """Remove irrelevant content before '}}' if present."""
    if '}}' in text:
        return text.split('}}', 1)[-1].strip()
    return text

def sample_large_json_file(json_file_path, output_folder, sample_size=10000, min_size_kb=4):
    valid_articles = []

    # Open the JSON file and read line by line
    with open(json_file_path, 'r', encoding='utf-8') as file:
        for line in file:
            try:
                article = json.loads(line.strip())  # Parse each line as JSON
                title = article.get("Title", "")
                text = article.get("Cleaned_Text", "")

                # Normalize title for consistent comparison
                normalized_title = title.strip().lower()

                # Exclude articles starting with "Kullanıcı mesaj" or "Keskustelu"
                if not normalized_title.startswith("kullanıcı mesaj") and not normalized_title.startswith("keskustelu") and len(text.encode('utf-8')) > min_size_kb * 1024:
                    article["Cleaned_Text"] = clean_text_content(text)  # Clean the text
                    valid_articles.append(article)
            except json.JSONDecodeError:
                print("Skipping invalid JSON line.")


    # Adjust sample size if there are fewer valid articles
    sample_size = min(sample_size, len(valid_articles))

    # Sample random articles from the valid ones
    sampled_articles = random.sample(valid_articles, sample_size)

    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Save each article as a separate .txt file
    for idx, article in enumerate(sampled_articles):
        title = article.get("Title", "No Title")
        text = article.get("Cleaned_Text", "No Text")

        # Clean the title to create a valid filename
        safe_title = sanitize_filename("_".join(title.split())[:50])  # Limit to 50 characters
        filename = f"article_{idx+1}_{safe_title}.txt"
        file_path = os.path.join(output_folder, filename)

        # Write the article to the .txt file
        with open(file_path, 'w', encoding='utf-8') as output_file:
            output_file.write(f"Title: {title}\n{text}\n")

    print(f"Sampled articles saved as separate .txt files in {output_folder}")

# Specify your file paths and folder
json_file_path = r"C:\Users\jinfa\OneDrive\Desktop\Research Dr. Mani\WikiDump\turkish.json"
output_folder = r"C:\Users\jinfa\OneDrive\Desktop\Research Dr. Mani\ZTurkish10k"

# Call the function
sample_large_json_file(json_file_path, output_folder)
