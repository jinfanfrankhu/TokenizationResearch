import json
import random
import os
import re
from Code.metasettings import LANGS, RUNNUMBER

def sanitize_filename(name):
    """Replace invalid filename characters with an underscore."""
    return re.sub(r'[<>:"/\\|?*]', '_', name)

def sample_10k_valid_articles(json_file_path, output_folder, target_count=10000, min_size_kb=4, min_threshold_kb=1):
    os.makedirs(output_folder, exist_ok=True)
    saved_titles = set()
    valid_count = 0
    total_lines_read = 0

    while valid_count < target_count and min_size_kb >= min_threshold_kb:
        new_articles_this_pass = 0

        with open(json_file_path, 'r', encoding='utf-8') as file:
            for line in file:
                total_lines_read += 1
                try:
                    article = json.loads(line.strip())
                    title = article.get("Title", "").strip()
                    text = article.get("Cleaned_Text", "").strip()

                    # Skip empty or duplicate titles
                    if not title or not text or title in saved_titles:
                        continue

                    # Size filter
                    if len(text.encode('utf-8')) < min_size_kb * 1024:
                        continue

                    # Save article
                    safe_title = sanitize_filename("_".join(title.split())[:50])
                    filename = f"article_{valid_count+1}_{safe_title}.txt"
                    file_path = os.path.join(output_folder, filename)

                    with open(file_path, 'w', encoding='utf-8') as output_file:
                        output_file.write(f"Title: {title}\n{text}\n")

                    saved_titles.add(title)
                    valid_count += 1
                    new_articles_this_pass += 1

                    if valid_count >= target_count:
                        break

                except json.JSONDecodeError:
                    continue

        if new_articles_this_pass == 0:
            min_size_kb -= 1
            print(f"⚠️ No new articles found. Lowering min_size_kb to {min_size_kb} and restarting...")
        else:
            print(f"✅ Added {new_articles_this_pass} articles this pass (min_size_kb={min_size_kb}) — {valid_count} total.")

    if valid_count < target_count:
        print(f"⚠️ Finished with {valid_count} articles. Could not reach 10k even after lowering size threshold.")
    else:
        print(f"🎉 Success! Saved {valid_count} valid articles to: {output_folder}")


# Main script
if __name__ == "__main__":
    for lang in LANGS:
        json_file_path = fr"C:\Users\jinfa\Desktop\Research Dr. Mani\WikiDump\{lang}.jsonl"
        output_folder = fr"C:\Users\jinfa\Desktop\Research Dr. Mani\{lang} Run {RUNNUMBER}\{lang}10k"
        sample_10k_valid_articles(json_file_path, output_folder)
