import bz2
import json
from lxml import etree
import mwparserfromhell
import os
from Code.metasettings import LANGS, LANGCODES

NS = {'ns': 'http://www.mediawiki.org/xml/export-0.10/'}

def clean_text(wiki_text):
    if wiki_text is None:
        return ""
    code = mwparserfromhell.parse(wiki_text)
    return code.strip_code().strip()

def process_page_xml(xml_str, page_number):
    try:
        elem = etree.fromstring(xml_str)
        title_elem = elem.find("title")
        title = title_elem.text if title_elem is not None else None

        revisions = elem.findall("revision")
        latest_revision = revisions[-1] if revisions else None

        raw_text = ""
        if latest_revision is not None:
            text_elem = latest_revision.find("text")
            raw_text = text_elem.text if text_elem is not None else ""

        cleaned = clean_text(raw_text)

        return {
            "Title": title,
            "Cleaned_Text": cleaned
        }
    except Exception as e:
        print(f"[ERROR] Failed to parse page #{page_number}: {e}")
        return None

def parse_wikipedia_dump(input_bz2, output_file):
    print(f"→ Opening {input_bz2}...")
    with bz2.open(input_bz2, 'rt', encoding='utf-8', errors='ignore') as file, open(output_file, 'w', encoding='utf-8') as out:
        inside_page = False
        page_lines = []
        page_count = 0

        for line in file:
            if "<page>" in line:
                inside_page = True
                page_lines = [line]
            elif "</page>" in line:
                page_lines.append(line)
                page_xml = "".join(page_lines)
                page_count += 1

                result = process_page_xml(page_xml, page_count)
                if result:
                    out.write(json.dumps(result, ensure_ascii=False) + "\n")
                else:
                    print(f"[SKIPPED] Page #{page_count} could not be parsed.")
                    # Optional: inspect what the raw XML looked like
                    if page_count <= 5:  # Limit to first few pages
                        print("=== RAW PAGE XML SAMPLE ===")
                        print(page_xml[:500])  # Print first 500 characters
                        print("=== END OF SAMPLE ===")

                # Reset state
                inside_page = False
                page_lines = []

                # Print status every 1000 pages
                if page_count % 100000 == 0:
                    print(f"[INFO] Processed {page_count} pages...")

            elif inside_page:
                page_lines.append(line)

        print(f"[DONE] Finished processing {page_count} total pages.")

if __name__ == "__main__":
    for lang, langcode in zip(LANGS, LANGCODES):
        input_bz2 = rf"C:\Users\jinfa\Desktop\Research Dr. Mani\WikiDump\{langcode}wiki-latest-pages-articles.xml.bz2"
        output_file = rf"C:\Users\jinfa\Desktop\Research Dr. Mani\WikiDump\{lang}.jsonl"

        if not os.path.exists(input_bz2):
            print(f"[WARNING] Input file not found: {input_bz2}")
            continue

        print(f"\n[START] Processing {lang} Wikipedia dump...")
        parse_wikipedia_dump(input_bz2, output_file)
        print(f"[END] Finished {lang}. Output written to: {output_file}\n")
