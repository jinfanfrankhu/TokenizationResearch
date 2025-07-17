import xml.etree.ElementTree as ET
import json

# Define input and output file paths
input_xml = r'C:\Users\jinfa\OneDrive\Desktop\Research Dr. Mani\WikiDump\trwiki-20241201-pages-meta-current.xml'
output_json = r'C:\Users\jinfa\OneDrive\Desktop\Research Dr. Mani\WikiDump\turkish.json'

# Namespace in the XML
namespace = {'mw': "http://www.mediawiki.org/xml/export-0.11/"}

# Optimized cleaning function
def clean_wikitext_optimized(text):
    import re
    if not isinstance(text, str):
        return ''
    # Remove templates
    text = re.sub(r'(\{\{.*?\}\})|(\[\[(File|Media):.*?\]\])', '', text, flags=re.DOTALL | re.IGNORECASE)
    # Internal links with pipes
    text = re.sub(r'\[\[.*?\|(.*?)\]\]', r'\1', text)
    # Remaining internal links
    text = re.sub(r'\[\[(.*?)\]\]', r'\1', text)
    # Headings
    text = re.sub(r'={2,}.*?={2,}', '', text)
    # Comments
    text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)
    # HTML tags
    text = re.sub(r'<.*?>', '', text, flags=re.DOTALL)
    # Normalize spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Process XML in chunks
def process_xml_to_json(xml_file, json_file, chunk_size=2000):
    context = ET.iterparse(xml_file, events=("start", "end"))
    context = iter(context)
    event, root = next(context)  # Get root element
    
    articles = []  # Temporary storage for chunk
    total_processed = 0  # Track total articles processed
    
    for event, elem in context:
        if event == "end" and elem.tag.endswith("page"):
            # Extract article data
            title = elem.find('mw:title', namespace).text
            text_elem = elem.find('.//mw:text', namespace)
            text = text_elem.text if text_elem is not None else ''
            cleaned_text = clean_wikitext_optimized(text)
            
            # Add article to the chunk
            articles.append({"Title": title, "Cleaned_Text": cleaned_text})
            total_processed += 1
            
            # Save and clear chunk if it reaches the chunk size
            if len(articles) >= chunk_size:
                save_json_chunk(articles, json_file)
                articles = []  # Clear chunk
                print(f"Processed {total_processed} articles...")

            # Clear processed element to save memory
            root.clear()
    
    # Save remaining articles
    if articles:
        save_json_chunk(articles, json_file)
        print(f"Processed {total_processed} articles...")
    
    print(f"All articles processed and saved to {json_file}")

# Helper function to save a chunk to JSON
def save_json_chunk(chunk, json_file):
    with open(json_file, mode='a', encoding='utf-8') as f:
        for record in chunk:
            json_record = json.dumps(record, ensure_ascii=False)
            f.write(json_record + '\n')  # Write each article as a JSON object on a new line

# Run the script
process_xml_to_json(input_xml, output_json)
