import pandas as pd
import re

# Optimized cleaning function
def clean_wikitext_optimized(text):
    if not isinstance(text, str):
        return ''  # Return an empty string for non-string values
    # Combine some regex patterns
    text = re.sub(r'(\{\{.*?\}\})|(\[\[(File|Media):.*?\]\])', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'\[\[.*?\|(.*?)\]\]', r'\1', text)  # Internal links with pipes
    text = re.sub(r'\[\[(.*?)\]\]', r'\1', text)  # Remaining internal links
    text = re.sub(r'={2,}.*?={2,}', '', text)  # Headings
    text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)  # Comments
    text = re.sub(r'<.*?>', '', text, flags=re.DOTALL)  # HTML tags
    text = re.sub(r'\s+', ' ', text).strip()  # Normalize spaces
    return text

# File paths
input_csv = r'C:\Users\jinfa\OneDrive\Desktop\Research Dr. Mani\WikiDump\finnish.csv'
output_csv = r'C:\Users\jinfa\OneDrive\Desktop\Research Dr. Mani\WikiDump\clean_finnish.csv'

chunk_size = 1000  # Number of rows per chunk

# Read and process CSV in chunks
reader = pd.read_csv(input_csv, chunksize=chunk_size, encoding='utf-8')

for i, chunk in enumerate(reader):
    print(f"Processing chunk {i + 1}")
    
    # Clean the 'Text' column
    chunk['Cleaned_Text'] = chunk['Text'].apply(clean_wikitext_optimized)
    
    # Save the cleaned chunk
    if i == 0:
        # Save the first chunk with the header
        chunk[['Title', 'Cleaned_Text']].to_csv(output_csv, index=False, mode='w')
    else:
        # Append subsequent chunks without the header
        chunk[['Title', 'Cleaned_Text']].to_csv(output_csv, index=False, mode='a', header=False)
    
    print(f"Chunk {i + 1} saved.")

print(f"All chunks processed and saved to {output_csv}")
