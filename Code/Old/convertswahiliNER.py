import json

def process_json_to_words(json_data, output_filename="output.txt"):
    """
    Process JSON data containing sentences and write each word on a separate line.
    
    Args:
        json_data: List of tuples containing [sentence, metadata]
        output_filename: Name of the output text file
    """
    
    with open(output_filename, 'w', encoding='utf-8') as f:
        for item in json_data:
            # Extract the sentence (first element of each tuple)
            sentence = item[0]
            
            # Split sentence into words and write each word on a new line
            words = sentence.split()
            for word in words:
                f.write(word + '\n')
            
            # Add a blank line between sentences for readability
            f.write('\n')
    
    print(f"Words written to {output_filename}")

# Example usage with your data
if __name__ == "__main__":
    # Your sample data
    
    
    # If you have a JSON file instead, you can load it like this:
    with open(r'C:\Users\jinfa\Desktop\Research Dr. Mani\NERSets\swahili.json', 'r', encoding='utf-8') as f:
        json_data = json.load(f)
        process_json_to_words(json_data, "words_output.txt")