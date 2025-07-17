import os
import string

def process_text_file(file_path):
    """
    Processes a .txt file to convert text to lowercase and remove all punctuation
    except for apostrophes, then overwrites the file with the cleaned text.

    :param file_path: Path to the .txt file to process
    """
    # Define the allowed characters (all punctuation except apostrophes will be removed)
    allowed_characters = string.ascii_letters + string.digits + string.whitespace + "'" + "çğıöşüÇĞİÖŞÜ" + "äåöÄÅÖ" + ".,?:;"

    try:
        with open(file_path, 'r', encoding='utf-8') as infile:
            text = infile.read()

        # Convert to lowercase and filter out unwanted characters
        cleaned_text = ''.join(char if char in allowed_characters else ' ' for char in text)

        # Remove instances of "'''" and "''"
        with open(file_path, 'w', encoding='utf-8') as outfile:
            outfile.write(cleaned_text)

        print(f"Processed file: {file_path}")

    except FileNotFoundError:
        print(f"Error: The file {file_path} does not exist.")
    except Exception as e:
        print(f"An error occurred: {e}")

def batch_process_txt_files(input_directory):
    """
    Batch processes all .txt files in a directory to convert text to lowercase
    and remove punctuation except for apostrophes. Overwrites the original files.

    :param input_directory: Path to the directory containing .txt files
    """
    for filename in os.listdir(input_directory):
        if filename.endswith('.txt'):
            file_path = os.path.join(input_directory, filename)

            print(f"Processing {filename}...")
            process_text_file(file_path)

if __name__ == "__main__":
    input_dir = input("Enter the path to the directory containing .txt files: ").strip()
    batch_process_txt_files(input_dir)
