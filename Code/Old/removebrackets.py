import os

def process_files(directory):
    # Loop through all files in the directory
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            file_path = os.path.join(directory, filename)

            # Read the content of the file
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()

            # Find the marker and remove text before it
            marker = "'''"
            marker_index = content.find(marker)
            if marker_index != -1:
                # Retain text starting after the marker
                modified_content = content[marker_index + len(marker):]

                # Write the modified content back to the file
                with open(file_path, 'w', encoding='utf-8') as file:
                    file.write(modified_content)
                print(f"Removed marker from {filename}")
            else:
                print(f"Marker not found in file: {filename}")

# Define directories
finnish_dir = r'C:\Users\jinfa\OneDrive\Desktop\Research Dr. Mani\Finnish10k'
turkish_dir = r'C:\Users\jinfa\OneDrive\Desktop\Research Dr. Mani\Turkish10k'

# Process files in both directories
process_files(finnish_dir)

print("Processing complete.")
