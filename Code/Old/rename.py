import os

# Define the directory path
directory = r"C:\Users\jinfa\Desktop\Research Dr. Mani\Turkish Run 2\Turkish Evaluation"

# Loop over files in the directory
for filename in os.listdir(directory):
    # Check if the filename starts with "ZFinnish_" and ends with "_POS_results.json"
    if filename.startswith("ZTurkish") and filename.endswith("_POS_results.json"):
        # Create the new filename by removing the leading "Z"
        new_filename = "Turkish_" + filename[len("ZTurkish_"):]
        
        # Get full file paths
        old_file = os.path.join(directory, filename)
        new_file = os.path.join(directory, new_filename)
        
        # Rename the file
        os.rename(old_file, new_file)
        print(f"Renamed: {filename} -> {new_filename}")

print("Renaming complete.")