# import os
# import shutil

# def empty_folder(folder_path):
#     """Delete all contents of a folder."""
#     if not os.path.exists(folder_path):
#         print(f"The folder '{folder_path}' does not exist.")
#         return

#     # Iterate through all files and subfolders in the folder
#     for filename in os.listdir(folder_path):
#         file_path = os.path.join(folder_path, filename)
#         try:
#             # Check if it's a file or a directory
#             if os.path.isfile(file_path) or os.path.islink(file_path):
#                 os.unlink(file_path)  # Remove the file or link
#             elif os.path.isdir(file_path):
#                 shutil.rmtree(file_path)  # Remove the directory
#         except Exception as e:
#             print(f"Failed to delete {file_path}. Reason: {e}")

#     print(f"The folder '{folder_path}' has been emptied.")

# # Specify the folder path to be emptied
# folder_to_empty = r"C:\Users\jinfa\OneDrive\Desktop\Research Dr. Mani\Finnish10k"  # Replace with your folder path

# # Call the function
# empty_folder(folder_to_empty)

print("hi")