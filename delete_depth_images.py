from pathlib import Path

# --- IMPORTANT: SET YOUR FOLDER PATH HERE ---
# Replace the placeholder with the actual path to your image folder.
# Example on Windows: r"C:\Users\YourUser\Desktop\MyImages"
# Example on Linux/macOS: "/home/leonardo/Projects/MyDataset/images"
folder_path = Path("./Dataset/Strawberry/week_1/") 
# ----------------------------------------------------


def delete_depth_files(target_folder: Path, perform_delete: bool = False):
    """
    Finds and optionally deletes depth images in a target folder.
    
    Args:
        target_folder (Path): The folder to search for images.
        perform_delete (bool): If True, deletes files. If False, just prints them (Dry Run).
    """
    if not target_folder.is_dir():
        print(f"Error: The folder '{target_folder}' does not exist.")
        return

    # Use glob to find all files ending with '_depth.png'
    # The '*' is a wildcard that matches any character sequence.
    depth_files = list(target_folder.glob('*_depth.png'))

    if not depth_files:
        print("No depth images found to delete.")
        return

    print("--- Files identified for deletion ---")
    for file_path in depth_files:
        print(file_path.name)
    print("-----------------------------------")
    
    if perform_delete:
        print("\nWARNING: Deleting the files listed above...")
        confirmation = input("Are you absolutely sure you want to proceed? (yes/no): ")
        
        if confirmation.lower() == 'yes':
            for file_path in depth_files:
                try:
                    file_path.unlink() # This is the command that deletes the file
                    print(f"Deleted: {file_path.name}")
                except Exception as e:
                    print(f"Error deleting {file_path.name}: {e}")
            print("\nDeletion complete.")
        else:
            print("\nDeletion cancelled by user.")
    else:
        print("\nThis was a DRY RUN. No files were deleted.")
        print("To delete these files, run the script again with the delete flag.")


if __name__ == "__main__":
    # --- CHOOSE YOUR MODE ---
    # Set to False to safely preview which files will be deleted.
    # Set to True to perform the actual deletion.
    DELETE_MODE = True
    # ------------------------

    delete_depth_files(folder_path, perform_delete=DELETE_MODE)