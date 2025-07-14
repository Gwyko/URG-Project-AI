import os
import pypdf

def is_pdf_empty(file_path: str) -> bool:
    """
    Checks if a PDF file is empty.

    An "empty" PDF is defined as one that either:
    1. Has zero pages.
    2. Contains no extractable text content.

    Args:
        file_path: The full path to the PDF file.

    Returns:
        True if the PDF is considered empty, False otherwise.
    """
    try:
        # Open the PDF file in binary read mode
        with open(file_path, "rb") as f:
            reader = pypdf.PdfReader(f)

            # 1. Check if there are any pages in the PDF
            if len(reader.pages) == 0:
                return True

            # 2. Check if there is any text content on any page
            for page in reader.pages:
                # Extract text from the page. If any is found, it's not empty.
                if page.extract_text() and page.extract_text().strip():
                    return False

            # If the loop completes with no text found, it's empty.
            return True

    except pypdf.errors.PdfReadError:
        # This can happen if the file is corrupted or not a valid PDF
        print(f"  -> Warning: Could not read '{os.path.basename(file_path)}'. It might be corrupted. Skipping.")
        return False # Treat as not-empty to be safe
    except FileNotFoundError:
        print(f"  -> Error: File not found at '{file_path}'.")
        return False
    except Exception as e:
        print(f"  -> An unexpected error occurred with '{os.path.basename(file_path)}': {e}")
        return False # Be safe, don't delete on unexpected errors

def process_and_delete_in_directory(directory_path: str):
    """
    Scans a directory for PDF files, checks if they are empty,
    and PERMANENTLY DELETES them if they are.
    """
    print(f"Scanning directory: '{directory_path}'...")
    print("--- WARNING: Auto-deleting empty PDFs without confirmation. ---")

    if not os.path.isdir(directory_path):
        print(f"Error: Directory '{directory_path}' not found. Aborting.")
        return

    # Loop through all files in the given directory
    for filename in os.listdir(directory_path):
        # Check if the file is a PDF
        if filename.lower().endswith('.pdf'):
            full_path = os.path.join(directory_path, filename)
            print(f"Checking '{filename}'...")

            # Check if the PDF is empty
            if is_pdf_empty(full_path):
                try:
                    # If it's empty, delete the file
                    os.remove(full_path)
                    print(f"  -> Empty. DELETED '{filename}'.")
                except OSError as e:
                    print(f"  -> Error deleting file '{filename}': {e}")
            else:
                # If it has content, keep it
                print(f"  -> Has content. Keeping file.")

    print("\nScan complete.")


# --- Main Execution ---
if __name__ == "__main__":
    # ---!!!--- DANGER ---!!!---
    # SET YOUR TARGET DIRECTORY PATH CAREFULLY.
    # This script runs IMMEDIATELY and PERMANENTLY DELETES files.
    #
    # Use "." for the current directory.
    # Use a full path for a specific directory, e.g., "C:/Users/YourUser/Documents/PDFs"
    
    target_directory = "./data" 

    # The script will now run directly without any further prompts.
    process_and_delete_in_directory(target_directory)