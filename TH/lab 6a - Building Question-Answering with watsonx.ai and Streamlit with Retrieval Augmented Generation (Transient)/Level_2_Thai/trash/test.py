import subprocess

def open_pdf_with_preview(pdf_path):
    try:
        # Use the open command with the path to the PDF file
        subprocess.run(["open", "-a", "Preview", pdf_path], check=True)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")

# # Example usage:
# # Replace the following path with the absolute path to your PDF file.
# pdf_path = '/absolute/path/to/thai_leave_policy.pdf'
# page_number = 12  # Replace with the page number you want to open.
# open_pdf_at_page_with_acrobat(pdf_path, page_number)



import os

# Get the absolute path to the PDF file if it's in the same directory as the script
current_directory = os.getcwd()
pdf_filename = "thai_leave_policy.pdf"
pdf_path = os.path.join(current_directory, pdf_filename)

print("Absolute path to the PDF is:", pdf_path)

open_pdf_with_preview(pdf_path)

# this will open your document on page 12
