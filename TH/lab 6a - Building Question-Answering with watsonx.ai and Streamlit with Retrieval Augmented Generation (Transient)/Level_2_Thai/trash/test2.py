import fitz
from PIL import Image
import os

current_directory = os.getcwd()
pdf_filename = "thai_leave_policy.pdf"
pdf_path = os.path.join(current_directory, pdf_filename)
# Path of the PDF file

def open_pdf(pdf_path, page_num):
    # Opening the PDF file and creating a handle for it
    file_handle = fitz.open(pdf_path)
    
    # The page no. denoted by the index would be loaded
    page = file_handle[page_num]
    
    # Set the desired DPI (e.g., 200)
    zoom_x = 2.0  # horizontal zoom
    zoom_y = 2.0  # vertical zoom
    mat = fitz.Matrix(zoom_x, zoom_y)  # zoom factor 2 in each dimension
    
    # Obtaining the pixelmap of the page
    page_img = page.get_pixmap(matrix=mat)
    
    # Saving the pixelmap into a png image file
    page_img.save('PDF_page_high_res.png')
    
    # Reading the PNG image file using pillow
    img = Image.open('PDF_page_high_res.png')
    
    # Displaying the png image file using an image viewer
    img.show()


open_pdf(pdf_filename, 2)