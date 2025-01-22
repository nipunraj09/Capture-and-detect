import pytesseract
from PIL import Image
import os

# Set the path to the Tesseract executable if it's not already in your PATH
# Uncomment the following line and provide the correct path if needed
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Path to the image file
image_path = 'OCRproj.jpg'

# Check if the image file exists
if not os.path.exists(image_path):
    print(f"Error: File '{image_path}' not found.")
else:
    try:
        # Open the image file
        image = Image.open(image_path)

        # Perform OCR using PyTesseract
        text = pytesseract.image_to_string(image)

        # Print the extracted text
        print("Extracted Text:")
        print(text)
    except Exception as e:
        print(f"An error occurred: {e}")
