
from pypdf import PdfReader
import sys

try:
    reader = PdfReader("/Users/suryaprakashk/Documents/AIMI-Projects/Blinkit/Blinkit Final Project.pdf")
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    print(text)
except Exception as e:
    print(f"Error reading PDF: {e}")
