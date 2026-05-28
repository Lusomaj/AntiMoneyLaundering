import pypdf
import sys

def extract_text(pdf_path):
    try:
        with open(pdf_path, 'rb') as file:
            reader = pypdf.PdfReader(file)
            text = ""
            for i, page in enumerate(reader.pages):
                text += f"--- Page {i+1} ---\n"
                text += page.extract_text() + "\n"
            return text
    except Exception as e:
        return str(e)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python extract_pdf.py <pdf_path>")
    else:
        full_text = extract_text(sys.argv[1])
        # Save to text file for easier reading
        with open("proposal_text.txt", "w", encoding="utf-8") as f:
            f.write(full_text)
        print("Text extracted and saved to proposal_text.txt")

