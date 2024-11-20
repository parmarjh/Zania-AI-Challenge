import PyPDF2

class PDFReader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.text = ""

    def read_pdf(self):
        with open(self.file_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                self.text += page.extract_text() + "\n"
        return self.text