import json
import os
from pdf_reader import PDFReader
from question_answering import QuestionAnswering

def main(pdf_path, questions, api_key):
    # Read PDF
    pdf_reader = PDFReader(pdf_path)
    context = pdf_reader.read_pdf()

    # Initialize Question Answering
    qa = QuestionAnswering(api_key)

    # Prepare results
    results = {}
    for question in questions:
        answer = qa.ask_question(question, context)
        results[question] = answer

    # Output results as JSON
    print(json.dumps(results, indent=4))

if __name__ == "__main__":
    PDF_PATH = "handbook.pdf"  # Replace with your PDF file path
    QUESTIONS = [
        "What is the name of the company?",
        "Who is the CEO of the company?",
        "What is their vacation policy?",
        "What is the termination policy?"
    ]
    API_KEY = os.getenv("OPENAI_API_KEY")  # Set your OpenAI API key as an environment variable
    main(PDF_PATH, QUESTIONS, API_KEY)
