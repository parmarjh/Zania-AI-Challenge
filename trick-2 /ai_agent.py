
import os
import json
import logging
import argparse
from typing import List, Dict, Tuple, Union
from openai import OpenAI
from dotenv import load_dotenv
import PyPDF2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()

# Set up OpenAI client with the API key
client = OpenAI(api_key="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")

def extract_text_from_pdf(pdf_path: str) -> str:
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
        return text
    except FileNotFoundError:
        raise FileNotFoundError(f"The PDF file '{pdf_path}' was not found.")
    except PyPDF2.errors.PdfReadError:
        raise ValueError(f"The file '{pdf_path}' is not a valid PDF or is corrupted.")
    except Exception as e:
        logging.error(f"Error extracting text from PDF: {e}")
        raise

def chunk_text(text: str, chunk_size: int = 1000) -> List[str]:
    words = text.split()
    chunks = []
    current_chunk = []
    current_size = 0
    for word in words:
        if current_size + len(word) > chunk_size:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_size = len(word)
        else:
            current_chunk.append(word)
            current_size += len(word) + 1
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

def generate_embeddings(chunks: List[str]) -> List[List[float]]:
    embeddings = []
    try:
        for chunk in chunks:
            response = client.embeddings.create(
                input=chunk,
                model="text-embedding-ada-002"
            )
            embeddings.append(response.data[0].embedding)
        return embeddings
    except Exception as e:
        logging.error(f"Error generating embeddings: {e}")
        raise

def find_most_relevant_chunk(question: str, chunks: List[str], embeddings: List[List[float]]) -> Tuple[str, float]:
    question_embedding = client.embeddings.create(
        input=question,
        model="text-embedding-ada-002"
    ).data[0].embedding
    
    similarities = cosine_similarity([question_embedding], embeddings)[0]
    most_relevant_idx = np.argmax(similarities)
    return chunks[most_relevant_idx], similarities[most_relevant_idx]

def answer_question(question: str, context: str) -> Tuple[str, float]:
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided context. If the answer is not in the context, say 'I don't have enough information to answer this question.'"},
                {"role": "user", "content": f"Context: {context}\n\nQuestion: {question}"}
            ],
            max_tokens=150,
            n=1,
            temperature=0.5,
        )
        answer = response.choices[0].message.content.strip()
        confidence = 1.0 if "I don't have enough information" not in answer else 0.0
        return answer, confidence
    except Exception as e:
        logging.error(f"Error answering question: {e}")
        return "Error: Unable to generate an answer.", 0.0

def process_pdf_and_questions(pdf_path: str, questions: List[str]) -> Dict[str, Dict[str, Union[str, float]]]:
    try:
        text = extract_text_from_pdf(pdf_path)
        chunks = chunk_text(text)
        embeddings = generate_embeddings(chunks)
        
        answers = {}
        for question in questions:
            relevant_chunk, chunk_similarity = find_most_relevant_chunk(question, chunks, embeddings)
            answer, answer_confidence = answer_question(question, relevant_chunk)
            confidence = chunk_similarity * answer_confidence
            answers[question] = {
                "answer": answer,
                "confidence": confidence
            }
        
        return answers
    except Exception as e:
        logging.error(f"Error processing PDF and questions: {e}")
        return {question: {"answer": f"Error: {str(e)}", "confidence": 0.0} for question in questions}

def main():
    parser = argparse.ArgumentParser(description="AI agent to answer questions based on a PDF document")
    parser.add_argument("pdf_path", help="Path to the PDF file")
    parser.add_argument("questions", nargs="+", help="List of questions to answer")
    args = parser.parse_args()

    try:
        answers = process_pdf_and_questions(args.pdf_path, args.questions)
        print(json.dumps(answers, indent=2))
    except Exception as e:
        logging.error(f"Error in main function: {e}")
        print(json.dumps({"error": str(e)}, indent=2))

if __name__ == "__main__":
    main()
