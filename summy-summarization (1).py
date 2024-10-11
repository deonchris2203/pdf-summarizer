import os
import pdfplumber
import streamlit as st
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words
import nltk
from keybert import KeyBERT
from concurrent.futures import ThreadPoolExecutor
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

uri = "mongodb+srv://mangoresoham:1Fb7vLgqdCkKd54t@summarizer.7kjk7.mongodb.net/?retryWrites=true&w=majority&appName=summarizer"

# Create a new client and connect to the server
client = MongoClient(uri, server_api=ServerApi('1'))

# Select the database and collection
db = client['my_database']  # Replace 'my_database' with your database name
collection = db['results']  # Replace 'results' with your collection name


# Download 'punkt' for tokenization
nltk.download('punkt')
nltk.download('punkt_tab')

# Configuration
LANGUAGE = "english"
SENTENCES_COUNT = 5  # Number of sentences to extract for the summary

# Function to extract text from PDF using pdfplumber
def extract_text_from_pdf(file):
    text = ""
    try:
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                text += page.extract_text()
    except Exception as e:
        text = f"Error extracting text: {e}"
    return text

# Function to summarize text using Sumy (LSA method)
def summarize_text(text, sentence_count=SENTENCES_COUNT):
    parser = PlaintextParser.from_string(text, Tokenizer(LANGUAGE))
    stemmer = Stemmer(LANGUAGE)
    summarizer = LsaSummarizer(stemmer)
    summarizer.stop_words = get_stop_words(LANGUAGE)
    summary = summarizer(parser.document, sentence_count)
    return ' '.join([str(sentence) for sentence in summary])

# Function to extract keywords using KeyBERT
def extract_keywords(text):
    kw_model = KeyBERT()
    keywords = kw_model.extract_keywords(text)
    return ', '.join([keyword for keyword, _ in keywords])

# Function to process a single PDF (summarize and extract keywords)
def process_pdf(file):
    text = extract_text_from_pdf(file)
    summary = summarize_text(text)
    keywords = extract_keywords(text)
    return summary, keywords

# Streamlit interface
def main():
    st.title("PDF Summarizer with Keywords Extraction")
    st.write("Upload multiple PDFs to get a summarized version and extracted keywords for each.")
    
    # Upload multiple PDFs
    uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)

    if uploaded_files:
        # Confirm before starting summarization
        if st.button("Summarize PDFs"):
            st.write("Summarizing...")

            # Create a progress bar
            progress_bar = st.progress(0)
            total_files = len(uploaded_files)

            # Initialize ThreadPoolExecutor for concurrency
            results = []
            with ThreadPoolExecutor() as executor:
                for i, file in enumerate(uploaded_files):
                    # Submit task to the executor
                    future = executor.submit(process_pdf, file)
                    results.append(future)

                    # Update the progress bar
                    progress_bar.progress((i + 1) / total_files)

            # Collect the summaries and keywords
            for i, future in enumerate(results):
                summary, keywords = future.result()
                st.subheader(f"Summary of {uploaded_files[i].name}:")
                st.write(summary)
                st.subheader(f"Keywords of {uploaded_files[i].name}:")
                st.write(keywords)

            result_document={
                'summary':summary,
                'keywords':keywords
            }

            collection.insert_one(result_document)
            st.write('Results saved.')

if __name__ == "__main__":
    main()