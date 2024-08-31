from flask import Flask, request, render_template

from test import disease_prompt_str

app = Flask(__name__)


import os
from dotenv import  load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import (
    PromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
)
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough

# Set up environment variables
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
HF_API_KEY = os.getenv("HF_API_KEY")

# Configure the chat model
chat_model = ChatGoogleGenerativeAI(
    model="gemini-pro",
    temperature=0.1
)

disease_prompt_str = """
Your task is to predict the most likely disease based on the given symptoms. Use only the information provided in the context. Be specific and provide a single diagnosis if possible..

Symptoms: {question}
Context: {context}

If there is a significant uncertainty, provide a disclaimer and list the possible diseases that match the symptoms. However, aim to provide the most likely diagnosis based on the given information.

Example of a disclaimer: "Based on the provided symptoms, it's challenging to pinpoint a single disease. Here are some possible conditions that match the symptoms..."
"""

disease_human_prompt = HumanMessagePromptTemplate(
    prompt=PromptTemplate(
        input_variables=["question", "context"],
        template=disease_prompt_str
    )
)

messages = [disease_human_prompt]
disease_prompt_template = ChatPromptTemplate(
    input_variables=["context", "question"],
    messages=messages
)

# Initialize embeddings
hf_embeddings = HuggingFaceInferenceAPIEmbeddings(
    api_key=HF_API_KEY,
    model_name="sentence-transformers/all-MiniLM-l6-v2"
)

# Load and prepare data
DISEASE_CSV_PATH = r"content/Processed_dataset.csv"
DISEASE_CHROMA_PATH = r"chroma_data"

loader = CSVLoader(file_path=DISEASE_CSV_PATH)
disease = loader.load()

disease_vector_db = Chroma(
    persist_directory=DISEASE_CHROMA_PATH,
    embedding_function=hf_embeddings
)

# Initialize vector store retriever
disease_retriever = disease_vector_db.as_retriever(k=2)


def prediction(sym):
    print("received symptoms:  ",sym)
    disease_chain = (
            {"context": disease_retriever, "question": RunnablePassthrough()}
            | disease_prompt_template
            | chat_model
            | StrOutputParser()
    )
    res = disease_chain.invoke(sym)
    return res

# Define the route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Define the route to handle the form submission
@app.route('/predict', methods=['POST'])
def predict():
    symptoms = request.form['symptoms']
    result = prediction(symptoms)
    return render_template('result.html', prediction=result)

# Start the Flask server
if __name__ == '__main__':
    app.run(debug=True)
