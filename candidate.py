from dotenv import load_dotenv, find_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter,RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
import openai
import os
import json
from tool.llm import LlmEngine
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#API_KEY
def get_openai_key():
    _ = load_dotenv(find_dotenv())
    return os.environ['OPENAI_API_KEY']


def get_pdf_text(pdf):
  pdf_reader = PdfReader(pdf)
  text = ""
  for page in pdf_reader.pages:
    text += page.extract_text()
  return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""],
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def st_first():
    st.set_page_config(page_title="Candidate Analysis")
    st.header("Candidate Analysis 💬")

def main():
    st_first()

    openai.api_key = get_openai_key()

    job_description = st.file_uploader("Upload the job description", type="pdf")

    text=""
    if job_description is not None:
        text = "Job Description: \"" + get_pdf_text(job_description) + "\""

    user_question = "You are a hiring manager with over 20 years of experience."
    user_question += "Then, you must prepare an evaluation with metrics related to Human Resources for each resume and fit with the job description."
    user_question += "In this evaluation, write using bullet points and considering that each resume start with ***."

    user_question += "The outcome is ALWAYS a json file that include the candidate with next fields, each one has an specification inside parenthesis:"
    user_question += ("\nname (text)"
                      "\naddress (text)"
                      "\nphone (text)"
                      "\nstrengths (text)"
                      "\nweaknesses (text)"
                      "\ntechnical skills (numeric from 0 to 100 according with job description)"
                      "\nleadership skills (numeric from 0 to 100 according with job description)"
                      "\ncommunication skills (numeric from 0 to 100 according with job description)"
                      "impact (numeric from 0 to 100 according with job description)"
                      "value (numeric from 0 to 100 according with job description)"
                      "fit (numeric from 0 to 100 according with job description)")

    index = 1
    file = {"candidates": []}
    df = pd.DataFrame()

    resumes = st.file_uploader("Upload resumes that want to evaluate", type="pdf", accept_multiple_files=True)
    for resume in resumes:
        eval = text + "\n***Resume " + str(index) + ": \"" + get_pdf_text(resume) + "\""
        index+=1
        if job_description is not None:
            chunks = get_text_chunks(eval)
            embeddings = OpenAIEmbeddings()
            knowledge_base = FAISS.from_texts(chunks, embeddings)
            docs = knowledge_base.similarity_search(user_question)
            llm = LlmEngine()
            chain = llm.get_qa_chain(knowledge_base)
            with get_openai_callback() as cb:
                response = chain({"query": user_question})
                print(cb)

            #st.write(response['result'])
            _json = json.loads(str(response['result']))
            file["candidates"].extend(_json["candidates"])
            df = pd.DataFrame(file["candidates"])

    st.write(file)

    if len(df)>1:
        colors = ['skyblue', 'lightgreen', 'orange', 'lightcoral']

        # Create the barplot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(df["name"], df["fit"], color=colors[: len(df)])
        plt.xlabel("Candidate Name")
        plt.ylabel("Fit Score")
        plt.title("Candidate Fit Score Comparison")
        plt.xticks(rotation=45, ha="right")
        st.pyplot(fig)


        for index, row in df.iterrows():
          candidate_name = row["name"]
          skills_data = {
              "Technical Skills": row["technical skills"],
              "Leadership Skills": row["leadership skills"],
              "Communication Skills": row["communication skills"],
              "Impact": row["impact"],
              "Value": row["value"],
          }
          fig, ax = plt.subplots()
          plt.pie(
              skills_data.values(), labels=skills_data.keys(), autopct="%1.1f%%", startangle=140
          )
          plt.axis("equal")
          st.subheader(f"Skills Breakdown for {candidate_name}")
          st.pyplot(fig)  # Embed the pie chart in Streamlit

if __name__ == '__main__':
    main()
