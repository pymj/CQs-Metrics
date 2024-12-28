import os
import PyPDF2
import pandas as pd
import numpy as np
import re
import spacy
from pathlib import Path

nlp = spacy.load("en_core_web_sm")
# base_path = os.getcwd() + "/data/"
base_path =  "CQsMetrics/data/"
# workdir=os.listdir(base_path)
# if '.DS_Store' in workdir:
#   workdir.remove('.DS_Store')
print (base_path)
exceptions= ['list']# words that should not be removed even if they are stopwords
def extract_pdf(pdf_path):
    print("pdf_path:", pdf_path)
    text = ""
    try:
        reader = PyPDF2.PdfReader(pdf_path)
        num_pages = len(reader.pages)
        print("Number of pages",num_pages)
        for page_num in range(num_pages):
            page = reader.pages[page_num]
            text += page.extract_text()
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
    return text
    

def extract_sentences(file):
    content = nlp(str(file))
    return [x.text for x in content.sents]


def get_files(path):
    file_names = [file_name for file_name in os.listdir(path) if os.path.isfile(os.path.join(path, file_name))]
    file_paths = [os.path.join(path, file_name) for file_name in file_names]
    return file_paths

def process_file(write_paths):
    for file_path in write_paths:
        if file_path.endswith('.pdf'):
            key_name = Path(file_path).stem
            filePath= os.path.join(base_path,key_name)
            outFile = os.path.join(base_path,f"{key_name}.txt")
            print(f"Processing {filePath}...")
            with open(file_path, 'rb') as f:
                text = extract_pdf(f)
                with open(outFile, "w", encoding="utf-8") as extract:
                    extract.write(text)
                print(f"Extracted text from {file_path} and saved to {outFile}")
        
        elif file_path.endswith('.txt'):
            # key_name = Path(file_path).stem
            # outFile = os.path.join(base_path,f"{key_name}.txt")
            outFile = file_path
            # print(f"Processing text file {outFile}...")
            # print(f" text file from {file_path} is the same")

        return outFile


def readTextFile(path):
    # print("text file path:", path)
    
    outText = []
    
    for file_path in path:
        if file_path.endswith('.txt'):
                try:
                    txtFile= open(file_path, "r")
                    mypdf = txtFile.readlines()
                    outText.append(mypdf)
                except FileNotFoundError:
                    print("File not found.")
                except Exception as e:
                    print("Error:", e)
                finally:
                    if 'txtFile' in locals():
                        txtFile.close()

    print("raw text:", outText)
    return outText

#---------------------------------Data Cleaning---------------------------------#

def format(outText):
    """Fix syntax issues in incoming text fields, before any further processing
    """
    print("view text:", outText)
    # sentences= [str(i).replace(";", ".") for i in outText]
    sentence_count = 0
    text_list= extract_sentences(outText)
    print("sentences:",text_list)
    # text_list= [str(i).split('.') for i in outText]
    text= ' '.join(text_list)
    for i in text.split('.'):
        sentence_count += 1
    print("count of sentences:", sentence_count )
    print("sentences:",text)
    text = text.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    rm_html_tags= re.sub(r'<[^>]*>', '', str(text))
    rm_email_address = re.sub(r'[\w\.\-]+@([\w\-]+\.)+(com|za|org)', ' ', str(rm_html_tags))
    rm_links=  re.sub(r'(https?:\/\/)?([\da-z\.-]+)\.([a-z\.]{2,6})([\/\w\.-]*)', '', str(rm_email_address))
    rm_quotes = re.sub(r'("){2,}', '', str(rm_links))
    respaced_text = re.sub(r'(?<!^)(?<![\s])(?=[A-Z])', ' ', rm_quotes)
    rm_extra_spaces = re.sub(r'\s+', ' ', respaced_text).strip()
    return rm_extra_spaces, sentence_count

def rm_non_alphabetic(rm_extra_spaces, exceptions):
    
    cleaned_text = [word if word in exceptions
                  else re.sub(r'[^a-zA-Z,\-\s]', '', word)
                  for word in str(rm_extra_spaces).split()]
    return cleaned_text

#clean text is a list of sentences

file_paths=get_files(base_path)
outFile= process_file(file_paths)
outText = readTextFile(outFile)
rm_extra_spaces, sentence_count = format(outText)
cleaned_data= rm_non_alphabetic(rm_extra_spaces, exceptions)


# def _drop_empty_rows(df, col):
#     df = df[~df[col].isin(["", "nan", "none"])].dropna()
#     return df