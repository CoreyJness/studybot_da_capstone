import re
from pypdf import PdfReader



#These are the functions I'm going to use to convert PDFs of Textbooks to .txt files. 



#This one uses PyPdf to parse the pdfs for text
def parse_pdf(pdf):
    reader = PdfReader(pdf)
    parts = []
    
    for page in reader.pages:
        text = page.extract_text()
        if text:
            parts.append(text)
    
    
    text_body = "\n".join(parts)
    txt_filename = pdf.replace(".pdf", ".txt")
    with open(txt_filename, "w") as f:
        f.write(text_body)
    
    
    return txt_filename



   
# Now that I have a pdf converter that I can use to pull text from books, I want to pull only the sentences 
# ending in a question mark.  Additional text cleaning happens in this function.
def question_separator(file):
    questions = []
    with open (file, "r") as f: 
        text = f.read()
    sentences = re.split(r'(?<=[?.!])\s+', text)
    
    
    for sentence in sentences:
        if sentence.strip().endswith("?"):
            sentence = sentence.replace("\n", " ").strip()
            cleaned_sentence = re.sub(r"[^a-zA-Z0-9\s\?]", "", sentence)
            questions.append(cleaned_sentence)
            
            
    with open(file, "w") as f:        
        f.write("\n".join(questions))
        
        
    return questions

def math_question_separator(file):
    questions = []
    with open (file, "r") as f:
        text = f.read()
    sentences = re.split(r'(?<=[?.!])\s+', text)
    
    
    for sentence in sentences:
        if sentence.strip().endswith("?"):
            sentence = sentence.replace("\n", " ").strip()
            cleaned_sentence = re.sub(r"[^a-zA-Z0-9\s\?]", "", sentence)
            questions.append(cleaned_sentence)
            
            
    with open(file, "w") as f:        
        f.write("\n".join(questions))
        
        
    return questions

def drop_rows(df ,start, end):
    rows_to_drop = [x for x in range(start, end)]
    df.drop(rows_to_drop, axis=0, inplace=True)