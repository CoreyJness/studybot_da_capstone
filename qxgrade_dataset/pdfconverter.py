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

    
    
    return text_body



   
# Now that I have a pdf converter that I can use to pull text from books, I want to pull only the sentences 
# ending in a question mark.  Additional text cleaning happens in this function.
def question_separator(text):
    questions = []

    sentences = re.split(r'(?<=[?.!])\s+', text)
    
    
    for sentence in sentences:
        if sentence.strip().endswith("?"):
            sentence = sentence.replace("\n", " ").strip()
            cleaned_sentence = re.sub(r"[^a-zA-Z0-9\s\?]", "", sentence)
            questions.append(cleaned_sentence)
            
        
    return questions




##These are functions I wrote for past projects and reused

def drop_rows(df ,start, end):
    rows_to_drop = [x for x in range(start, end)]
    df.drop(rows_to_drop, axis=0, inplace=True)
    
    
def pdf_txt_retrieval(folder_path):
    q = []
    pdf_files = [f for f in os.listdir(folder_path) if f.endswith(".pdf")]
    
    
    for file in pdf_files:
        file_path = os.path.join(folder_path, file) 
        
    
    ##Run out functions on the files
        txt = parse_pdf(file_path)
        questions = question_separator(txt)
        
    ##Put them into a DF    
        df = pd.DataFrame({"question": questions})
        df["Grade"] = file  
        q.append(df)
    return pd.concat(q, ignore_index=True) if q else pd.DataFrame()

def txt_retrieval(folder_path):
    q = []
    txt_files = [f for f in os.listdir(folder_path) if f.endswith(".txt")]
    for file in txt_files:
        file_path = os.path.join(folder_path, file) 
        df = pd.read_csv(file_path) 
        df["source_file"] = file  
        q.append(df)
    return pd.concat(q, ignore_index=True) if q else pd.DataFrame()