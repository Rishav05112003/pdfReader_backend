import os
import uuid
from typing import Dict, Any
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pypdf import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_google_genai import chat_models
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from dotenv import load_dotenv
import logging
from pydantic import BaseModel

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def verify_google_key():
    try:
        from google.generativeai import configure
        configure(api_key=os.getenv("GOOGLE_API_KEY"))
        return True
    except Exception as e:
        logger.error(f"Google API key verification failed: {str(e)}")
        return False

# checking for google api key
load_dotenv()
apiKey = os.getenv("GOOGLE_API_KEY")
print(f"Retrieved API Key: {apiKey}")
if not verify_google_key():
    raise RuntimeError("Invalid Google API Key configuration")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

sessions: Dict[str, Any] = {}


# function to extract the text from the pdf(s)
def get_pdf_text(files: list[UploadFile]) -> str:
    text = ""
    for file in files:
        pdf = PdfReader(file.file)
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text

# function trim and seperate the text
def get_text_chunks(text: str) -> list[str]:
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return text_splitter.split_text(text)


# function to generate
def create_conversation_chain(text_chunks: list[str]):
    # Initialize Google embeddings
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    # Create vector store
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    
    # Initialize Google Chat model
    llm = chat_models.ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3)
    
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )


@app.post("/upload")
async def upload_files(files: list[UploadFile]):
    file_names = []
    for file in files:
        file_names.append(file.filename)
        if file.content_type != 'application/pdf':
            raise HTTPException(400, "Only PDF files allowed")
    print("filenames", file_names)
    

    raw_text = get_pdf_text(files)

    if not raw_text.strip():
        raise HTTPException(400, "No text could be extracted from PDFs")
    
    for file in files:
        if file.content_type != 'application/pdf':
            raise HTTPException(400, "Only PDF files allowed")
                
            # Check file size (10 MB limit)
        if file.size > 10 * 1024 * 1024:
            raise HTTPException(413, "File size exceeds 10MB limit")

    session_id = str(uuid.uuid4())

    text_chunks = get_text_chunks(raw_text)
    conversation_chain = create_conversation_chain(text_chunks)
        
    sessions[session_id] = {
        "conversation": conversation_chain,
        "chunks": text_chunks
    }

    return {"session_id": session_id}

class AskRequest(BaseModel):
    question: str
    session_id: str

@app.post("/ask")
async def ask_test(request_body: AskRequest):
    print(f"Received question in /test_ask: {request_body.question}, type: {type(request_body.question)}")
    print(f"Received question in /test_ask: {request_body.session_id}, type: {type(request_body.session_id)}")

    session_id = request_body.session_id
    question = request_body.question

    if session_id not in sessions:
        raise HTTPException(404, "Session not found")
    
    try:
        conversation_chain = sessions[session_id]["conversation"]
        response = conversation_chain({"question": question})
        return {"answer": response["answer"]}
    
    except Exception as e:
        raise HTTPException(500, f"Error generating answer: {str(e)}")
        

if __name__ == "__main__":
    load_dotenv()
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)