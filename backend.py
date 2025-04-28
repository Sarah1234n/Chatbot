from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from pydantic import BaseModel
from openai import OpenAI
from fastapi.responses import StreamingResponse
from dotenv import load_dotenv
import json
import psycopg2
import os
import uuid
from psycopg2.extras import RealDictCursor
from typing import List, Optional
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage, AIMessage
from azure.storage.blob import BlobClient
import chromadb

load_dotenv()

DB_CONFIG = {
    "dbname": os.environ.get("DB_NAME"),
    "user": os.environ.get("DB_USER"),
    "password": os.environ.get("DB_PASSWORD"),
    "host": os.environ.get("DB_HOST"),
    "port": os.environ.get("DB_PORT"),
}

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
model = "gpt-3.5-turbo"
llm = ChatOpenAI(model=model)

embedding_function = OpenAIEmbeddings()
chroma_client = chromadb.HttpClient(host=os.environ.get("CHROMADB_HOST"), port=os.environ.get("CHROMADB_PORT"))
collection = chroma_client.get_or_create_collection("langchain")
vectorstore = Chroma(
    client=chroma_client,
    collection_name="langchain",
    embedding_function=embedding_function,
)

storage_account_sas_url = os.environ.get("AZURE_STORAGE_SAS_URL")
storage_container_name = os.environ.get("AZURE_STORAGE_CONTAINER")
storage_resource_uri = storage_account_sas_url.split('?')[0]
token = storage_account_sas_url.split('?')[1]

app = FastAPI()

# Request models
class ChatRequest(BaseModel):
    messages: List[dict]

class SaveChatRequest(BaseModel):
    chat_id: str
    chat_name: str
    messages: List[dict]
    pdf_name: Optional[str] = None
    pdf_path: Optional[str] = None
    pdf_uuid: Optional[str] = None

class DeleteChatRequest(BaseModel):
    chat_id: str

class RAGChatRequest(BaseModel):
    messages: List[dict]
    pdf_uuid: str

# Dependency to manage database connection
def get_db():
    conn = psycopg2.connect(**DB_CONFIG)
    try:
        yield conn
    finally:
        conn.close()

# Initialize DB: create table if not exists
def init_db():
    conn = psycopg2.connect(**DB_CONFIG)
    try:
        with conn.cursor() as cursor:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS advanced_chats (
                    id VARCHAR PRIMARY KEY,
                    name VARCHAR,
                    file_path VARCHAR,
                    last_update TIMESTAMP,
                    pdf_path VARCHAR,
                    pdf_name VARCHAR,
                    pdf_uuid VARCHAR
                )
            """)
            conn.commit()
    finally:
        conn.close()

# Call it on startup
init_db()

@app.post("/chat/")
async def chat(request: ChatRequest):
    try:
        stream = client.chat.completions.create(
            model=model,
            messages=request.messages,
            stream=True,
        )

        def stream_response():
            for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta

        return StreamingResponse(stream_response(), media_type="text/plain")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/load_chat/")
async def load_chat(db: psycopg2.extensions.connection = Depends(get_db)):
    try:
        with db.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute("SELECT id, name, file_path, pdf_name, pdf_path, pdf_uuid FROM advanced_chats ORDER BY last_update DESC")
            rows = cursor.fetchall()

        records = []
        for row in rows:
            chat_id, name, file_path, pdf_name, pdf_path, pdf_uuid = row["id"], row["name"], row["file_path"], row["pdf_name"], row["pdf_path"], row["pdf_uuid"]
            blob_sas_url = f"{storage_resource_uri}/{storage_container_name}/{file_path}?{token}"
            blob_client = BlobClient.from_blob_url(blob_sas_url)
            if blob_client.exists():
                blob_data = blob_client.download_blob().readall()
                messages = json.loads(blob_data)
                records.append({"id": chat_id, "chat_name": name, "messages": messages, "pdf_name": pdf_name, "pdf_path": pdf_path, "pdf_uuid": pdf_uuid})

        return records

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.post("/save_chat/")
async def save_chat(request: SaveChatRequest, db: psycopg2.extensions.connection = Depends(get_db)):
    try:
        file_path = f"chat_logs/{request.chat_id}.json"
        blob_sas_url = f"{storage_resource_uri}/{storage_container_name}/{file_path}?{token}"
        blob_client = BlobClient.from_blob_url(blob_sas_url)
        messages_data = json.dumps(request.messages, ensure_ascii=False, indent=4)
        blob_client.upload_blob(messages_data, overwrite=True)

        with db.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO advanced_chats (id, name, file_path, last_update, pdf_path, pdf_name, pdf_uuid)
                VALUES (%s, %s, %s, CURRENT_TIMESTAMP, %s, %s, %s)
                ON CONFLICT (id)
                DO UPDATE SET name = EXCLUDED.name, file_path = EXCLUDED.file_path, last_update = CURRENT_TIMESTAMP, pdf_path = EXCLUDED.pdf_path, pdf_name = EXCLUDED.pdf_name, pdf_uuid = EXCLUDED.pdf_uuid
                """,
                (request.chat_id, request.chat_name, file_path, request.pdf_path, request.pdf_name, request.pdf_uuid),
            )
        db.commit()
        return {"message": "Chat saved successfully"}

    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.post("/delete_chat/")
async def delete_chat(request: DeleteChatRequest, db: psycopg2.extensions.connection = Depends(get_db)):
    try:
        file_path = None
        with db.cursor() as cursor:
            cursor.execute("SELECT file_path, pdf_path FROM advanced_chats WHERE id = %s", (request.chat_id,))
            result = cursor.fetchone()
            if result:
                file_path, pdf_path = result
            else:
                raise HTTPException(status_code=404, detail="Chat not found")

        with db.cursor() as cursor:
            cursor.execute("DELETE FROM advanced_chats WHERE id = %s", (request.chat_id,))
        db.commit()

        if file_path:
            blob_sas_url = f"{storage_resource_uri}/{storage_container_name}/{file_path}?{token}"
            blob_client = BlobClient.from_blob_url(blob_sas_url)
            if blob_client.exists():
                blob_client.delete_blob()

        if pdf_path:
            blob_sas_url = f"{storage_resource_uri}/{storage_container_name}/{pdf_path}?{token}"
            blob_client = BlobClient.from_blob_url(blob_sas_url)
            if blob_client.exists():
                blob_client.delete_blob()

        return {"message": "Chat deleted successfully"}

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.post("/upload_pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")
    try:
        pdf_uuid = str(uuid.uuid4())
        file_path = f"pdf_store/{pdf_uuid}_{file.filename}"
        os.makedirs("pdf_store", exist_ok=True)
        with open(file_path, "wb") as f:
            f.write(await file.read())

        blob_sas_url = f"{storage_resource_uri}/{storage_container_name}/{file_path}?{token}"
        blob_client = BlobClient.from_blob_url(blob_sas_url)
        blob_client.upload_blob(file_path, overwrite=True)

        loader = PyPDFLoader(file_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        texts = text_splitter.split_documents(documents)

        vectorstore.add_texts(
            [doc.page_content for doc in texts],
            ids=[str(uuid.uuid4()) for _ in texts],
            metadatas=[{"pdf_uuid": pdf_uuid} for _ in texts]
        )

        os.remove(file_path)

        return {"message": "File uploaded successfully", "pdf_path": file_path, "pdf_uuid": pdf_uuid}
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@app.post("/rag_chat/")
async def rag_chat(request: RAGChatRequest):
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 5, "filter": {"pdf_uuid": request.pdf_uuid}}
    )

    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question."
        "\n\n"
        "{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    chat_history = []
    for message in request.messages:
        if message["role"] == "user":
            chat_history.append(HumanMessage(content=message["content"]))
        if message["role"] == "assistant":
            chat_history.append(AIMessage(content=message["content"]))

    chain = rag_chain.pick("answer")
    stream = chain.stream({
        "chat_history": chat_history,
        "input": request.messages[-1]
    })

    def stream_response():
        for chunk in stream:
            yield chunk

    return StreamingResponse(stream_response(), media_type="text/plain")
