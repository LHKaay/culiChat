from flask import Flask, request
import requests


from langchain_community.document_loaders import  PyPDFLoader
import os

from langchain_text_splitters.character import  RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_google_genai import ChatGoogleGenerativeAI

app = Flask(__name__)

os.environ["GOOGLE_API_KEY"]  = 'AIzaSyBG7pjT2RVa0vQmpWWkvt5LqQAgotmKJb4'

file_path = "documents/3.pdf"
loader = PyPDFLoader(file_path)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=200
)
doc_chunks = text_splitter.split_documents(docs)
#Embedding các chunks này và đưa vào vector store
embeddings = HuggingFaceEmbeddings()
vectordb = FAISS.from_documents(doc_chunks, embeddings)

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
retriever = vectordb.as_retriever(
        search_kwargs={
                "k": 2, # sl kq truy vấn trả về
    }
    )
memory = ConversationBufferMemory(
        llm=llm,
        output_key="answer",
        memory_key="chat_history",
        return_messages=True
    )
chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        chain_type="map_reduce",
        memory=memory,
        verbose=True
    )


VERIFY_TOKEN = "LHKaayishandsome"
PAGE_ACCESS_TOKEN = "EAAtVlnB73fEBO7i86nJkdyz1Q1ma7TcwTaZCICVZBsijsWdONreafKfCqkECsx2j5jDK4LI3XSMZA0UFNZBJwpsKMGuXWUtF1wJZBgmrokbfW2VULiotdsd2rBd0RhbquAZCSVTRlDnPaQL2YvRMkMMOi0nc7h8pCTmskQNChG9s5ZCWwwd5tuCMpAzeKXtP51go5ouaUG0JEsgwUI6CwZDZD"

@app.route('/webhook', methods=['GET'])
def verify_webhook():
    if request.args.get("hub.verify_token") == VERIFY_TOKEN:
        return request.args.get("hub.challenge")
    return "Verification token mismatch", 403

@app.route('/webhook', methods=['POST'])
def handle_message():
    data = request.get_json()
    for entry in data.get('entry', []):
        for message_event in entry.get('messaging', []):
            sender_id = message_event['sender']['id']
            if 'message' in message_event:
                user_message = message_event['message'].get('text', '')
                response = generate_response(user_message)
                send_message(sender_id, response)
    return "OK", 200

def generate_response(user_message):
    memory.clear()
    # Define your rules
    response = chain({"question": user_message + " hãy trả lời tất cả bằng tiếng Việt", "chat_history": memory.chat_memory.messages})
    assistant_response = response["answer"]

    return assistant_response

def send_message(recipient_id, message_text):
    url = f"https://graph.facebook.com/v15.0/me/messages?access_token={PAGE_ACCESS_TOKEN}"
    headers = {"Content-Type": "application/json"}
    payload = {
        "recipient": {"id": recipient_id},
        "message": {"text": message_text}
    }
    response = requests.post(url, headers=headers, json=payload)
    return response.json()

if __name__ == '__main__':
    app.run(port=5000)
