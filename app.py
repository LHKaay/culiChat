import os, requests
from typing_extensions import List, TypedDict
from flask import Flask, request

from langchain import hub

from langchain_community.document_loaders import  PyPDFLoader
from langchain_community.vectorstores import InMemoryVectorStore, FAISS
from langchain_core.documents import Document
from langchain_text_splitters.character import  RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_vertexai import VertexAIEmbeddings


app = Flask(__name__)

os.environ["GOOGLE_API_KEY"]  = 'AIzaSyBG7pjT2RVa0vQmpWWkvt5LqQAgotmKJb4'

file_path = "documents/4.pdf"
loader = PyPDFLoader(file_path)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=300
)
doc_chunks = text_splitter.split_documents(docs)
#Embedding document chunks and stored
embeddings = HuggingFaceEmbeddings()
# embeddings = VertexAIEmbeddings(model_name='text-multilingual-embedding-002')

vector_store = InMemoryVectorStore(embeddings)
# vector_store = FAISS(embedding_function=embeddings)
# vector_store = faiss.add_embeddings(text_embeddings=embeddings)
_ = vector_store.add_documents(documents=doc_chunks)

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-001")

prompt = hub.pull("rlm/rag-prompt")
prompt[0].prompt.template = "Your name is Long. You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Do not memorize and answer in VietNamese language. \nQuestion: {question} \nContext: {context} \nAnswer:"

# Define state for application
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}

def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])

    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}

# Compile application and test
graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()

VERIFY_TOKEN = "LHKaayishandsome"
PAGE_ACCESS_TOKEN = "EAAHPPjR8ctABO8hc4u8vHZAT8tWPaXoxbTY7xpJEyEleNrMZBI8ZB5WZCUgbX8kubS77wP7XFdl7FzEpPUZBIpsQ2UEt81HXUoYi84DVHFjDSg7ztLzdWbRWmZA673VTkcnREjP2Pd8FHVtk0t3ogDtRjZChgxVm0QibVybmoD0ZAzOZBzdoB0qx5RDkM6YXjkhPjZAsWTGcsji88XABxLHgZDZD"

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
            if 'message' in message_event and 'is_echo' not in message_event['message']:
                user_message = message_event['message'].get('text', '')
                response = generate_response(user_message)
                send_message(sender_id, response)

    return "OK", 200

def generate_response(user_message):
    print('#################################')
    print(f'User: {user_message}')

    response = graph.invoke({"question": user_message})
    answer = response["answer"]

    # print('---------------Start Relevant document------------------')
    # for doc in response['context']:
    #     print(doc.page_content)
    #     print('xxxxxxxxx')
    # print('---------------End Relevant document------------------')
    print(f'Culi: {answer}')
    print('#################################')

    return answer

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
