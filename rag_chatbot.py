# --- Initial Setup and Imports ---

# Load our secret API key from the .env file
from dotenv import load_dotenv
load_dotenv()

# For data loading
from langchain_community.document_loaders import TextLoader

# For splitting text into chunks
from langchain_text_splitters import RecursiveCharacterTextSplitter

# For creating the embeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

# For the vector store
from langchain_community.vectorstores import FAISS

# For the LLM
from langchain_anthropic import ChatAnthropic

# For building the RAG chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- Part 1: Data Loading, Splitting, and Storing ---

# 1. Load the document from the text file
print("Loading HR policy document...")
loader = TextLoader("Lesson7_RAG_Chatbot/hr_policy.txt")
docs = loader.load()

print(f"Loaded {len(docs)} document(s).")

# 2. Split the document into smaller chunks
print("Splitting document into chunks...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(docs)

print(f"Split document into {len(chunks)} chunks.")

# 3. Create embeddings and store them in a FAISS vector store
print("Creating embeddings and building FAISS index...")

# For this, we'll use a powerful, open-source model from Hugging Face
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


# Create the vector store from our chunks and embeddings
vector_store = FAISS.from_documents(chunks, embeddings)
print("FAISS index built successfully.")

# --- Part 2: Building the RAG Chain and Chatbot ---

# 4. Initialize the LLM (we'll use Claude for this example)
print("Initializing LLM...")
llm = ChatAnthropic(model="claude-sonnet-4-5", temperature=0.3)

print("LLM initialized.")

# 5. Define the prompt for the RAG chain
rag_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful and professional HR assistant. Answer the user's question ONLY based on the provided context. If the answer is not in the context, politely state that you don't have enough information.\\\\n\\\\nContext:\\\\n{context}"),
    ("human", "Question: {input}")
])

# 6. Create the retriever from the vector store
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# 7. Build the RAG chain using Runnable-based pipeline
rag_chain = (
    {
        "context": retriever,
        "input": RunnablePassthrough(),
    }
    | rag_prompt
    | llm
    | StrOutputParser()
)

# --- Part 3: Running the Chatbot ---

print("\nWelcome to the TerraBrew HR Chatbot! Ask me anything about HR policies (type 'exit' to quit).")

while True:
    user_question = input("\nYour Question: ")

    if user_question.lower() == "exit":
        print("Goodbye!")
        break

    # Invoke the RAG chain directly with the user's question
    response = rag_chain.invoke(user_question)

    # Print the answer from the LLM
    print(f"\nBot: {response}\n")
