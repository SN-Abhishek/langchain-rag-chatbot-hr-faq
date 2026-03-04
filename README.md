## RAG Chatbot for HR Policies
A custom chatbot built with LangChain that provides instant, accurate answers to questions from a company's internal HR policy document.
<img width="686" height="562" alt="Screenshot 2026-03-04 at 7 09 48 PM" src="https://github.com/user-attachments/assets/414be727-9930-4e25-88ca-5844118b5d81" />

### Features
* **Context-Aware Responses:** Answers questions based solely on the provided PDF document.
* **Prevents Hallucinations:** Designed to respond with "I don't know" if the answer is not in the document.
* **Easy to Use:** Simple command-line interface for asking questions.

### Tech Stack
* **Framework:** LangChain
* **Language:** Python
* **LLM:** Anthropic Claude (via API)
* **Vector Store:** FAISS (for local in-memory storage)
* **Environment Variables:** python-dotenv


### How to Run This Project Locally
1. **Clone the repository:**
```bash
git clone <https://github.com/your-username/langchain-rag-chatbot-hr-faq.git>
cd langchain-rag-chatbot-hr-faq
```
2. **Create and activate a virtual environment:**
```bash
python -m venv venv
source venv/bin/activate # On Windows, use `venv\\\\Scripts\\\\activate`
```
3. **Install the required dependencies:**
* "Now, for this next step, we need to make sure we have our `requirements.txt`
file. I'm going to quickly jump to my terminal and generate it."
* (Visual: Presenter switches to the terminal and runs `pip freeze >
requirements.txt`. The new file appears in the file explorer.)
* "Perfect. Now that the file exists, the instruction is simple:"
```bash
pip install -r requirements.txt
```
4. **Set up your environment variables:**
* "Remember we talked about never sharing your secret keys? This is how you tell
others to use their own."
```bash
Create a file named `.env` in the root directory and add your Anthropic API key:
ANTHROPIC_API_KEY="your_api_key_here"
```
5. **Run the application:**
```bash
python app.py
```


### Project Impact
This project serves as a powerful demonstration of how Retrieval-Augmented Generation (RAG) can be used to create highly accurate and secure chatbots for business use. By grounding the model's responses in a specific knowledge base, it effectively eliminates the risk of hallucination and ensures that the information provided is always relevant and trusted.

