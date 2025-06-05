
                                                               **Implementing using Hugging Face Space**
   
                                           **RAG-based Ticket Summarizer** : https://huggingface.co/spaces/apatra06/rag-ticket-summarizer

                                                      If you need code Kindly click the above link and go to the code base 

                                             **Note**: This projects i ll update to upgrade the Rag ticket summarizer application 

                                                                      # 🧠 RAG-based Ticket Summarizer

A **Retrieval-Augmented Generation (RAG)** application for summarizing customer support tickets. This tool helps customer support teams quickly understand issue context and resolution history using intelligent semantic search + summarization.

🔗 **Live on Hugging Face Spaces**:  
👉 [Try it here](https://huggingface.co/spaces/apatra06/rag-ticket-summarizer)



## 📁 File Structure

rag-ticket-summarizer/
│
├── .streamlit/                # Streamlit configuration files
├── src/                       # Core app code (e.g., streamlit\app.py)
├── tickets.csv                # Sample dataset of support tickets
├── Dockerfile                 # Deployment configuration for containerized builds
├── requirements.txt           # Python dependencies
├── README.md                  # You're here!
└── .gitattributes             # GitHub attributes file

## ✅ Features

- 🔍 **Semantic Search**: Finds the most relevant past ticket using sentence embeddings and FAISS.
- ✍️ **LLM Summarization**: Generates short, clear summaries using pretrained transformers (distilBART).
- 🧠 **End-to-End RAG**: Combines retrieval + generation to provide accurate, contextual answers.
- ☁️ **Deployed via Hugging Face**: Fully interactive UI using Streamlit.
- 📤 **CSV File Support**: Easily upload your own support ticket data.

## 📊 Sample Dataset (`tickets.csv`)

The CSV file contains historical support tickets with two key fields:

- `body`: Description of the customer’s issue
- `resolution`: How the issue was resolved

Example:

csv
body,resolution
"Customer unable to reset password via email",
"Guided user to clear cache and resend reset link"
"Payment failed on checkout","Customer used a different card to complete transaction"


## ⚙️ Installation (For Local Use)

git clone https://huggingface.co/spaces/apatra06/rag-ticket-summarizer
cd rag-ticket-summarizer
pip install -r requirements.txt
streamlit run src/streamlit_app.py


## 💡 How It Works

1. Loads historical ticket data from `tickets.csv`.
2. Encodes all ticket `body` texts into embeddings using `all-MiniLM-L6-v2`.
3. Uses `FAISS` for fast semantic search on new queries.
4. Summarizes the best-matching ticket + resolution using a transformer-based model.
5. Presents the output in an intuitive Streamlit UI.

## 🧠 Technologies Used

| Purpose       | Tool/Library                     |
| ------------- | -------------------------------- |
| Embeddings    | `sentence-transformers` (MiniLM) |
| Vector Search | `faiss-cpu`                      |
| Summarization | `transformers` (DistilBART)      |
| Frontend      | `Streamlit`                      |
| Deployment    | `Hugging Face Spaces`            |
| Dataset       | CSV file (`tickets.csv`)         |

## 🧪 Example Query

> **Input:** "How do I change my registered phone number?"

> **Output:**
> "Customer was guided through profile settings to update phone number and verify with OTP."

---

## ✍️ Author
🔗 [Hugging Face Profile](https://huggingface.co/apatra06)



>>>>>>> 9fce9f3 ( added)
