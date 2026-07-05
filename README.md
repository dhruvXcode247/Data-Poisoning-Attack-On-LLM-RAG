## Data Poisoning Attack on RAG Apps

### **📚 FAISS-based Text Search**
A simple pipeline to chunk text, generate embeddings, and perform **semantic search** using **faiss** for fast retrieval.  

### 👾 **Data Poisoning Attack on RAG** 
Create a **malicious .txt** file and inject it into the vectordb to **poison the knowledge** and the RAG App. 

### ⚠️ **Results**

| Base RAG Response | Data Poison Attack: RAG Response |
| :---: | :---: | 
| ![Base RAG Response](img/Example%20RAG%20Output.jpg) | ![Data Poison Attack: RAG Response](img/Poisoned%20RAG%20Output.jpg) |

---

### 📦 **Setup**

1. **clone the repository**  
    ```bash
    git clone https://github.com/dhruvXcode247/Data-Poisoning-Attack-On-LLM-RAG.git
    cd Data-Poisoning-Attack-on-LLM-RAG
    ```

2. **create a virtual environment (optional)**  
    ```bash
    python -m venv venv
    source venv/bin/activate  # windows: venv\Scripts\activate
    ```

3. **install dependencies**  
    ```bash
    pip install -r requirements.txt
    ```

---

### 📊 **Usage**

1. **Get Text Files**: <br>Run [create_dataset.ipynb](create_dataset.ipynb) to create a dataset from scratch.
    ```bash
    export GROQ_API_KEY="<groq-api-key>"
    ```

2. **Create VectorDB**:  <br>Run [create_vectorstore.ipynb](create_vectorstore.ipynb) to generate the HuggingFace Embeddings ([intfloat/multilingual-e5-small](https://huggingface.co/intfloat/multilingual-e5-small)) and store in FAISS Vector Store.


3. **Poison Data Injection**: <br>Run [poison_data.py](poison_data.py) script which adds `malicious payload` to single .txt file, and creates relevant vector stores for retreival. 
    ```bash
    python poison_data.py
    ```

4. **Run Streamlit App** <br>Run the [app.py](app.py) python script.
    ```bash
    streamlit start app.py
    ```

---

### 🚀 **Features**
- load and process `.txt` files from a directory  
- use `HuggingFace` embeddings for vector representation  
- store and retrieve chunks with **faiss**  
- persistent storage with **faiss** index + **json**  

--- 

### 📜 **todo**
- [x] create streamlit app for user interaction
- [x] create poisoned .txt file  
- [x] add poisoned .txt into vector store
- [x] include vector store selection (base/poison) in streamlit app
- [x] test data poisoning vulnerability 

---

### 🛠️ **Tech stack**
- LangChain
- FAISS
- Groq API

# Data-Poisoning-Attack-On-LLM-RAG
This repository implements a data poisoning attack done on a LLM-RAG application hosted on Streamlit using FAISS.
