import os
import streamlit as st
import pickle
import time
from langchain import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()

st.title("Chintu Bot: News Research Analysis Tool 📈")
st.sidebar.title("Enter your News Article URLs")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
file_path = "faiss_store_openai.pkl"
main_placeholder = st.empty()
llm = OpenAI(temperature=0.9, max_tokens=500)

if process_url_clicked:
    # Load data
    loader = UnstructuredURLLoader(urls=urls)
    main_placeholder.text("Data Loading... Started...✅✅✅")
    data = loader.load()

    if not data:
        st.error("Failed to load data from the provided URLs. Please check the URLs and try again.")
    else:
        # Split data
        text_splitter = RecursiveCharacterTextSplitter(
            separators=['\n\n', '\n', '.', ','],
            chunk_size=1000
        )
        main_placeholder.text("Text Splitting... Started...✅✅✅")
        docs = text_splitter.split_documents(data)

        if not docs:
            st.error("No documents were found after splitting. Please check the URLs.")
        else:
            # Create embeddings
            embeddings = OpenAIEmbeddings()
            doc_embeddings = embeddings.embed_documents([doc.page_content for doc in docs])

            if len(doc_embeddings) == 0:
                st.error("Failed to generate embeddings. Please check the input documents.")
            else:
                # Create FAISS index
                vectorstore_openai = FAISS.from_documents(docs, embeddings)
                main_placeholder.text("Embedding Vector Started Building...✅✅✅")
                time.sleep(2)

                # Save the FAISS index to a pickle file
                with open(file_path, "wb") as f:
                    pickle.dump(vectorstore_openai, f)

query = main_placeholder.text_input("Question: ")
if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)
            chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
            result = chain({"question": query}, return_only_outputs=True)

            st.header("Answer")
            st.write(result["answer"])

            sources = result.get("sources", "")
            if sources:
                st.subheader("Sources:")
                sources_list = sources.split("\n")
                for source in sources_list:
                    st.write(source)
    else:
        st.error("FAISS index file not found. Please process the URLs first.")
