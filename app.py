# --- RAG Analysis ---
st.header("🧠 RAG: Abnormal Report Analysis & Recommendations")

# Add these new imports at the top (near other langchain imports)
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI          # ← better than old OpenAI llm

if st.button("Run RAG Analysis"):
    # Fetch all records
    rows = run_query("SELECT * FROM blood_reports", fetch=True)
    
    if not rows:
        st.warning("No records found in the database.")
        st.stop()

    # Convert rows into text docs
    docs = []
    for r in rows:
        text = f"Patient {r['name']} | Test: {r['test_name']} | Result: {r['result']} {r['unit']} | Ref Range: {r['ref_range']} | Flag: {r['flag']} | Date: {r['timestamp']}"
        docs.append(text)

    # Build FAISS vector store
    embeddings = OpenAIEmbeddings(openai_api_key=st.secrets["openai"]["api_key"])
    vectorstore = FAISS.from_texts(docs, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 6})   # ← you can tune k

    # Modern LLM (recommended over old OpenAI class)
    llm = ChatOpenAI(
        model="gpt-4o-mini",                # or gpt-4o, gpt-3.5-turbo, etc.
        openai_api_key=st.secrets["openai"]["api_key"],
        temperature=0.3
    )

    # Define a clear prompt (very important!)
    system_prompt = (
        "You are a helpful medical AI assistant analyzing blood test results.\n"
        "Use the following patient blood report excerpts to identify abnormal values "
        "(those marked with flag or clearly outside reference range) and give reasonable "
        "general recommendations.\n"
        "Do NOT give definitive medical advice — always recommend consulting a doctor.\n\n"
        "{context}"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    # Create chains
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    # Run
    query = "Identify abnormal blood test reports and provide general recommendations."
    result = rag_chain.invoke({"input": query})

    st.subheader("🔎 AI Analysis & Recommendations")
    st.markdown(result["answer"])
