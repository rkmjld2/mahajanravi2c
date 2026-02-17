import streamlit as st
import mysql.connector
import tempfile
from datetime import datetime

# ── Modern LangChain imports ────────────────────────────────────────
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings                 # ← embeddings (Groq has no embeddings)
from langchain_groq import ChatGroq                           # ← NEW: Groq LLM
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

st.set_page_config(page_title="Blood Reports Manager + RAG", layout="wide")

st.title("Blood Reports Database Manager + RAG Analysis")

# ── TiDB Connection Config ──────────────────────────────────────────
db_config = {
    "host": st.secrets["tidb"]["host"],
    "port": st.secrets["tidb"]["port"],
    "user": st.secrets["tidb"]["user"],
    "password": st.secrets["tidb"]["password"],
    "database": st.secrets["tidb"]["database"],
}

# Write CA certificate to temp file (TiDB usually needs SSL)
with tempfile.NamedTemporaryFile(delete=False) as tmp_ca:
    tmp_ca.write(st.secrets["tidb"]["ssl_ca"].encode())
    db_config["ssl_ca"] = tmp_ca.name
    db_config["ssl_verify_cert"] = True

# ── Helper: run query ───────────────────────────────────────────────
def run_query(query, params=None, fetch=False):
    conn = mysql.connector.connect(**db_config)
    cursor = conn.cursor(dictionary=True)
    cursor.execute(query, params or ())
    result = cursor.fetchall() if fetch else None
    conn.commit()
    cursor.close()
    conn.close()
    return result

# ── Insert new record ───────────────────────────────────────────────
st.header("➕ Insert Record")
with st.form("insert_form"):
    col1, col2 = st.columns(2)
    with col1:
        name = st.text_input("Patient Name")
        test_name = st.text_input("Test Name")
        result = st.number_input("Result", step=0.01, format="%.2f")
    with col2:
        unit = st.text_input("Unit")
        ref_range = st.text_input("Reference Range")
        flag = st.text_input("Flag (e.g. High / Low / Normal)")

    submitted = st.form_submit_button("Insert Record")
    if submitted and name and test_name:
        run_query(
            """
            INSERT INTO blood_reports 
            (name, test_name, result, unit, ref_range, flag, timestamp)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """,
            (name, test_name, result, unit, ref_range, flag, datetime.now()),
        )
        st.success("✅ Record inserted!")
    elif submitted:
        st.warning("Please fill at least Patient Name and Test Name.")

# ── Search ──────────────────────────────────────────────────────────
st.header("🔍 Search Records")
col1, col2, col3 = st.columns([2, 1.2, 1.2])
with col1:
    search_name = st.text_input("Patient Name", key="search_name")
with col2:
    start_date = st.date_input("From", value=datetime.now().date())
with col3:
    end_date = st.date_input("To", value=datetime.now().date())

if st.button("Search"):
    if search_name:
        rows = run_query(
            """
            SELECT * FROM blood_reports 
            WHERE name LIKE %s 
            AND timestamp BETWEEN %s AND %s
            ORDER BY timestamp DESC
            """,
            (f"%{search_name}%", start_date, end_date),
            fetch=True,
        )
        if rows:
            st.dataframe(rows)
        else:
            st.info("No records found.")
    else:
        st.warning("Enter a patient name to search.")

# ── Show all ────────────────────────────────────────────────────────
st.header("📋 All Records")
if st.button("Show All"):
    rows = run_query("SELECT * FROM blood_reports ORDER BY timestamp DESC", fetch=True)
    if rows:
        st.dataframe(rows)
    else:
        st.info("Database is empty.")

# ── RAG Analysis (lazy – only runs when clicked) ───────────────────
st.header("🧠 RAG: Abnormal Reports & Recommendations")

if st.button("Run RAG Analysis (may take 5–20 seconds)"):
    with st.spinner("Fetching records → embedding → vector store → Groq analysis..."):
        rows = run_query("SELECT * FROM blood_reports", fetch=True)

        if not rows:
            st.warning("No reports in database yet.")
        else:
            # Prepare texts
            texts = []
            for r in rows:
                texts.append(
                    f"Patient: {r['name']} | Test: {r['test_name']} | "
                    f"Result: {r['result']} {r['unit']} | Ref: {r['ref_range']} | "
                    f"Flag: {r['flag']} | Date: {r.get('timestamp', '—')}"
                )

            # Embed + FAISS (still using OpenAI embeddings – Groq has none)
            embeddings = OpenAIEmbeddings(openai_api_key=st.secrets["openai"]["api_key"])
            vectorstore = FAISS.from_texts(texts, embeddings)
            retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

            # LLM = Groq (fast & cheap)
            llm = ChatGroq(
                model="llama-3.1-70b-versatile",          # or "mixtral-8x7b-32768", "llama3-70b-8192"
                temperature=0.25,
                groq_api_key=st.secrets["groq"]["api_key"],
            )

            # Prompt
            system_prompt = """You are a helpful medical report analyzer.
Use only the provided blood test excerpts.
Identify values that are clearly abnormal (flagged or outside reference range).
Give general, educational insights only.
Always include: "This is not medical advice — consult a qualified doctor."

Context:
{context}"""

            prompt = ChatPromptTemplate.from_messages(
                [("system", system_prompt), ("human", "{input}")]
            )

            # Chains (using langchain-classic compatibility)
            combine_docs_chain = create_stuff_documents_chain(llm, prompt)
            rag_chain = create_retrieval_chain(retriever, combine_docs_chain)

            # Run
            query = "Identify abnormal blood test results and suggest general next steps or possible interpretations."
            result = rag_chain.invoke({"input": query})

            st.subheader("AI Analysis & Recommendations (powered by Groq)")
            st.markdown(result["answer"])


