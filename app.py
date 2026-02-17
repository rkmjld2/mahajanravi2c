import streamlit as st
import mysql.connector
import tempfile
import pandas as pd

# LangChain Classic imports
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI

st.title("Blood Reports Database Manager + RAG Analysis with Medicines")

# --- TiDB Config ---
db_config = {
    "host": st.secrets["tidb"]["host"],
    "port": st.secrets["tidb"]["port"],
    "user": st.secrets["tidb"]["user"],
    "password": st.secrets["tidb"]["password"],
    "database": st.secrets["tidb"]["database"],
}

# Write SSL certificate string from secrets to a temporary file
with tempfile.NamedTemporaryFile(delete=False) as tmp:
    tmp.write(st.secrets["tidb"]["ssl_ca"].encode())
    db_config["ssl_ca"] = tmp.name

# --- Helper Function ---
def run_query(query, params=None, fetch=False):
    conn = mysql.connector.connect(**db_config)
    cursor = conn.cursor(dictionary=True)
    cursor.execute(query, params or ())
    result = cursor.fetchall() if fetch else None
    conn.commit()
    cursor.close()
    conn.close()
    return result

# --- Insert Record ---
st.header("➕ Insert Record")
with st.form("insert_form"):
    name = st.text_input("Patient Name")
    test_name = st.text_input("Test Name")
    result = st.number_input("Result", step=0.01)
    unit = st.text_input("Unit")
    ref_range = st.text_input("Reference Range")
    flag = st.text_input("Flag (normal / low / high / abnormal)")
    submitted = st.form_submit_button("Insert")
    if submitted:
        run_query(
            "INSERT INTO blood_reports (name, test_name, result, unit, ref_range, flag) VALUES (%s,%s,%s,%s,%s,%s)",
            (name, test_name, result, unit, ref_range, flag)
        )
        st.success("✅ Record inserted successfully!")

# --- Highlighting Function ---
def highlight_flags(val):
    if val and str(val).lower() in ["low", "abnormal", "high"]:
        return "background-color: red; color: white"
    elif val and str(val).lower() == "normal":
        return "background-color: lightgreen; color: black"
    return ""

# --- Search by Name + Date Range ---
st.header("🔍 Search by Name and Date Range")
search_name = st.text_input("Enter Patient Name")
start_date = st.date_input("Start Date")
end_date = st.date_input("End Date")
if st.button("Search Records"):
    rows = run_query(
        "SELECT * FROM blood_reports WHERE name=%s AND timestamp BETWEEN %s AND %s",
        (search_name, start_date, end_date),
        fetch=True
    )
    if rows:
        df = pd.DataFrame(rows)
        st.dataframe(df.style.applymap(highlight_flags, subset=["flag"]))

# --- Display All Records ---
st.header("📋 All Records")
if st.button("Show All Records"):
    rows = run_query("SELECT * FROM blood_reports", fetch=True)
    if rows:
        df = pd.DataFrame(rows)
        st.dataframe(df.style.applymap(highlight_flags, subset=["flag"]))

# --- RAG Analysis ---
st.header("🧠 RAG: Abnormal Report Analysis & Medicine Recommendations")

if st.button("Run RAG Analysis"):
    # Fetch all records
    rows = run_query("SELECT * FROM blood_reports", fetch=True)

    # Convert rows into text docs
    docs = []
    for r in rows:
        text = f"Patient {r['name']} | Test: {r['test_name']} | Result: {r['result']} {r['unit']} | Ref Range: {r['ref_range']} | Flag: {r['flag']} | Date: {r['timestamp']}"
        docs.append(text)

    # Build FAISS vector store
    embeddings = OpenAIEmbeddings(openai_api_key=st.secrets["openai"]["api_key"])
    vectorstore = FAISS.from_texts(docs, embeddings)

    # Create RAG chain
    llm = OpenAI(openai_api_key=st.secrets["openai"]["api_key"])
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())

    # Ask the model to find abnormal reports AND suggest common medicines
    query = (
        "Identify abnormal blood test reports. For each low value, "
        "recommend common medicines or supplements typically prescribed "
        "to improve that condition. Provide clear, patient-friendly recommendations."
    )
    answer = qa.run(query)

    st.subheader("🔎 AI Recommendations with Medicines")
    st.write(answer)
