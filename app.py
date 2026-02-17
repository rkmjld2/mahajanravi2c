import streamlit as st
import mysql.connector
import tempfile
from datetime import datetime

# ── LangChain imports ───────────────────────────────────────────────
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

st.set_page_config(page_title="Blood Reports Manager + RAG", layout="wide")

st.title("Blood Reports Database Manager + RAG Analysis")

# ── TiDB Config ─────────────────────────────────────────────────────
db_config = {
    "host": st.secrets["tidb"]["host"],
    "port": st.secrets["tidb"]["port"],
    "user": st.secrets["tidb"]["user"],
    "password": st.secrets["tidb"]["password"],
    "database": st.secrets["tidb"]["database"],
}

# Write SSL certificate to temporary file
with tempfile.NamedTemporaryFile(delete=False) as tmp:
    tmp.write(st.secrets["tidb"]["ssl_ca"].encode())
    db_config["ssl_ca"] = tmp.name
    db_config["ssl_verify_cert"] = True

# ── Helper function to run SQL queries ──────────────────────────────
def run_query(query, params=None, fetch=False):
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor(dictionary=True)
        cursor.execute(query, params or ())
        result = cursor.fetchall() if fetch else None
        conn.commit()
    except Exception as e:
        st.error(f"Database error: {e}")
        result = None
    finally:
        cursor.close()
        conn.close()
    return result

# ── Insert Record Form ──────────────────────────────────────────────
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
    if submitted:
        if name and test_name:
            run_query(
                """
                INSERT INTO blood_reports 
                (name, test_name, result, unit, ref_range, flag, timestamp)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (name, test_name, result, unit, ref_range, flag, datetime.now()),
            )
            st.success("✅ Record inserted successfully!")
        else:
            st.warning("Please fill at least Patient Name and Test Name.")


# ── Search Records ──────────────────────────────────────────────────
# ── Search Records ──────────────────────────────────────────────────
st.header("🔍 Search Records")
col1, col2, col3 = st.columns([3, 2, 2])
with col1:
    search_name = st.text_input("Patient Name (exact match)", key="search_name_exact")
with col2:
    start_date = st.date_input("From Date", format="YYYY-MM-DD")
with col3:
    end_date = st.date_input("To Date", format="YYYY-MM-DD")

# We'll store the last search results in session state
if "last_search_rows" not in st.session_state:
    st.session_state.last_search_rows = None
    st.session_state.last_search_name = None

if st.button("Search"):
    if search_name and start_date and end_date:
        # Make end date inclusive (full day)
        from datetime import timedelta
        end_date_inclusive = end_date + timedelta(days=1)

        rows = run_query(
            """
            SELECT * FROM blood_reports 
            WHERE name = %s 
            AND timestamp >= %s 
            AND timestamp < %s
            ORDER BY timestamp DESC
            """,
            (search_name.strip(), start_date, end_date_inclusive),
            fetch=True,
        )
        
        if rows:
            st.session_state.last_search_rows = rows
            st.session_state.last_search_name = search_name.strip()
            st.dataframe(rows)
            st.success(f"Found {len(rows)} record(s) for exact name: {search_name}")
        else:
            st.session_state.last_search_rows = []
            st.info("No records found for this exact name and date range.")
    else:
        st.warning("Please enter patient name and both dates.")
# ── Show All Records ────────────────────────────────────────────────
st.header("📋 All Records")
if st.button("Show All Records"):
    rows = run_query("SELECT * FROM blood_reports ORDER BY timestamp DESC", fetch=True)
    if rows:
        st.dataframe(rows)
    else:
        st.info("No records in the database yet.")

# ── RAG Analysis Section ────────────────────────────────────────────
# ── RAG Analysis Section ────────────────────────────────────────────
st.header("🧠 RAG: Abnormal Reports & Recommendations")

if st.button("Run RAG Analysis (may take 10–30s first time)"):
    with st.spinner("Preparing records + building vector store + analyzing..."):
        
        # Decide which records to analyze
        if st.session_state.get("last_search_rows") is not None:
            rows = st.session_state.last_search_rows
            source_info = f"filtered search results for '{st.session_state.last_search_name}'"
        else:
            rows = run_query("SELECT * FROM blood_reports", fetch=True)
            source_info = "all records in database (no search filter applied)"

        if not rows:
            st.warning("No records available to analyze.")
        else:
            st.info(f"Analyzing {len(rows)} record(s) from: {source_info}")

            # Prepare document texts
            texts = []
            for r in rows:
                texts.append(
                    f"Patient: {r['name']} | Test: {r['test_name']} | "
                    f"Result: {r['result']} {r['unit']} | Ref Range: {r['ref_range']} | "
                    f"Flag: {r['flag']} | Date: {r.get('timestamp', 'N/A')}"
                )

            # Free local embeddings
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )

            vectorstore = FAISS.from_texts(texts, embeddings)
            retriever = vectorstore.as_retriever(search_kwargs={"k": min(5, len(texts))})

            # Groq LLM – using current valid model
            llm = ChatGroq(
                model="llama-3.3-70b-versatile",
                temperature=0.25,
                groq_api_key=st.secrets["groq"]["api_key"],
            )

            # Prompt
            system_prompt = """You are a helpful assistant analyzing blood test results.
Use only the following patient blood report excerpts.
Identify abnormal values (flagged or clearly outside reference range).
Provide general educational insights and possible next steps.
Always end with: "This is not medical advice — consult a qualified physician."

Context (reports):
{context}"""

            prompt = ChatPromptTemplate.from_messages(
                [("system", system_prompt), ("human", "{input}")]
            )

            # Build chain
            combine_docs_chain = create_stuff_documents_chain(llm, prompt)
            rag_chain = create_retrieval_chain(retriever, combine_docs_chain)

            # Run
            query = "Identify any abnormal blood test results and provide general recommendations or interpretations."
            try:
                result = rag_chain.invoke({"input": query})
                st.subheader(f"🔎 AI Analysis (based on {source_info})")
                st.markdown(result["answer"])
            except Exception as e:
                st.error(f"Error during analysis: {str(e)}")

