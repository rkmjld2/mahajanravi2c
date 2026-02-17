import streamlit as st
import mysql.connector
import tempfile
from datetime import datetime, timedelta
import pandas as pd
from io import StringIO

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
                (name.strip(), test_name, result, unit, ref_range, flag, datetime.now()),
            )
            st.success("✅ Record inserted successfully!")
        else:
            st.warning("Please fill at least Patient Name and Test Name.")

# ── Search Records (EXACT name match) ───────────────────────────────
st.header("🔍 Search Records")
col1, col2, col3 = st.columns([3, 2, 2])
with col1:
    search_name = st.text_input("Patient Name (exact match required)", key="search_name_exact")
with col2:
    start_date = st.date_input("From Date", format="YYYY-MM-DD")
with col3:
    end_date = st.date_input("To Date", format="YYYY-MM-DD")

# Store last search results in session state
if "last_search_rows" not in st.session_state:
    st.session_state.last_search_rows = None
    st.session_state.last_search_name = None

if st.button("Search"):
    if search_name and start_date and end_date:
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
            st.success(f"Found {len(rows)} record(s) for exact name: {search_name.strip()}")
        else:
            st.session_state.last_search_rows = []
            st.info("No records found for this exact name and date range.")
    else:
        st.warning("Please enter patient name and both dates.")

    # Download button for searched records
    if st.session_state.get("last_search_rows") and st.session_state.last_search_rows:
        df_search = pd.DataFrame(st.session_state.last_search_rows)
        csv_search = df_search.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Searched Records (CSV)",
            data=csv_search,
            file_name=f"blood_reports_{st.session_state.last_search_name}.csv",
            mime="text/csv",
            key="download_searched"
        )

# ── Show All Records ────────────────────────────────────────────────
st.header("📋 All Records")
if st.button("Show All Records"):
    rows = run_query("SELECT * FROM blood_reports ORDER BY timestamp DESC", fetch=True)
    if rows:
        df_all = pd.DataFrame(rows)
        st.dataframe(df_all)

        # Download button for all records
        csv_all = df_all.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download All Records (CSV)",
            data=csv_all,
            file_name="blood_reports_all.csv",
            mime="text/csv",
            key="download_all"
        )
    else:
        st.info("No records in the database yet.")

# ── RAG Analysis ────────────────────────────────────────────────────
st.header("🧠 RAG: Abnormal Reports & Recommendations")

if st.button("Run RAG Analysis (may take 10–30s first time)"):
    with st.spinner("Preparing records + building vector store + analyzing..."):
        
        # Decide which records to analyze
        if st.session_state.get("last_search_rows") is not None and st.session_state.last_search_rows:
            rows = st.session_state.last_search_rows
            source_info = f"filtered search results for exact name '{st.session_state.last_search_name}'"
        else:
            rows = run_query("SELECT * FROM blood_reports", fetch=True)
            source_info = "ALL records in database (no search filter applied yet)"

        if not rows:
            st.warning("No records available to analyze. Please insert or search for records first.")
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

            # Embeddings
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )

            vectorstore = FAISS.from_texts(texts, embeddings)
            retriever = vectorstore.as_retriever(search_kwargs={"k": min(5, len(texts))})

            # LLM
            llm = ChatGroq(
                model="llama-3.3-70b-versatile",
                temperature=0.3,
                groq_api_key=st.secrets["groq"]["api_key"],
            )

            # Updated prompt with medicine suggestions
            system_prompt = """You are a helpful educational assistant summarizing blood test results.
Use ONLY the provided report excerpts below.
Your response MUST include:

1. Identification of clearly abnormal values (where flag is 'High'/'Low' or result is outside ref range).
2. For EACH abnormal test: very brief general interpretation.
3. For EACH abnormal test: common lifestyle recommendations (diet, exercise, habits) AND typical/commonly associated medicines, supplements or treatments (e.g. statins for high cholesterol, iron for low hemoglobin, vitamin D for deficiency, etc.).

VERY IMPORTANT – ALWAYS INCLUDE THIS EXACT DISCLAIMER AT  END OF YOUR RESPONSE:
"THIS IS GENERAL EDUCATIONAL INFORMATION ONLY – NOT MEDICAL ADVICE, NOT A DIAGNOSIS, NOT A TREATMENT PLAN. 
DO NOT TAKE ANY MEDICATION OR SUPPLEMENT BASED ON THIS OUTPUT. 
CONSULT A QUALIFIED DOCTOR FOR PERSONALIZED INTERPRETATION, DIAGNOSIS AND PRESCRIPTION."

Never recommend specific doses, brands or starting/stopping medicines.
Keep response clear, structured and concise.

Context (blood reports):
{context}"""

            prompt = ChatPromptTemplate.from_messages(
                [("system", system_prompt), ("human", "{input}")]
            )

            combine_docs_chain = create_stuff_documents_chain(llm, prompt)
            rag_chain = create_retrieval_chain(retriever, combine_docs_chain)

            query = "Identify abnormal blood test results, explain briefly, list common general recommendations and typical medicines/supplements for each abnormal parameter."
            try:
                result = rag_chain.invoke({"input": query})
                answer_text = result["answer"]

                st.subheader(f"🔎 AI Analysis (based on {source_info})")
                st.markdown(answer_text)

                # Download RAG result as text
                st.download_button(
                    label="📥 Download RAG Analysis Result (TXT)",
                    data=answer_text,
                    file_name="rag_analysis_abnormal_reports.txt",
                    mime="text/plain",
                    key="download_rag"
                )

            except Exception as e:
                st.error(f"Error during analysis: {str(e)}")


