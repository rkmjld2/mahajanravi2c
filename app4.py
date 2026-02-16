import streamlit as st
import mysql.connector
import tempfile
from groq import Groq

st.title("RAG Demo: Blood Reports Assistant (Semantic Filtering)")

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

# --- User Query ---
user_question = st.text_input("Ask about blood reports (e.g., 'Show me abnormal glucose results')")

if user_question:
    # --- Simple semantic filter: extract keywords ---
    # For demo, we just look for test names mentioned in the question
    keywords = []
    for kw in ["glucose", "cholesterol", "hemoglobin", "platelet", "WBC", "RBC"]:
        if kw.lower() in user_question.lower():
            keywords.append(kw)

    # --- Query TiDB ---
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor(dictionary=True)

        if keywords:
            # Build WHERE clause dynamically
            conditions = " OR ".join([f"test_name LIKE '%{kw}%'" for kw in keywords])
            query = f"SELECT id, timestamp, test_name, result, unit, ref_range, flag FROM blood_reports WHERE {conditions} LIMIT 50;"
        else:
            # Fallback: fetch all rows if no keyword detected
            query = "SELECT id, timestamp, test_name, result, unit, ref_range, flag FROM blood_reports LIMIT 20;"

        cursor.execute(query)
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
        st.success(f"✅ TiDB Connected and retrieved {len(rows)} rows")
    except Exception as e:
        st.error(f"❌ TiDB query failed: {e}")
        rows = []

    # --- Pass to Groq ---
    if rows:
        client = Groq(api_key=st.secrets["groq"]["api_key"])
        report_text = "\n".join([
            f"{r['timestamp']} - {r['test_name']}: {r['result']} {r['unit']} (Ref: {r['ref_range']}, Flag: {r['flag']})"
            for r in rows
        ])

        try:
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": "You are a medical assistant that answers questions based on blood test reports."},
                    {"role": "user", "content": f"Question: {user_question}\n\nRelevant blood reports:\n{report_text}"}
                ]
            )
            st.markdown("### 🧾 Answer")
            st.write(response.choices[0].message.content)
        except Exception as e:
            st.error(f"❌ Groq summarization failed: {e}")
