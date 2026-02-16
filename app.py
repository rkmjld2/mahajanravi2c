import streamlit as st
import mysql.connector
import tempfile
from groq import Groq

st.title("RAG Demo: Blood Reports Assistant")

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
    # --- Query TiDB ---
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor(dictionary=True)

        # Simple retrieval: fetch relevant rows
        # For demo, we just pull all rows; later you can add WHERE clauses or embeddings
        cursor.execute("SELECT id, timestamp, test_name, result, unit, ref_range, flag FROM blood_reports LIMIT 20;")
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
        st.success("✅ TiDB Connected and data retrieved!")
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
                    {"role": "user", "content": f"Question: {user_question}\n\nHere are the blood reports:\n{report_text}"}
                ]
            )
            st.markdown("### 🧾 Answer")
            st.write(response.choices[0].message.content)
        except Exception as e:
            st.error(f"❌ Groq summarization failed: {e}")
