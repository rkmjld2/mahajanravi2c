import streamlit as st
import mysql.connector
import tempfile
from groq import Groq

st.title("RAG Demo: Blood Reports + Groq")

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

# --- Fetch Data from TiDB ---
try:
    conn = mysql.connector.connect(**db_config)
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT id, timestamp, test_name, result, unit, ref_range, flag FROM blood_reports LIMIT 5;")
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    st.success("✅ TiDB Connected and data retrieved!")
    st.write(rows)
except Exception as e:
    st.error(f"❌ TiDB query failed: {e}")
    rows = []

# --- Groq Summarization ---
if rows:
    client = Groq(api_key=st.secrets["groq"]["api_key"])
    # Convert rows into a text block
    report_text = "\n".join([f"{r['timestamp']} - {r['test_name']}: {r['result']} {r['unit']} (Ref: {r['ref_range']}, Flag: {r['flag']})" for r in rows])

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "You are a medical report summarizer."},
                {"role": "user", "content": f"Summarize these blood test results:\n{report_text}"}
            ]
        )
        st.success("✅ Groq Summarization Complete")
        st.write(response.choices[0].message.content)
    except Exception as e:
        st.error(f"❌ Groq summarization failed: {e}")
