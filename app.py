import streamlit as st
import mysql.connector
import tempfile
from groq import Groq

st.title("TiDB + Groq Connection Test")

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

# --- Test TiDB Connection ---
try:
    conn = mysql.connector.connect(**db_config)
    cursor = conn.cursor()
    cursor.execute("SELECT NOW();")
    result = cursor.fetchone()
    st.success(f"✅ TiDB Connected! Current DB time: {result[0]}")
    cursor.close()
    conn.close()
except Exception as e:
    st.error(f"❌ TiDB Connection failed: {e}")

# --- Test Groq API ---
try:
    client = Groq(api_key=st.secrets["groq"]["api_key"])
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": "Hello Groq!"}]
    )
    st.success(f"✅ Groq Connected! Response: {response.choices[0].message.content}")
except Exception as e:
    st.error(f"❌ Groq Connection failed: {e}")
