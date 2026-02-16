import streamlit as st
import mysql.connector
import tempfile
from groq import Groq
import faiss
import numpy as np

st.title("RAG Demo: Blood Reports Assistant (Embeddings + Vector Search)")

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
def fetch_reports():
    conn = mysql.connector.connect(**db_config)
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT id, timestamp, test_name, result, unit, ref_range, flag FROM blood_reports LIMIT 200;")
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    return rows

# --- Build Embeddings Index ---
def build_index(rows, client):
    texts = [
        f"{r['timestamp']} - {r['test_name']}: {r['result']} {r['unit']} (Ref: {r['ref_range']}, Flag: {r['flag']})"
        for r in rows
    ]
    embeddings = []
    for txt in texts:
        emb = client.embeddings.create(model="llama-3.1-8b-embedding", input=txt)
        embeddings.append(emb.data[0].embedding)
    embeddings = np.array(embeddings).astype("float32")

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index, texts

# --- Main Flow ---
user_question = st.text_input("Ask about blood reports (semantic search enabled)")

if user_question:
    client = Groq(api_key=st.secrets["groq"]["api_key"])
    rows = fetch_reports()
    index, texts = build_index(rows, client)

    # Embed the user query
    query_emb = client.embeddings.create(model="llama-3.1-8b-embedding", input=user_question).data[0].embedding
    query_emb = np.array([query_emb]).astype("float32")

    # Search top-k results
    D, I = index.search(query_emb, k=5)
    retrieved = [texts[i] for i in I[0]]

    # Pass to Groq for summarization
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "You are a medical assistant that answers questions based on blood test reports."},
            {"role": "user", "content": f"Question: {user_question}\n\nRelevant blood reports:\n" + "\n".join(retrieved)}
        ]
    )
    st.markdown("### 🧾 Answer")
    st.write(response.choices[0].message.content)
