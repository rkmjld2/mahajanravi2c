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
