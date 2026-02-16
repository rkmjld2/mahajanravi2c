# TiDB Cloud Connection Test

This is a minimal Streamlit app to verify connectivity to TiDB Cloud.

## Files
- `app.py` → Streamlit app that connects to TiDB Cloud and runs a simple query.
- `.streamlit/secrets.toml` → Stores database credentials and SSL certificate.
- `requirements.txt` → Dependencies for Streamlit Cloud.

## Usage
1. Add your TiDB Cloud credentials and CA certificate in `.streamlit/secrets.toml`.
2. Run locally:
   ```bash
   streamlit run app.py
