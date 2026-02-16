import streamlit as st
import mysql.connector
import tempfile

st.title("Blood Reports Database Manager")

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
    flag = st.text_input("Flag")
    submitted = st.form_submit_button("Insert")
    if submitted:
        run_query(
            "INSERT INTO blood_reports (name, test_name, result, unit, ref_range, flag) VALUES (%s,%s,%s,%s,%s,%s)",
            (name, test_name, result, unit, ref_range, flag)
        )
        st.success("✅ Record inserted successfully!")

# --- Search Records ---
st.header("🔍 Search Records")
search_test = st.text_input("Search by test name")
if search_test:
    rows = run_query("SELECT * FROM blood_reports WHERE test_name LIKE %s", (f"%{search_test}%",), fetch=True)
    st.write(rows)

# --- Edit Record ---
st.header("✏️ Edit Record")
edit_id = st.number_input("Enter ID to edit", min_value=1, step=1)
if edit_id:
    rows = run_query("SELECT * FROM blood_reports WHERE id=%s", (edit_id,), fetch=True)
    if rows:
        row = rows[0]
        with st.form("edit_form"):
            new_result = st.number_input("New Result", value=row["result"], step=0.01)
            new_flag = st.text_input("New Flag", value=row["flag"])
            update = st.form_submit_button("Update")
            if update:
                run_query("UPDATE blood_reports SET result=%s, flag=%s WHERE id=%s", (new_result, new_flag, edit_id))
                st.success("✅ Record updated successfully!")

# --- Delete Record ---
st.header("🗑️ Delete Record")
delete_id = st.number_input("Enter ID to delete", min_value=1, step=1, key="delete")
if st.button("Delete"):
    run_query("DELETE FROM blood_reports WHERE id=%s", (delete_id,))
    st.success("✅ Record deleted successfully!")
