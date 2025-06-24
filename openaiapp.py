import os
import json
import datetime
import pandas as pd
import duckdb
import openai
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import traceback

# Load API key from .env file
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Flask app setup
app = Flask(__name__)
CORS(app)

print("✅ Fraud detection service starting...")

# Load Excel
try:
    df = pd.read_excel("data.xlsx", sheet_name=0)
    if df.empty or df.shape[1] == 0:
        raise ValueError("Excel file is empty or missing columns.")
    print("✅ Excel file loaded successfully.")
    print(f"[DEBUG] Columns: {df.columns.tolist()}")
except Exception as e:
    print(f"[ERROR] Failed to load Excel file: {e}")
    traceback.print_exc()
    df = pd.DataFrame()

# Generate table metadata for OpenAI prompt
table_metadata = "TABLE: fraud_data\n" + "\n".join(
    [f"{col}: {str(dtype)}" for col, dtype in df.dtypes.items()]
)

# Utility: Send prompt to OpenAI
def send_to_openai(prompt, model="gpt-4", temperature=0):
    try:
        print(f"[DEBUG] Prompt to OpenAI:\n{prompt[:400]}...\n")
        response = openai.ChatCompletion.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
        reply = response.choices[0].message.content.strip()
        print(f"[DEBUG] OpenAI reply:\n{reply[:300]}...\n")
        return reply
    except Exception as e:
        print(f"[ERROR] OpenAI error: {e}")
        traceback.print_exc()
        return None

# Classify query type
def classify_query(user_prompt):
    prompt = f"""
You are a fraud data assistant. Classify the following user request as either:
- "SQL": if they want structured data from a table
- "GENERAL": if they want an explanation or insight.

Prompt:
{user_prompt}

Respond with ONLY: SQL or GENERAL.
"""
    result = send_to_openai(prompt)
    return result.strip().upper() if result else "GENERAL"

# Handle general questions
def answer_general(question, chat_summary=""):
    context = f"\nContext:\n{chat_summary}" if chat_summary else ""
    prompt = f"""
You are a helpful assistant for fraud analysts.{context}

Provide a direct and concise answer to the following question:

{question}
"""
    return send_to_openai(prompt).strip()


# Generate SQL query using metadata
def generate_duckdb_sql_query(user_prompt, table_metadata, chat_summary=""):
    current_date = datetime.date.today().isoformat()
    context = f"\nContext:\n{chat_summary}" if chat_summary else ""

    prompt = f"""
Generate a SQL query compatible with DuckDB to answer the user's request below.

Use only column names from the table schema.
Use functions supported by DuckDB.
Use this date for TODAY: {current_date}

PROMPT:
{user_prompt}

{context}

TABLE SCHEMA:
{table_metadata}

Respond ONLY with the SQL query, no explanation.
"""
    return send_to_openai(prompt)

# Natural language summary
def summarize_result_naturally(sql_query, rows, columns):
    preview = [dict(zip(columns, row)) for row in rows[:5]]
    prompt = f"""
You are a data analyst assistant.

SQL Query:
{sql_query}

Sample Rows:
{json.dumps(preview, indent=2)}
Total rows: {len(rows)}

Summarize this result in 2–3 sentences for fraud analysis.
"""
    return send_to_openai(prompt)

# Generate conversation title
def get_title(summary, user_prompt=""):
    prompt = f"""
Generate a short (3–5 word) title for the user query below.

User prompt: {user_prompt}
Summary: {summary}

Return only the title.
"""
    title = send_to_openai(prompt)
    return title.strip().strip('"\'') if title else "New Conversation"

# Chat summarizer
def summarize_chat_history(chat_history):
    try:
        context = json.dumps(chat_history[-3:], indent=2)
        prompt = f"""
Summarize the following recent chat context for better understanding:
{context}
"""
        return send_to_openai(prompt)
    except Exception as e:
        print(f"[ERROR] Failed to summarize chat: {e}")
        traceback.print_exc()
        return ""

# === Main endpoint ===
@app.route("/query", methods=["POST"])
def handle_query():
    data = request.get_json()
    print(f"[DEBUG] Request data:\n{json.dumps(data, indent=2)}")

    user_prompt = data.get("query", "")
    isNewChat = data.get("isNewChat", False)
    chat_history = data.get("chatHistory", [])

    try:
        if df.empty:
            return jsonify({"error": "Excel data not loaded."}), 500

        chat_summary = summarize_chat_history(chat_history) if chat_history and not isNewChat else ""

        query_type = classify_query(user_prompt)
        if query_type == "GENERAL":
            response = answer_general(user_prompt, chat_summary)
            title = get_title(response, user_prompt) if isNewChat else ""
            return jsonify({
                "summary": response,
                "title": title,
                "is_sql_query": False,
                "chat_summary": chat_summary
            })

        # SQL path
        generated_sql = generate_duckdb_sql_query(user_prompt, table_metadata, chat_summary)
        print(f"[DEBUG] SQL generated:\n{generated_sql}")

        if not generated_sql:
            return jsonify({"error": "Failed to generate SQL"}), 500

        # Execute SQL using DuckDB
        try:
            con = duckdb.connect()
            con.register("fraud_data", df)
            result_df = con.execute(generated_sql).fetchdf()
            result_json = result_df.to_dict(orient="records")
            rows = result_df.values.tolist()
            columns = list(result_df.columns)
            print(f"[DEBUG] SQL executed. Rows returned: {len(result_json)}")
        except Exception as sql_error:
            print(f"[ERROR] SQL execution failed: {sql_error}")
            traceback.print_exc()
            return jsonify({
                "error": str(sql_error),
                "generated_sql": generated_sql
            }), 500

        summary = summarize_result_naturally(generated_sql, rows, columns)
        title = get_title(summary, user_prompt) if isNewChat else ""

        return jsonify({
            "sql": generated_sql,
            "result": result_json,
            "summary": summary,
            "title": title,
            "is_sql_query": True,
            "chat_summary": chat_summary
        })

    except Exception as e:
        print(f"[ERROR] Unexpected error: {str(e)}")
        traceback.print_exc()
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500

# Run the app
if __name__ == "__main__":
    app.run(debug=True)
