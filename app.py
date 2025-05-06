from flask import Flask, request, jsonify
from flask_cors import CORS
import datetime
import os
from dotenv import load_dotenv
from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
import prestodb
from prestodb.auth import BasicAuthentication
import json

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

wxd_cred = {
    "password": os.getenv("presto_key"),
    "engine_host": os.getenv("presto_host"),
    "ssl": """true""",
    "engine_port": os.getenv("presto_port"),
    "username": os.getenv("presto_user"),
}

# Connect to Watsonx.data via Presto
conn = prestodb.dbapi.connect(
    host=wxd_cred["engine_host"],
    port=int(wxd_cred["engine_port"]),
    user=wxd_cred["username"],
    http_scheme="https",
    auth=BasicAuthentication(wxd_cred["username"], wxd_cred["password"]),
)
cur = conn.cursor()


# ---------------- IBMWatsonx Class ----------------


class IBMWatsonx:
    def __init__(
        self,
        model_name="meta-llama/llama-3-1-70b-instruct",
        decoding_method="greedy",
        max_new_tokens=300,
        min_new_tokens=1,
        random_seed=42,
        repetition_penalty=1.0,
    ):

        self.model = self.load_model(
            model_name,
            decoding_method,
            max_new_tokens,
            min_new_tokens,
            random_seed,
            repetition_penalty,
        )

    def get_watson_creds(self):
        """Load Watsonx credentials from .env"""
        api_key = os.getenv("API_KEY")
        ibm_cloud_url = os.getenv("IBM_CLOUD_URL")
        project_id = os.getenv("PROJECT_ID")

        if not all([api_key, ibm_cloud_url, project_id]):
            raise Exception("Missing WatsonX credentials. Check your .env file.")

        creds = {"url": ibm_cloud_url, "apikey": api_key, "project_id": project_id}
        return creds

    def load_model(
        self,
        model_name,
        decoding_method,
        max_new_tokens,
        min_new_tokens,
        random_seed,
        repetition_penalty,
    ):
        """Load WatsonX model."""
        creds = self.get_watson_creds()

        model_params = {
            GenParams.DECODING_METHOD: decoding_method,
            GenParams.MIN_NEW_TOKENS: min_new_tokens,
            GenParams.MAX_NEW_TOKENS: max_new_tokens,
            GenParams.RANDOM_SEED: random_seed,
            GenParams.REPETITION_PENALTY: repetition_penalty,
        }

        model = Model(
            model_id=model_name,
            params=model_params,
            credentials=creds,
            project_id=creds["project_id"],
        )

        return model

    def get_model(self):
        return self.model

    def send_to_watsonxai(self, prompt):
        response = self.model.generate(prompt=prompt)
        return response["results"][0]["generated_text"].strip()


# --------------------------------------------------

# Initialize WatsonX model once when the server starts
watsonx_client = IBMWatsonx()


def get_table_metadata_from_presto(schema_name, table_name, database="hive_data"):
    """Fetch dynamic schema from Presto's information_schema."""
    query = f"""
    SELECT column_name
    FROM {database}.information_schema.columns
    WHERE table_schema = '{schema_name}' AND table_name = '{table_name}'
    ORDER BY ordinal_position
    """
 
    cur.execute(query)
    columns = [row[0] for row in cur.fetchall()]
 
    if not columns:
        raise Exception(f"No columns found for {schema_name}.{table_name}")
 
    metadata = (
        f'"{database}"."{schema_name}"."{table_name}" (' + ", ".join(columns) + ")"
    )
    return metadata


def classify_query(user_prompt):
    """
    Classify the user prompt as:
    - "SQL": Needs specific SQL generation (like "show me transactions from last week")
    - "GENERAL": Needs general fraud data information (like "what patterns indicate fraud?")
    """
    prompt = f"""
You are an assistant specializing in fraud data analysis. Given the user prompt below, determine whether the user is asking for:

1. A specific SQL query to retrieve or analyze fraud data (like "show me all fraud transactions from yesterday" or "get data for transactions over $10,000") - respond with "SQL" only
2. A general question about fraud data, patterns, or insights (like "what are common fraud indicators?" or "explain the trends in our fraud data") - respond with "GENERAL" only

User Prompt:
{user_prompt}

Reply ONLY with either "SQL" or "GENERAL".
"""
    try:
        result = watsonx_client.send_to_watsonxai(prompt).strip().upper()
        return "SQL" if result == "SQL" else "GENERAL"
    except Exception as e:
        print(f"Failed to classify prompt: {e}")
        return "GENERAL"  # Default to general case if classification fails


def answerGeneral(question, chat_summary=""):
    """Handle general questions with optional conversation context."""
    context = ""
    if chat_summary:
        context = f"""
CONVERSATION CONTEXT:
{chat_summary}

This context may help you understand the user's question better. Consider this history when providing your answer.
"""
    
    prompt = f"""
You are a fraud analysis expert with deep knowledge of financial fraud patterns, detection methods, and trends.
Answer the following question about fraud data with specific, actionable insights.
Focus on being helpful for fraud analysts by providing:
- Clear explanations of fraud patterns and indicators
- Context about how certain data points relate to potential fraud
- Best practices for fraud detection and prevention
- Interpretation of fraud-related metrics and trends

{context}

Question:
{question}

Provide a concise but informative answer that would help a fraud analyst understand the issue better.
"""
    try:
        result = watsonx_client.send_to_watsonxai(prompt)
        return result
    except Exception as e:
        print(f"Failed to generate fraud analysis response: {e}")
        return "Unable to process your question about fraud data. Please try rephrasing your query."


def generate_hive_sql_query(user_prompt, table_metadata, chat_summary=""):
    """Use Watsonx.ai to generate Hive SQL from a natural language prompt with optional context."""
    # Add context if available
    context = ""
    if chat_summary:
        context = f"""
CONVERSATION CONTEXT:
{chat_summary}

This context may help you understand the user's query intent. Consider this history when generating SQL.
"""
    
    full_prompt = f"""
You are an expert Hive SQL Developer.

Your task:
- Read the USER PROMPT.
- Using the TABLE METADATA{" and CONVERSATION CONTEXT" if chat_summary else ""}, generate a valid Hive SQL query.
- Output ONLY the SQL query (no explanation, no extra text).

Rules:
- Use strict Hive SQL syntax.
- Use single quotes for string literals.
- Do not use semicolons at the end.
- Avoid complex subqueries unless explicitly requested.
- Do not use Oracle-specific functions or keywords.
- Ensure the query is properly formatted and can be executed directly.

{context if chat_summary else ""}

---

USER PROMPT:
{user_prompt}

---

TABLE METADATA:
{table_metadata}

---
"""
    try:
        sql_query = watsonx_client.send_to_watsonxai(full_prompt)
        # Clean the SQL query (remove any markdown formatting the LLM might add)
        sql_query = sql_query.strip()
        if sql_query.startswith("```sql"):
            sql_query = sql_query[6:]
        if sql_query.startswith("```"):
            sql_query = sql_query[3:]
        if sql_query.endswith("```"):
            sql_query = sql_query[:-3]
        
        # Final clean
        sql_query = sql_query.strip()
        
        print("Generated SQL:", sql_query)
        return sql_query
    except Exception as e:
        print(f"Error generating SQL with Watsonx: {e}")
        return None


def summarize_result_naturally(sql_query, rows, columns):
    """Ask WatsonX to summarize SQL results into a human-readable format."""
    preview_data = [dict(zip(columns, row)) for row in rows[:5]]

    prompt = f"""
You are a data analyst assistant. You are provided with a SQL query and sample data from the query result. Your job is to summarize the result in a user-friendly way.

SQL Query:
{sql_query}

Sample Result Data (only first 5 rows shown as JSON):
{preview_data}

Please summarize what this data tells in one or two clear sentences.
"""

    try:
        natural_summary = watsonx_client.send_to_watsonxai(prompt)
        return natural_summary
    except Exception as e:
        print(f"Error generating natural language summary: {e}")
        return "Unable to summarize the results."


def get_title(summary, user_prompt=""):
    """
    Generate a concise, relevant title based on the context.
    Takes both the summary and original user prompt into account.
    """
    # For very short messages like greetings, use a simple greeting title
    if len(user_prompt.strip()) < 10 and any(greeting in user_prompt.lower() for greeting in ["hello", "hi", "hey", "greetings"]):
        return "New Conversation"
    
    # For SQL queries about transactions, use a more direct title
    if "transaction" in user_prompt.lower() and "sql" in summary.lower():
        return "Recent Transactions"
    
    # For more substantial content, ask the LLM to generate a title
    prompt = f"""
Your task is to generate a concise title (3-5 words) for a conversation.

User query: {user_prompt}
Response: {summary}

The title should directly reflect what the user asked for.
Return ONLY the title text, with NO additional explanations or instructions.
"""
    try:
        response = watsonx_client.send_to_watsonxai(prompt)
        # Clean up the title
        title = response.strip().strip('"\'').strip()
        
        # If the title is still too long or contains explanations, extract just the first line
        if "\n" in title:
            title = title.split("\n")[0].strip()
        
        # Remove common prefixes
        for prefix in ["Title:", "Title", "#"]:
            if title.startswith(prefix):
                title = title[len(prefix):].strip()
        
        # Additional safety check - if the title is too long or contains instructions
        if len(title) > 50 or "not" in title.lower() or "do not" in title.lower() or "unless" in title.lower():
            # Fall back to a simple title based on the query
            if "transaction" in user_prompt.lower():
                return "Recent Transactions"
            elif "sales" in user_prompt.lower():
                return "Sales Data"
            elif "fraud" in user_prompt.lower():
                return "Fraud Analysis"
            else:
                return "Data Query Results"
        
        return title if title else "New Conversation"
    except Exception as e:
        print(f"[ERROR] Failed to generate title: {e}")
        return "New Conversation"

def convert_value(val):
    """Convert non-serializable types like datetime to JSON-safe types."""
    if isinstance(val, datetime.datetime):
        return val.isoformat()
    return val

def handle_title(natural_summary):
    data = request.get_json()
    new_chat = data.get("is_new_chat", "")
    if new_chat:
        title = get_title(natural_summary)
        return title
    else:
        return ""

def summarize_chat_history(chat_history):
    """
    Summarize the chat history with emphasis on the last two exchanges.
    Only extracts the text content from user and bot messages.
    
    Args:
        chat_history (list): The list of message objects from the frontend
        
    Returns:
        str: A summarized version of the conversation
    """
    if not chat_history:
        return ""
        
    # Extract just the message text and sender info from the chat history
    # The frontend is already sending simplified history with just text and sender
    simplified_history = chat_history
    
    # Identify the last two user questions and their responses
    recent_exchanges = []
    user_questions_found = 0
    
    # Start from the end of the history and work backwards
    for i in range(len(simplified_history) - 1, -1, -1):
        msg = simplified_history[i]
        if msg["sender"] == "user" and user_questions_found < 2:
            # Found a user question
            user_questions_found += 1
            recent_exchanges.insert(0, msg)
            
            # Look for the associated bot response (should be after this message)
            if i + 1 < len(simplified_history) and simplified_history[i + 1]["sender"] == "bot":
                recent_exchanges.insert(1, simplified_history[i + 1])
    
    # Prepare the prompt for WatsonX
    prompt = f"""
You are a specialized AI assistant for fraud analysis and data querying. Your task is to summarize a conversation while emphasizing the most recent exchanges.

INSTRUCTIONS:
1. Summarize the entire conversation in 4-5 sentences, capturing the main topics and context.
2. Then, provide special emphasis on the last two questions and answers.
3. Be concise but accurate, especially for any SQL queries or fraud analysis terms.
4. Do not include timestamps, IDs, or any metadata in your summary.
5. Format your response as a clear, cohesive paragraph.

CHAT HISTORY:
{json.dumps(simplified_history, indent=2)}

RECENT EXCHANGES TO EMPHASIZE:
{json.dumps(recent_exchanges, indent=2)}

Provide your summary below:
"""
    
    try:
        # Send to WatsonX.AI for summarization
        summary = watsonx_client.send_to_watsonxai(prompt)
        return summary
    except Exception as e:
        print(f"Error summarizing chat history: {e}")
        return "Unable to generate conversation summary."


@app.route("/query", methods=["POST"])
def handle_query():
    data = request.get_json()
    user_prompt = data.get("query", "")
    isNewChat = data.get("isNewChat")
    chat_history = data.get("chatHistory", [])  # Will be empty for new chats
    
    # These could be dynamically passed by the user in a production system
    schema = "ejada_somni"
    database = "hive_data"
    tables = ["account", "direct", "client", "frauddata"]
    
    title = ""
    is_sql_query = False
    
    # Only process chat history if this is not a new chat and we have history
    chat_summary = ""
    if not isNewChat and chat_history and len(chat_history) > 2:
        chat_summary = summarize_chat_history(chat_history)
        print("Chat summary:", chat_summary)

    try:
        # First, determine if we need SQL or a general fraud analysis response
        query_type = classify_query(user_prompt)
        
        # Handle general fraud analysis questions
        if query_type == "GENERAL":
            # Only include chat summary if we have it
            if chat_summary:
                general_response = answerGeneral(user_prompt, chat_summary)
            else:
                general_response = answerGeneral(user_prompt)
            
            # Generate title for new chats
            if isNewChat == "True" or isNewChat is True:
                title = get_title(general_response, user_prompt)
                
            return jsonify({
                "summary": general_response,
                "title": title,
                "is_sql_query": False,
                "chat_summary": chat_summary  # Will be empty for new chats
            })

        # If we reach here, it's an SQL query
        is_sql_query = True
        
        # Get metadata from Presto directly
        combined_schema_parts = []
        for table in tables:
            try:
                metadata = get_table_metadata_from_presto(schema, table, database)
                combined_schema_parts.append(metadata)
            except Exception as e:
                print(f"Error retrieving metadata for {table}: {e}")
        
        # Final combined schema as one string
        combined_schema = "1. " + "\n2. ".join(combined_schema_parts)
        
        # Generate SQL from natural language prompt
        if chat_summary:
            generated_sql = generate_hive_sql_query(user_prompt, combined_schema, chat_summary)
        else:
            generated_sql = generate_hive_sql_query(user_prompt, combined_schema)
            
        if not generated_sql:
            return jsonify({"error": "Failed to generate SQL", "is_sql_query": True}), 500

        # Execute the SQL with better error handling
        try:
            print("Executing SQL:", generated_sql)
            cur.execute(generated_sql)
            print("SQL executed successfully")
        except Exception as sql_error:
            print(f"Error executing SQL: {sql_error}")
            return jsonify({
                "error": f"SQL execution error: {str(sql_error)}",
                "generated_sql": generated_sql,
                "is_sql_query": True
            }), 500

        if cur.description:
            columns = [col[0] for col in cur.description]
            rows = cur.fetchall()
            result = [
                dict(zip(columns, [convert_value(v) for v in row])) for row in rows
            ]
            natural_summary = summarize_result_naturally(generated_sql, rows, columns)
            
            # If this is a new chat, generate a better title based on the results
            if isNewChat == "True" or isNewChat is True:
                title = get_title(natural_summary, user_prompt)
        else:
            result = {"message": f"{cur.rowcount} rows affected"}
            natural_summary = result["message"]
            
            # Generate title for new chats when there are no results
            if (isNewChat == "True" or isNewChat is True):
                title = get_title(f"SQL Query: {generated_sql}", user_prompt)
            
        return jsonify({
            "sql": generated_sql,
            "result": result,
            "summary": natural_summary,
            "title": title,
            "is_sql_query": is_sql_query,
            "chat_summary": chat_summary
        })
        
    except Exception as e:
        return jsonify({
            "error": str(e),
        }), 500


if __name__ == "__main__":
    app.run(debug=True)