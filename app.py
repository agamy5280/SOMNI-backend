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

print("Starting fraud detection service")

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
    SELECT column_name, data_type
    FROM {database}.information_schema.columns
    WHERE table_schema = '{schema_name}' AND table_name = '{table_name}'
    ORDER BY ordinal_position
    """
 
    cur.execute(query)
    columns = cur.fetchall()
    
    if not columns:
        raise Exception(f"No columns found for {schema_name}.{table_name}")
    
    # Create more detailed metadata with data types
    column_details = []
    for col_name, data_type in columns:
        column_details.append(f"{col_name} ({data_type})")
    
    metadata = f'"{database}"."{schema_name}"."{table_name}" (' + ", ".join(column_details) + ")"
    return metadata, columns


def get_detailed_table_info(schema_name, table_name, database="hive_data"):
    """Get sample data, date columns, and unique column values for a table."""
    # First get columns and their types
    _, columns = get_table_metadata_from_presto(schema_name, table_name, database)
    
    # Identify date/timestamp columns
    date_columns = [col_name for col_name, data_type in columns 
                   if 'date' in data_type.lower() or 'time' in data_type.lower()]
    
    # Get sample data (top 5 rows)
    try:
        sample_query = f"""
        SELECT * FROM {database}.{schema_name}.{table_name} 
        LIMIT 5
        """
        cur.execute(sample_query)
        sample_data = []
        if cur.description:
            sample_columns = [col[0] for col in cur.description]
            sample_rows = cur.fetchall()
            for row in sample_rows:
                sample_data.append(dict(zip(sample_columns, [convert_value(v) for v in row])))
    except Exception as e:
        print(f"Error fetching sample data: {e}")
        sample_data = []
    
    # Get unique values for important columns (limited to avoid performance issues)
    unique_values = {}
    important_columns = []
    
    # Identify important columns based on their names
    for col_name, data_type in columns:
        col_lower = col_name.lower()
        if (any(term in col_lower for term in ['type', 'category', 'status', 'class', 'key', 'id', 'flag'])):
            important_columns.append(col_name)
    
    # Get unique values for important columns (with a reasonable limit)
    for col_name in important_columns[:5]:  # Limit to 5 columns to prevent performance issues
        try:
            distinct_query = f"""
            SELECT DISTINCT {col_name} 
            FROM {database}.{schema_name}.{table_name}
            LIMIT 20
            """
            cur.execute(distinct_query)
            values = [row[0] for row in cur.fetchall() if row[0] is not None]
            if values:
                unique_values[col_name] = values
        except Exception as e:
            print(f"Error fetching unique values for {col_name}: {e}")
    
    # Get current date from database to ensure consistency
    current_date = None
    try:
        cur.execute("SELECT CURRENT_DATE")
        current_date = cur.fetchone()[0]
        if isinstance(current_date, datetime.date):
            current_date = current_date.isoformat()
    except Exception as e:
        print(f"Error fetching current date: {e}")
        # Fallback to system date
        current_date = datetime.date.today().isoformat()
    
    return {
        "date_columns": date_columns,
        "sample_data": sample_data,
        "unique_values": unique_values,
        "current_date": current_date
    }


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


import datetime
import json

def generate_hive_sql_query(user_prompt, table_metadata, detailed_info, chat_summary="", sql_engine="Presto"):
    """Use Watsonx.ai to generate SQL from a natural language prompt with optional context.
    Adjusts SQL generation rules for either Hive or Presto syntax.
    """

    # Add context if available
    context = ""
    if chat_summary:
        context = f"""
CONVERSATION CONTEXT:
{chat_summary}

This context may help you understand the user's query intent. Consider this history when generating SQL.
"""

    # Format the detailed info for the prompt
    date_columns_info = ""
    for table, info in detailed_info.items():
        if info["date_columns"]:
            date_columns_info += f"\n{table} date columns: {', '.join(info['date_columns'])}"

    # Get current date from metadata, or fallback to today's date
    current_date = None
    for table, info in detailed_info.items():
        if "current_date" in info and info["current_date"]:
            current_date = info["current_date"]
            break
    if not current_date:
        current_date = datetime.date.today().isoformat()

    # Unique values for lookup columns
    unique_values_info = ""
    for table, info in detailed_info.items():
        if "unique_values" in info and info["unique_values"]:
            unique_values_info += f"\n{table} key values:\n"
            for col, values in info["unique_values"].items():
                unique_values_info += f"  - {col}: {', '.join(str(v) for v in values[:10])}"
                if len(values) > 10:
                    unique_values_info += f" (and {len(values) - 10} more)"
                unique_values_info += "\n"

    # Sample data preview
    sample_data_info = ""
    for table, info in detailed_info.items():
        if info["sample_data"]:
            sample_data_info += f"\n{table} sample data (limited preview):\n"
            sample_data_info += json.dumps(info["sample_data"][:2], indent=2)

    # Date subtraction syntax based on SQL engine
    if sql_engine.lower() == "presto":
        date_sub_instruction = (
            f"- Use DATE '{current_date}' - INTERVAL 'N' DAY for subtracting N days"
        )
    else:
        date_sub_instruction = (
            f"- Use DATE_SUB(date '{current_date}', N) for N days ago"
        )

    # Prompt to send to Watsonx
    full_prompt = f"""
You are an expert SQL Developer specializing in fraud analysis queries.

Your task:
- Read the USER PROMPT carefully.
- Match the user's intent to the most appropriate table and columns using the metadata provided.
- Generate a valid {sql_engine} SQL query that answers their question precisely.
- Output ONLY the SQL query (no explanation, no extra text).

IMPORTANT INFORMATION:
- Today's date (from database): {current_date}
- When the user mentions "recent", "last X days", or any date-based filters, use this exact date.

Rules:
- Use strict {sql_engine} SQL syntax.
- Always use fully qualified table names in the format database.schema_name.table_name.
- Use single quotes for string literals.
- Do not use semicolons at the end.
- For date/time operations:
  {date_sub_instruction}
- Use proper date formatting: YYYY-MM-DD
- if i asked about this month only give me the data in this month not more.
- Match user requests to the most appropriate table and columns.
- Prefer exact column matches over fuzzy matches.
- Use the unique values listed to help match user requests to appropriate columns.
- Ensure the query is properly formatted and can be executed directly.

{context if chat_summary else ""}

---

USER PROMPT:
{user_prompt}

---

TABLE METADATA:
{table_metadata}

---

DATE COLUMNS INFO:
{date_columns_info}

---

KEY COLUMNS AND VALUES:
{unique_values_info}

---

SAMPLE DATA:
{sample_data_info}

---

Remember:
- For time-based queries, use the exact date provided: {current_date}
- Match user requests to specific columns using the KEY COLUMNS information
- Always use the correct date operation syntax for {sql_engine}
- Choose the most appropriate table based on the user's query intent
- For fraud-specific queries, prioritize using the fraud table

Generate a valid {sql_engine} SQL query:
"""

    try:
        sql_query = watsonx_client.send_to_watsonxai(full_prompt)
        sql_query = sql_query.strip()
        if sql_query.startswith("```sql"):
            sql_query = sql_query[6:]
        if sql_query.startswith("```"):
            sql_query = sql_query[3:]
        if sql_query.endswith("```"):
            sql_query = sql_query[:-3]
        return sql_query.strip()
    except Exception as e:
        print(f"Error generating SQL with Watsonx: {e}")
        return None



def summarize_result_naturally(sql_query, rows, columns):
    """Ask WatsonX to summarize SQL results into a human-readable format."""
    preview_data = [dict(zip(columns, row)) for row in rows[:5]]
    
    total_rows = len(rows)
    row_count_info = f"The query returned {total_rows} rows in total."

    prompt = f"""
You are a data analyst assistant. You are provided with a SQL query and sample data from the query result. Your job is to summarize the result in a user-friendly way.

SQL Query:
{sql_query}

Sample Result Data (first 5 rows shown as JSON):
{preview_data}

Additional Info:
{row_count_info}

Please summarize what this data indicates in 2-3 clear sentences. Focus on what a fraud analyst would find most relevant. Make sure your summary is insightful and actionable.
,If there is any Currency it is SAR not $. 
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
    tables = ["account", "direct", "client", "fraud"]
    
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
        
        # Process the user query to identify potential entities, columns, or specific data points
        query_analysis_prompt = f"""
You are a data analysis expert. Analyze this query to identify:
1. Specific entities/categories mentioned (e.g., "fraud type", "transaction category")
2. Time periods mentioned (e.g., "last 30 days", "this month")
3. Key metrics requested (e.g., "count", "average", "total")

Query: {user_prompt}

Format your response as JSON with these keys: "entities", "time_period", "metrics"
Only include keys that are relevant. Be concise and extract ONLY what is explicitly mentioned.
"""
        try:
            query_analysis = watsonx_client.send_to_watsonxai(query_analysis_prompt)
            # Try to parse as JSON, but don't fail if it's not valid JSON
            try:
                query_analysis_json = json.loads(query_analysis)
                print("Query analysis:", query_analysis_json)
            except:
                print("Could not parse query analysis as JSON")
        except Exception as e:
            print(f"Query analysis failed: {e}")
        
        # Get metadata from Presto directly
        combined_schema_parts = []
        detailed_info = {}
        # Cache to store column mapping information for faster lookups
        column_to_table_map = {}
        
        for table in tables:
            try:
                metadata, columns = get_table_metadata_from_presto(schema, table, database)
                combined_schema_parts.append(metadata)
                
                # Create a mapping of column names to tables for quick reference
                for col_name, _ in columns:
                    col_lower = col_name.lower()
                    if col_lower not in column_to_table_map:
                        column_to_table_map[col_lower] = []
                    column_to_table_map[col_lower].append(table)
                
                # Get detailed info for each table including date columns and sample data
                detailed_info[table] = get_detailed_table_info(schema, table, database)
            except Exception as e:
                print(f"Error retrieving metadata for {table}: {e}")
        
        # Create table recommendations based on query analysis and column mapping
        table_recommendations = []
        
        # Get all important words from the query by tokenizing and removing stopwords
        query_words = user_prompt.lower().split()
        stopwords = ["the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "with", "by", "about", "as"]
        query_words = [word for word in query_words if word not in stopwords]
        
        # For each table, check how many query words match its columns
        table_match_scores = {table: 0 for table in tables}
        
        # If we have query_analysis_json, use it for better matching
        try:
            if 'entities' in query_analysis_json:
                for entity in query_analysis_json['entities']:
                    for word in entity.lower().split():
                        for col, tables in column_to_table_map.items():
                            if word in col:
                                for t in tables:
                                    table_match_scores[t] += 3  # Higher score for entity matches
        except:
            pass
        
        # Fall back to basic word matching
        for word in query_words:
            for col, tables in column_to_table_map.items():
                if word in col:
                    for t in tables:
                        table_match_scores[t] += 1
        
        # Special handling for fraud-related queries
        if "fraud" in user_prompt.lower():
            table_match_scores["fraud"] += 5
        
        # Sort tables by match score
        sorted_tables = sorted(table_match_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Add table recommendations based on scores
        for table, score in sorted_tables:
            if score > 0:
                table_recommendations.append(f"{table} (relevance score: {score})")
        
        # Include table recommendations in metadata
        if table_recommendations:
            combined_schema_parts.append("\nRECOMMENDED TABLES (in order of relevance):\n- " + 
                                        "\n- ".join(table_recommendations))
                                        
        # Final combined schema as one string
        combined_schema = "1. " + "\n2. ".join(combined_schema_parts)
        
        # Generate SQL from natural language prompt with enhanced metadata
        if chat_summary:
            generated_sql = generate_hive_sql_query(user_prompt, combined_schema, detailed_info, chat_summary)
        else:
            generated_sql = generate_hive_sql_query(user_prompt, combined_schema, detailed_info)
            
        if not generated_sql:
            return jsonify({"error": "Failed to generate SQL", "is_sql_query": True}), 500

        # Execute the SQL with better error handling
        try:
            print("Executing SQL:", generated_sql)
            cur.execute(generated_sql)
            print("SQL executed successfully")
        except Exception as sql_error:
            print(f"Error executing SQL: {sql_error}")
            
            # Try to fix common SQL errors automatically
            fix_sql_prompt = f"""
You are an expert SQL developer. A Hive SQL query failed with the following error:
{str(sql_error)}

The query that failed was:
{generated_sql}

Provide a corrected version of the SQL query that would fix this error.
Focus specifically on:
1. Using the correct column names (check for typos)
2. Using the correct date format and functions
3. Fixing syntax errors
4. Using the correct table name

Return ONLY the fixed SQL query with no explanations:
"""
            try:
                fixed_sql = watsonx_client.send_to_watsonxai(fix_sql_prompt).strip()
                # Try to execute the fixed SQL
                print("Attempting fixed SQL:", fixed_sql)
                try:
                    cur.execute(fixed_sql)
                    print("Fixed SQL executed successfully")
                    generated_sql = fixed_sql  # Use the fixed SQL going forward
                except Exception as fixed_error:
                    # If the fixed SQL also fails, create a user-friendly error message
                    error_explanation_prompt = f"""
You are a SQL debugging expert. An SQL query failed with the following error:
{str(sql_error)}

The query that failed was:
{generated_sql}

Explain in simple terms what went wrong and how to fix it. Focus on:
1. Potential syntax errors
2. Column name issues (especially date columns)
3. Table access problems
4. Data type mismatches

Be concise but helpful in your explanation:
"""
                    try:
                        error_explanation = watsonx_client.send_to_watsonxai(error_explanation_prompt)
                    except:
                        error_explanation = f"Error executing SQL: {str(sql_error)}"
                    
                    return jsonify({
                        "error": error_explanation,
                        "generated_sql": generated_sql,
                        "is_sql_query": True
                    }), 500
            except Exception as e:
                # If automatic fixing fails, return the original error
                error_explanation = f"Error executing SQL: {str(sql_error)}"
                return jsonify({
                    "error": error_explanation,
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
        print(f"Unexpected error: {str(e)}")
        return jsonify({
            "error": f"An unexpected error occurred: {str(e)}",
        }), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
