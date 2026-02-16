# app.py
import os
from flask import Flask, render_template, request, jsonify
import pandas as pd
import openai
from openai import OpenAI
import duckdb
import sqlparse
from io import BytesIO
from uuid import uuid4
from flask import send_file
import json

# Load environment variables for OpenAI
from dotenv import load_dotenv
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY
client = OpenAI(api_key=OPENAI_API_KEY)

# -----------------------------
# Flask app
# -----------------------------
app = Flask(__name__)

QUERY_RESULTS = {}  # id -> pandas DataFrame
MAX_STORED_RESULTS = 2000

# -----------------------------
# Function: Load JSONL households
# -----------------------------


def load_households(path="data/households.jsonl"):
    households = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                households.append(json.loads(line))
    return households


# ---------- Load data from JSONL ----------
DATA_FILE = "data/households.jsonl"  # Change to your path


def load_data():
    """Read JSONL file into a Pandas DataFrame"""
    df = pd.read_json(DATA_FILE, lines=True)
    return df


df = load_data()


# Create an in-memory DuckDB connection and register the dataframe as a SQL table
con = duckdb.connect(database=":memory:")
con.register("households", df)  # SQL table name will be: households

# ---------- Prepare stats for charts ----------


def prepare_stats(df):
    stats = {}

    numeric_cols = [
        "عدد أجهزة التكييف",
        "عدد أفراد الأسرة",
        "عدد الإناث",
        "عدد الذكور",
        "عدد السيارات",
        "عدد الغرف",
        "عدد خزانات المياه الأرضية",
        "عدد خزانات المياه السطحية",
        "عدد كمبيوتر"
    ]

    # Numeric description

    total = len(df)

    # GroupBy "الحي"
    stats["per_neighborhood"] = df.groupby("الحي").size().to_dict()

    # Ownership
    stats["ملكية_السكن"] = df.groupby("ملكية السكن").size().to_dict()
    stats["ملكية_سيارة"] = df.groupby("ملكية سيارة").size().to_dict()

    stats["ملكية_كمبيوتر"] = df.groupby("ملكية كمبيوتر ").size().to_dict()

    # Services
    stats["خدمة_انترنت"] = df.groupby("خدمة انترنت").size().to_dict()
    stats["خدمة_تلفون_أرضي"] = df.groupby("خدمة تلفون أرضي").size().to_dict()
    stats["الصرف_الصحي"] = df.groupby("الصرف الصحي").size().to_dict()
    stats["جمع_النفايات"] = df.groupby(
        "هل تقوم البلدية بجمع النفايات؟").size().to_dict()
    stats["مصدر_المياه"] = df.groupby("مصدر المياه").size().to_dict()
    stats["مصدر_كهرباء"] = df.groupby("مصدر كهرباء").size().to_dict()
    stats["مصدر_مياه_اخر"] = df.groupby(
        "ما هو مصدر المياه الاخر").size().to_dict()
    stats["الدخل_الشهري"] = df.groupby("الدخل الشهري").size().to_dict()

    # Special categories
    stats["expatriates"] = df.groupby(
        "أحد أفراد الأسرة مغترب").size().to_dict()
    stats["special_needs"] = df.groupby(
        "وجود شخص من ذوي الإحتياجات الخاصة ").size().to_dict()

    return stats


def store_result_df(result_df: pd.DataFrame) -> str:
    # keep memory bounded
    if len(QUERY_RESULTS) >= MAX_STORED_RESULTS:
        QUERY_RESULTS.pop(next(iter(QUERY_RESULTS)))  # remove oldest (simple)

    result_id = str(uuid4())
    QUERY_RESULTS[result_id] = result_df
    return result_id


def df_schema_text(df: pd.DataFrame) -> str:
    """
    Build a compact schema description for the model.
    Note: your columns are Arabic and some contain spaces; SQL must quote them with double-quotes.
    """
    lines = []
    for col, dtype in df.dtypes.items():
        lines.append(f'- "{col}" ({str(dtype)})')
    return "\n".join(lines)


def generate_sql_from_question(question: str, df: pd.DataFrame) -> str:
    schema = df_schema_text(df)

    prompt = f"""
You convert user questions into SQL queries for DuckDB.

Rules:
- Output SQL ONLY.
- Must be a single SELECT query.
- Query the table named: households
- Do NOT use INSERT/UPDATE/DELETE/CREATE/DROP/ATTACH/COPY/PRAGMA.
- If you need a column with spaces or Arabic, wrap it in double quotes, e.g. "خدمة انترنت".
- Prefer COUNT(*), AVG(), SUM(), GROUP BY, ORDER BY.
- If returning many rows, add LIMIT 50.

Schema (table households):
{schema}

User question (Arabic possible):
{question}
""".strip()

    resp = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": "You are a strict SQL generator for DuckDB."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    sql = resp.choices[0].message.content.strip()
    # Sometimes the model wraps in ```sql ... ```
    sql = sql.replace("```sql", "").replace("```", "").strip()
    return sql


def is_safe_select(sql: str) -> bool:
    """
    Very conservative validation: only allow a single SELECT statement.
    """
    if not sql:
        return False

    # Disallow multiple statements
    statements = [s for s in sqlparse.split(sql) if s.strip()]
    if len(statements) != 1:
        return False

    s = statements[0].strip().lower()

    # Must start with select
    if not s.startswith("select"):
        return False

    # Block dangerous keywords
    blocked = [
        "insert", "update", "delete", "drop", "create", "alter", "attach",
        "copy", "pragma", "call", "export", "import"
    ]
    if any(b in s for b in blocked):
        return False

    # Optional: require it references the households table
    if " from households" not in s and "from households" not in s:
        return False

    return True


def run_sql_on_df(sql: str):
    """
    Execute SQL on DuckDB where df is registered as households.
    Returns a pandas dataframe.
    """
    return con.execute(sql).df()

# ---------- Routes ----------


@app.route("/download/<result_id>.xlsx", methods=["GET"])
def download_xlsx(result_id):
    if result_id not in QUERY_RESULTS:
        return jsonify({"error": "Result not found or expired"}), 404

    df_out = QUERY_RESULTS[result_id]

    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df_out.to_excel(writer, index=False, sheet_name="result")

    output.seek(0)

    return send_file(
        output,
        as_attachment=True,
        download_name=f"query_result_{result_id}.xlsx",
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )


@app.route("/")
def index():
    stats = prepare_stats(df)
    return render_template("index.html", stats=stats)


@app.route("/ask", methods=["POST"])
def ask():
    question = request.json.get("question", "").strip()
    if not question:
        return jsonify({"answer": "Please enter a question."}), 400

    try:
        sql = generate_sql_from_question(question, df)

        if not is_safe_select(sql):
            return jsonify({
                "answer": "I couldn't generate a safe SQL query for that question. Please rephrase.",
                "sql": sql
            }), 400

        result_df = run_sql_on_df(sql)

        download_url = None

        if not result_df.empty and not (result_df.shape == (1, 1)):
            result_id = store_result_df(result_df)
            download_url = f"/download/{result_id}.xlsx"

        # Format the result for display
        if result_df.empty:
            answer_text = "No results found."
        elif result_df.shape == (1, 1):
            # single scalar
            answer_text = str(result_df.iat[0, 0])
        else:
            # show a small table
            answer_text = result_df.head(5).to_markdown(index=False)

        return jsonify({
            "answer": answer_text,
            "sql": sql,                 # optional
            "download_url": download_url
        })

    except Exception as e:
        return jsonify({"answer": f"Error running query: {str(e)}"}), 500


# ---------- Run ----------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(debug=True, host="0.0.0.0", port=port)
