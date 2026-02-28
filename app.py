# app.py
import os
from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import openai
from openai import OpenAI
import duckdb
import sqlparse
from io import BytesIO
from uuid import uuid4
import json
from dotenv import load_dotenv

# -----------------------------
# Load environment variables
# -----------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY
client = OpenAI(api_key=OPENAI_API_KEY)

# -----------------------------
# Flask app
# -----------------------------
app = Flask(__name__)
QUERY_RESULTS = {}
MAX_STORED_RESULTS = 2000

# -----------------------------
# Load JSONL households
# -----------------------------
DATA_FILE = "data/households.jsonl"


def load_households(path=DATA_FILE):
    households = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                households.append(json.loads(line))
    return households

# -----------------------------
# Flatten JSON into multiple DataFrames
# -----------------------------


def load_data():
    households_json = load_households(DATA_FILE)

    df_households = pd.json_normalize(households_json)

    # Family members
    family_rows = []
    for hh in households_json:
        hh_id = hh["household_id"]
        # <-- convert None to empty list
        members = hh.get("family_members") or []
        for member in members:
            member_row = member.copy()
            member_row["household_id"] = hh_id
            family_rows.append(member_row)
    df_family = pd.DataFrame(family_rows)

    # Expatriates
    expat_rows = []
    for hh in households_json:
        hh_id = hh["household_id"]
        expats = hh.get("expatriates") or []  # <-- convert None to empty list
        for expat in expats:
            expat_row = expat.copy()
            expat_row["household_id"] = hh_id
            expat_rows.append(expat_row)
    df_expat = pd.DataFrame(expat_rows)

    return df_households, df_family, df_expat


df_households, df_family, df_expat = load_data()

# -----------------------------
# DuckDB in-memory connection
# -----------------------------
con = duckdb.connect(database=":memory:")
con.register("households", df_households)
con.register("family_members", df_family)
con.register("expatriates", df_expat)

# -----------------------------
# Prepare stats for charts
# -----------------------------


def prepare_stats(df):
    stats = {}
    stats["per_neighborhood"] = df.groupby("الحي").size().to_dict()
    stats["ملكية_السكن"] = df.groupby("ملكية السكن").size().to_dict()
    stats["ملكية_سيارة"] = df.groupby("ملكية سيارة").size().to_dict()
    stats["ملكية_كمبيوتر"] = df.groupby("ملكية كمبيوتر ").size().to_dict()
    stats["خدمة_انترنت"] = df.groupby("خدمة انترنت").size().to_dict()
    stats["خدمة_تلفون_أرضي"] = df.groupby("خدمة تلفون أرضي").size().to_dict()

    stats["الصرف_الصحي"] = (
        df.explode("الصرف الصحي")
        .dropna(subset=["الصرف الصحي"])
        .loc[lambda x: x["الصرف الصحي"] != ""]
        .groupby("الصرف الصحي")
        .size()
        .to_dict()
    )
    stats["مصدر_المياه"] = (
        df.explode("مصدر المياه")
        .dropna(subset=["مصدر المياه"])
        .loc[lambda x: x["مصدر المياه"] != ""]
        .groupby("مصدر المياه")
        .size()
        .to_dict()
    )
    stats["مصدر_كهرباء"] = (
        df.explode("مصدر كهرباء")
        .dropna(subset=["مصدر كهرباء"])
        .loc[lambda x: x["مصدر كهرباء"] != ""]
        .groupby("مصدر كهرباء")
        .size()
        .to_dict()
    )
    stats["مصدر_مياه_اخر"] = df.groupby(
        "ما هو مصدر المياه الاخر").size().to_dict()
    stats["الدخل_الشهري"] = df.groupby("الدخل الشهري").size().to_dict()
    stats["expatriates"] = df.groupby(
        "أحد أفراد الأسرة مغترب").size().to_dict()
    stats["special_needs"] = df.groupby(
        "وجود شخص من ذوي الإحتياجات الخاصة ").size().to_dict()
    return stats

# -----------------------------
# Helpers
# -----------------------------


def store_result_df(result_df: pd.DataFrame) -> str:
    if len(QUERY_RESULTS) >= MAX_STORED_RESULTS:
        QUERY_RESULTS.pop(next(iter(QUERY_RESULTS)))
    result_id = str(uuid4())
    QUERY_RESULTS[result_id] = result_df
    return result_id


def df_schema_text(df: pd.DataFrame) -> str:
    lines = []
    for col, dtype in df.dtypes.items():
        lines.append(f'- "{col}" ({str(dtype)})')
    return "\n".join(lines)

# -----------------------------
# GPT SQL Generation with multi-table awareness
# -----------------------------


def generate_sql_from_question(question: str) -> str:
    """
    GPT now knows all three tables: households, family_members, expatriates
    and that they are linked via household_id. GPT can generate JOINs.
    """
    prompt = f"""
You are a strict SQL generator for DuckDB.

Rules:
- Output SQL ONLY.
- Must be a single SELECT query.
- Tables available:
  1. households
  2. family_members (linked via household_id)
  3. expatriates (linked via household_id)
- Do NOT use INSERT/UPDATE/DELETE/CREATE/DROP/ATTACH/COPY/PRAGMA.
- Quote Arabic column names or columns with spaces using double quotes.
- Prefer COUNT(*), SUM(), AVG(), GROUP BY, ORDER BY.
- If needed, join family_members or expatriates using households.household_id.

Table schemas:

households:
{df_schema_text(df_households)}

family_members:
{df_schema_text(df_family)}

expatriates:
{df_schema_text(df_expat)}

User question (Arabic possible):
{question}
""".strip()

    resp = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": "You are a strict SQL generator "
             "for DuckDB."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )
    sql = resp.choices[0].message.content.strip()
    sql = sql.replace("```sql", "").replace("```", "").strip()
    return sql


def is_safe_select(sql: str) -> bool:
    if not sql:
        return False
    statements = [s for s in sqlparse.split(sql) if s.strip()]
    if len(statements) != 1:
        return False
    s = statements[0].strip().lower()
    if not s.startswith("select"):
        return False
    blocked = [
        "insert", "update", "delete", "drop", "create", "alter", "attach",
        "copy", "pragma", "call", "export", "import"
    ]
    if any(b in s for b in blocked):
        return False
    if " from households" not in s and "from households" not in s:
        return False
    return True


def run_sql_on_df(sql: str):
    return con.execute(sql).df()

# -----------------------------
# Routes
# -----------------------------


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
        mimetype="application/vnd.openxmlformats-officedocument"
        ".spreadsheetml.sheet"
    )


@app.route("/")
def index():
    stats = prepare_stats(df_households)
    return render_template("index.html", stats=stats)


@app.route("/ask", methods=["POST"])
def ask():
    question = request.json.get("question", "").strip()
    if not question:
        return jsonify({"answer": "Please enter a question."}), 400
    try:
        sql = generate_sql_from_question(question)
        if not is_safe_select(sql):
            return jsonify({
                "answer": "I couldn't generate a safe SQL query for that "
                "question. Please rephrase.",
                "sql": sql
            }), 400
        result_df = run_sql_on_df(sql)
        download_url = None
        if not result_df.empty and not (result_df.shape == (1, 1)):
            result_id = store_result_df(result_df)
            download_url = f"/download/{result_id}.xlsx"

        if result_df.empty:
            answer_text = "No results found."
        elif result_df.shape == (1, 1):
            answer_text = str(result_df.iat[0, 0])
        else:
            answer_text = result_df.head(5).to_markdown(index=False)

        return jsonify({
            "answer": answer_text,
            "sql": sql,
            "download_url": download_url
        })
    except Exception as e:
        return jsonify({"answer": f"Error running query: {str(e)}"}), 500


# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(debug=True, host="0.0.0.0", port=port)
