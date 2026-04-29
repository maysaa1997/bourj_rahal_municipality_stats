# app.py
import os
import logging
from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import numpy as np
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
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY environment variable is not set.")
client = OpenAI(api_key=OPENAI_API_KEY)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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


def load_data():
    households_json = load_households(DATA_FILE)
    df_households = pd.json_normalize(households_json)

    family_rows = []
    for hh in households_json:
        hh_id = hh["household_id"]
        for member in (hh.get("family_members") or []):
            member_row = member.copy()
            member_row["household_id"] = hh_id
            family_rows.append(member_row)
    df_family = pd.DataFrame(family_rows)

    expat_rows = []
    for hh in households_json:
        hh_id = hh["household_id"]
        for expat in (hh.get("expatriates") or []):
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
if not df_family.empty:
    con.register("family_members", df_family)
if not df_expat.empty:
    con.register("expatriates", df_expat)

# -----------------------------
# Comprehensive stats
# -----------------------------
CURRENT_YEAR = 2026


def safe_groupby(df, col):
    if col not in df.columns:
        return {}
    return df.dropna(subset=[col]).groupby(col).size().to_dict()


def safe_explode_groupby(df, col):
    if col not in df.columns:
        return {}
    return (
        df.explode(col)
        .dropna(subset=[col])
        .loc[lambda x: x[col] != ""]
        .groupby(col)
        .size()
        .to_dict()
    )


def compute_age_brackets(df):
    if "تاريخ الولادة" not in df.columns:
        return {}
    ages = CURRENT_YEAR - df["تاريخ الولادة"].dropna()
    brackets = pd.cut(
        ages,
        bins=[0, 18, 30, 45, 60, 120],
        labels=["أقل من 18", "18-30", "31-45", "46-60", "أكثر من 60"],
        right=True,
    )
    return brackets.value_counts().sort_index().to_dict()


def compute_rooms_distribution(df):
    if "عدد الغرف" not in df.columns:
        return {}
    rooms = df["عدد الغرف"].dropna().astype(int)
    bins = pd.cut(
        rooms,
        bins=[0, 2, 4, 6, 100],
        labels=["1-2 غرف", "3-4 غرف", "5-6 غرف", "7+ غرف"],
        right=True,
    )
    return bins.value_counts().sort_index().to_dict()


def prepare_stats(df, df_fam=None, df_exp=None):
    if df_fam is None:
        df_fam = df_family
    if df_exp is None:
        df_exp = df_expat

    stats = {}

    # --- Demographics ---
    stats["الأحياء"] = safe_groupby(df, "الحي")
    stats["الجنس"] = safe_groupby(df, "الجنس")
    stats["الوضع_العائلي"] = safe_groupby(df, "الوضع العائلي")
    stats["فئة_الدم"] = safe_groupby(df, "فئة الدم")
    stats["الفئة_العمرية"] = compute_age_brackets(df)

    # --- Gender comparison across all groups ---
    gender_comparison = {}
    # Heads of household
    head_g = safe_groupby(df, "الجنس")
    gender_comparison["أرباب_الأسر"] = {"ذكور": head_g.get(
        "Male", 0), "إناث": head_g.get("Female", 0)}
    # Conjoints
    if "conjoint_الجنس" in df.columns:
        conj_g = df[df["conjoint_الاسم_الأول"].notna() & (df["conjoint_الاسم_الأول"].astype(
            str).str.strip() != "")]["conjoint_الجنس"].value_counts().to_dict()
        gender_comparison["الأزواج"] = {"ذكور": conj_g.get(
            "Male", 0), "إناث": conj_g.get("Female", 0)}
    # Family members
    if not df_fam.empty and "الجنس" in df_fam.columns:
        fam_g = df_fam["الجنس"].value_counts().to_dict()
        gender_comparison["أفراد_الأسر"] = {"ذكور": fam_g.get(
            "Male", 0), "إناث": fam_g.get("Female", 0)}
    # Expatriates
    if not df_exp.empty and "الجنس" in df_exp.columns:
        exp_g = df_exp["الجنس"].value_counts().to_dict()
        gender_comparison["المغتربون"] = {"ذكور": exp_g.get(
            "Male", 0), "إناث": exp_g.get("Female", 0)}
    # Total
    total_m = sum(v.get("ذكور", 0) for v in gender_comparison.values())
    total_f = sum(v.get("إناث", 0) for v in gender_comparison.values())
    gender_comparison["الإجمالي"] = {"ذكور": total_m, "إناث": total_f}
    stats["مقارنة_الجنس"] = gender_comparison

    # --- Housing ---
    stats["ملكية_السكن"] = safe_groupby(df, "ملكية السكن")
    stats["نوع_المسكن"] = safe_groupby(df, "نوع المسكن")
    stats["توزيع_الغرف"] = compute_rooms_distribution(df)

    # --- Infrastructure ---
    stats["الصرف_الصحي"] = safe_explode_groupby(df, "الصرف الصحي")
    stats["مصدر_المياه"] = safe_explode_groupby(df, "مصدر المياه")
    stats["مصدر_كهرباء"] = safe_explode_groupby(df, "مصدر كهرباء")
    stats["خدمة_انترنت"] = safe_groupby(df, "خدمة انترنت")
    stats["خدمة_تلفون_أرضي"] = safe_groupby(df, "خدمة تلفون أرضي")
    stats["جمع_النفايات"] = safe_groupby(df, "هل تقوم البلدية بجمع النفايات؟")

    # --- Assets ---
    stats["ملكية_سيارة"] = safe_groupby(df, "ملكية سيارة")
    stats["ملكية_كمبيوتر"] = safe_groupby(df, "ملكية كمبيوتر ")
    stats["عاملة_منزل"] = safe_groupby(df, "عاملة  منزل")

    # --- Income ---
    stats["الدخل_الشهري"] = safe_groupby(df, "الدخل الشهري")

    # --- Family ---
    stats["المغتربون"] = safe_groupby(df, "أحد أفراد الأسرة مغترب")
    stats["ذوي_الاحتياجات_الخاصة"] = safe_groupby(
        df, "وجود شخص من ذوي الإحتياجات الخاصة "
    )

    # --- Disability details ---
    col_disability = "وجود شخص من ذوي الإحتياجات الخاصة "
    col_count = "عدد الإحتياجات الخاصة"
    if col_disability in df.columns:
        disabled_df = df[df[col_disability].astype(str).str.strip() == "نعم"]
        if not disabled_df.empty:
            stats["ذوي_الاحتياجات_حسب_الحي"] = safe_groupby(
                disabled_df, "الحي")
            # Total people with disabilities
            if col_count in disabled_df.columns:
                counts = pd.to_numeric(
                    disabled_df[col_count], errors="coerce").dropna()
                stats["عدد_ذوي_الاحتياجات"] = int(counts.sum())
                stats["أسر_ذوي_الاحتياجات"] = len(disabled_df)
            else:
                stats["عدد_ذوي_الاحتياجات"] = len(disabled_df)
                stats["أسر_ذوي_الاحتياجات"] = len(disabled_df)

    # --- Employment ---
    professions = safe_groupby(df, "المهنة")
    if professions:
        stats["المهن"] = dict(
            sorted(professions.items(), key=lambda x: x[1], reverse=True)[:15]
        )

    # --- Education (from family_members) ---
    if not df_fam.empty:
        stats["التعليم"] = safe_groupby(df_fam, "التعليم")

    # --- Expatriates ---
    if not df_exp.empty:
        stats["مكان_الإقامة"] = safe_groupby(df_exp, "مكان الإقامة")
        stats["نوع_الاغتراب"] = safe_groupby(df_exp, "العمل/الدراسة")

    return stats


def compute_summary(df, df_fam=None, df_exp=None):
    if df_fam is None:
        df_fam = df_family
    if df_exp is None:
        df_exp = df_expat

    total = len(df)

    def pct(col, val):
        if col not in df.columns:
            return 0
        count = df[col].astype(str).str.strip().eq(val).sum()
        return round(count / total * 100, 1) if total else 0

    family_sizes = df["عدد أفراد الأسرة"].dropna()
    avg_family = round(family_sizes.mean(), 1) if len(family_sizes) else 0

    neighborhoods = df["الحي"].nunique() if "الحي" in df.columns else 0

    # Count conjoints (spouse) from household-level fields
    conjoint_count = 0
    if "conjoint_الاسم_الأول" in df.columns:
        conjoint_count = df["conjoint_الاسم_الأول"].dropna().astype(
            str).str.strip().ne("").sum()

    # Total population = heads of household + conjoints + family members + expatriates
    total_population = total + int(conjoint_count) + len(df_fam) + len(df_exp)

    # Count people with disabilities
    col_disability = "وجود شخص من ذوي الإحتياجات الخاصة "
    col_count = "عدد الإحتياجات الخاصة"
    total_pwd = 0
    if col_disability in df.columns:
        disabled_df = df[df[col_disability].astype(str).str.strip() == "نعم"]
        if col_count in disabled_df.columns:
            total_pwd = int(pd.to_numeric(
                disabled_df[col_count], errors="coerce").dropna().sum())
        else:
            total_pwd = len(disabled_df)

    return {
        "إجمالي_الأسر": total,
        "إجمالي_السكان": total_population,
        "إجمالي_أفراد_الأسر": len(df_fam),
        "إجمالي_المغتربين": len(df_exp),
        "عدد_الأزواج": int(conjoint_count),
        "ذوي_الاحتياجات_الخاصة": total_pwd,
        "متوسط_حجم_الأسرة": avg_family,
        "عدد_الأحياء": neighborhoods,
        "نسبة_الإنترنت": pct("خدمة انترنت", "نعم"),
        "نسبة_ملكية_سيارة": pct("ملكية سيارة", "نعم"),
        "نسبة_ملكية_السكن": pct("ملكية السكن", "ملك"),
    }


# Get filter options for the frontend
def get_filter_options():
    return {
        "neighborhoods": sorted(
            df_households["الحي"].dropna().unique().tolist()
        ),
        "incomes": df_households["الدخل الشهري"].dropna().unique().tolist(),
        "ownership": df_households["ملكية السكن"]
        .dropna()
        .unique()
        .tolist(),
    }


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
# GPT SQL Generation (gpt-4o, improved prompt)
# -----------------------------

NEIGHBORHOODS = "، ".join(
    df_households["الحي"].dropna().unique().tolist()
)
INCOME_LEVELS = "، ".join(
    df_households["الدخل الشهري"].dropna().unique().tolist()
)


def generate_sql_from_question(question: str) -> str:
    prompt = f"""You are a strict SQL generator for DuckDB.

Rules:
- Output ONLY the SQL query, no explanation.
- Must be a single SELECT query (UNION ALL is allowed).
- Tables available:
  1. households (main table, one row per household)
  2. family_members (linked via household_id, one row per family member)
  3. expatriates (linked via household_id, one row per expatriate)
- Do NOT use INSERT/UPDATE/DELETE/CREATE/DROP/ATTACH/COPY/PRAGMA.
- Quote Arabic column names or columns with spaces using double quotes.
- Prefer COUNT(*), SUM(), AVG(), GROUP BY, ORDER BY.
- JOIN family_members or expatriates using household_id when needed.
- For array columns (الصرف الصحي, مصدر المياه, مصدر كهرباء), use UNNEST() to expand.
- Yes/No values in Arabic: نعم = yes, كلا = no.
- Gender values: Male, Female (in English in the data).

IMPORTANT - Searching for a person by name:
Names exist in MULTIPLE tables. When the user asks about a person, you MUST search ALL of these:
  1. households."إسم رب الأسرة" (head of household full name)
  2. households."conjoint_الاسم_الأول" (spouse/conjoint first name)
  3. family_members."الاسم الأول" (family member first name)
  4. expatriates."الاسم الأول" (expatriate first name)
Use UNION ALL to combine results from all tables, or use multiple LEFT JOINs.
Use LIKE '%name%' for partial matching since some fields have first name only.
Always use the FULL name the user provides in a single LIKE '%full_name%' — do NOT split the name into parts.
Always use partial LIKE matching — NEVER use exact equality (=) for names.
When the user asks for DETAILS about a person, include household info (الحي, نوع المسكن, etc.) by JOINing with households.
Example for searching details about "أحمد علي":
SELECT 'رب الأسرة' AS المصدر, h."إسم رب الأسرة" AS الاسم, h."الحي", h."نوع المسكن", h."الدخل الشهري", h.household_id FROM households h WHERE h."إسم رب الأسرة" LIKE '%أحمد علي%'
UNION ALL
SELECT 'الزوج/ة' AS المصدر, h."conjoint_الاسم_الأول" AS الاسم, h."الحي", h."نوع المسكن", h."الدخل الشهري", h.household_id FROM households h WHERE h."conjoint_الاسم_الأول" LIKE '%أحمد علي%'
UNION ALL
SELECT 'فرد الأسرة' AS المصدر, f."الاسم الأول" AS الاسم, h."الحي", h."نوع المسكن", h."الدخل الشهري", f.household_id FROM family_members f JOIN households h ON f.household_id = h.household_id WHERE f."الاسم الأول" LIKE '%أحمد علي%'
UNION ALL
SELECT 'مغترب' AS المصدر, e."الاسم الأول" AS الاسم, h."الحي", h."نوع المسكن", h."الدخل الشهري", e.household_id FROM expatriates e JOIN households h ON e.household_id = h.household_id WHERE e."الاسم الأول" LIKE '%أحمد علي%'

Data reference:
- Neighborhoods (الحي): {NEIGHBORHOODS}
- Income levels (الدخل الشهري): {INCOME_LEVELS}
- Ownership (ملكية السكن): ملك, أجار
- تاريخ الولادة contains birth year (e.g. 1990). Current year is {CURRENT_YEAR}.

Table schemas:

households:
{df_schema_text(df_households)}

family_members:
{df_schema_text(df_family) if not df_family.empty else "(empty table)"}

expatriates:
{df_schema_text(df_expat) if not df_expat.empty else "(empty table)"}

User question:
{question}"""

    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": "You are a strict SQL generator for DuckDB. "
                "Output ONLY the SQL query.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0,
        timeout=30,
    )
    sql = resp.choices[0].message.content.strip()
    sql = sql.replace("```sql", "").replace("```", "").strip()
    return sql


def interpret_answer(question: str, sql: str, result_text: str) -> str:
    prompt = f"""أنت مساعد ذكي لبلدية برج رحال. أجب عن سؤال المستخدم بناءً على نتائج الاستعلام.

السؤال: {question}
الاستعلام: {sql}
النتيجة: {result_text}

القواعد:
- أجب بالعربية بشكل واضح ومختصر.
- إذا كانت النتيجة رقماً واحداً، اذكره مع شرح بسيط.
- إذا كانت النتيجة جدولاً، لخّص أهم النقاط.
- لا تذكر تفاصيل SQL.
- إذا لم تكن هناك نتائج، أخبر المستخدم بذلك بلطف.
- عند البحث عن شخص: ابدأ بالتطابق الأقرب للاسم المطلوب، ثم اذكر الأسماء المشابهة.
- اذكر نوع السجل (رب الأسرة، الزوج/ة، فرد الأسرة، مغترب) لكل نتيجة.
- اعرض كل النتائج الموجودة، لا تخفِ أي نتيجة."""

    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": "أنت مساعد بلدية برج رحال. أجب بالعربية فقط.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
        timeout=30,
    )
    return resp.choices[0].message.content.strip()


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
        "insert",
        "update",
        "delete",
        "drop",
        "create",
        "alter",
        "attach",
        "copy",
        "pragma",
        "call",
        "export",
        "import",
    ]
    if any(b in s for b in blocked):
        return False
    # Allow any of the 3 registered tables
    valid_tables = ["households", "family_members", "expatriates"]
    if not any(f"from {t}" in s for t in valid_tables):
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
        return jsonify({"error": "النتيجة غير موجودة أو منتهية الصلاحية"}), 404
    df_out = QUERY_RESULTS[result_id]
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df_out.to_excel(writer, index=False, sheet_name="نتيجة")
    output.seek(0)
    return send_file(
        output,
        as_attachment=True,
        download_name=f"query_result_{result_id}.xlsx",
        mimetype="application/vnd.openxmlformats-officedocument"
        ".spreadsheetml.sheet",
    )


@app.route("/download/dataset.xlsx", methods=["GET"])
def download_dataset():
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df_households.to_excel(
            writer, index=False, sheet_name="الأسر"
        )
        if not df_family.empty:
            df_family.to_excel(
                writer, index=False, sheet_name="أفراد الأسر"
            )
        if not df_expat.empty:
            df_expat.to_excel(
                writer, index=False, sheet_name="المغتربون"
            )
    output.seek(0)
    return send_file(
        output,
        as_attachment=True,
        download_name="بيانات_بلدية_برج_رحال.xlsx",
        mimetype="application/vnd.openxmlformats-officedocument"
        ".spreadsheetml.sheet",
    )


@app.route("/")
def index():
    stats = prepare_stats(df_households)
    summary = compute_summary(df_households)
    filters = get_filter_options()
    return render_template(
        "index.html", stats=stats, summary=summary, filters=filters
    )


@app.route("/api/stats", methods=["GET"])
def api_stats():
    df = df_households.copy()
    df_fam = df_family.copy() if not df_family.empty else df_family
    df_exp = df_expat.copy() if not df_expat.empty else df_expat

    # Apply filters
    neighborhood = request.args.get("neighborhood")
    income = request.args.get("income")
    ownership = request.args.get("ownership")

    if neighborhood:
        df = df[df["الحي"] == neighborhood]
        hh_ids = df["household_id"].tolist()
        if not df_fam.empty:
            df_fam = df_fam[df_fam["household_id"].isin(hh_ids)]
        if not df_exp.empty:
            df_exp = df_exp[df_exp["household_id"].isin(hh_ids)]
    if income:
        df = df[df["الدخل الشهري"] == income]
    if ownership:
        df = df[df["ملكية السكن"].str.strip() == ownership.strip()]

    stats = prepare_stats(df, df_fam, df_exp)
    summary = compute_summary(df, df_fam, df_exp)

    # Convert numpy types to native Python for JSON serialization
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        return obj

    return jsonify({
        "stats": convert(stats),
        "summary": convert(summary),
    })


@app.route("/api/summary", methods=["GET"])
def api_summary():
    summary = compute_summary(df_households)

    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        return obj

    return jsonify({k: convert(v) for k, v in summary.items()})


@app.route("/ask", methods=["POST"])
def ask():
    question = request.json.get("question", "").strip()
    if not question:
        return jsonify({"answer": "الرجاء إدخال سؤال."}), 400
    try:
        sql = generate_sql_from_question(question)
        if not is_safe_select(sql):
            logger.warning("Unsafe SQL blocked: %s", sql)
            return jsonify({
                "answer": "لم أتمكن من إنشاء استعلام آمن لهذا السؤال. "
                "يرجى إعادة صياغة السؤال.",
                "sql": sql,
            }), 400

        result_df = run_sql_on_df(sql)
        download_url = None
        if not result_df.empty and not (result_df.shape == (1, 1)):
            result_id = store_result_df(result_df)
            download_url = f"/download/{result_id}.xlsx"

        if result_df.empty:
            raw_text = "لا توجد نتائج."
        elif result_df.shape == (1, 1):
            raw_text = str(result_df.iat[0, 0])
        else:
            raw_text = result_df.to_markdown(index=False)

        # Get Arabic interpretation from GPT
        answer_text = interpret_answer(question, sql, raw_text)

        return jsonify({
            "answer": answer_text,
            "sql": sql,
            "raw": raw_text,
            "download_url": download_url,
        })
    except Exception as e:
        logger.error("Error in /ask: %s", str(e))
        return jsonify({
            "answer": f"حدث خطأ أثناء تنفيذ الاستعلام: {str(e)}"
        }), 500


# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(debug=True, host="0.0.0.0", port=port)
