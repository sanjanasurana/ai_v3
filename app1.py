import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import google.generativeai as genai
from openai import OpenAI
from mistralai import Mistral
import re
from datetime import datetime
import os
from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
import json
from bson import ObjectId

load_dotenv()

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="AI Intelligence Platform V3",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
    }
    .section-header {
        font-size: 1.8rem;
        font-weight: 600;
        color: #667eea;
        border-bottom: 2px solid #667eea;
        padding-bottom: 0.5rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# MONGODB FUNCTIONS
# ============================================================================

@st.cache_resource
def get_mongodb_client():
    try:
        mongo_uri = os.getenv("MONGODB_URI", "")
        if not mongo_uri:
            return None
        client = MongoClient(
            mongo_uri,
            serverSelectionTimeoutMS=5000,
            connectTimeoutMS=10000,
            socketTimeoutMS=10000
        )
        client.admin.command('ping')
        return client
    except (ConnectionFailure, ServerSelectionTimeoutError) as e:
        st.error(f"MongoDB Connection Error: {e}")
        return None
    except Exception as e:
        st.error(f"Unexpected MongoDB Error: {e}")
        return None


def get_database_collection(module_name):
    client = get_mongodb_client()
    if client is None:
        return None, None
    if module_name == "server_health":
        db = client['ai_platform_v1']
        collection = db['server_health_queries']
    elif module_name == "inventory":
        db = client['ai_platform_v2']
        collection = db['inventory_queries']
    else:
        return None, None
    return db, collection


def save_query_to_mongodb(module_name, query_data):
    try:
        db, collection = get_database_collection(module_name)
        if collection is None:
            st.warning("⚠️ MongoDB not configured. Query not saved.")
            return None
        query_data['timestamp'] = datetime.now()
        query_data['module'] = module_name
        result = collection.insert_one(query_data)
        return str(result.inserted_id)
    except Exception as e:
        st.error(f"Error saving to MongoDB: {e}")
        return None


def get_query_history(module_name, limit=10):
    try:
        db, collection = get_database_collection(module_name)
        if collection is None:
            return []
        queries = collection.find().sort('timestamp', -1).limit(limit)
        return list(queries)
    except Exception as e:
        st.error(f"Error retrieving history: {e}")
        return []


def get_context_from_db(module_name, limit=5):
    history = get_query_history(module_name, limit=limit)
    if not history:
        return "No historical data found in the database."
    context_str = "--- START OF HISTORICAL DATABASE ENTRIES ---\n"
    for i, doc in enumerate(history):
        ts = doc.get('timestamp', 'N/A')
        prompt = doc.get('user_prompt', 'N/A')
        response = doc.get('ai_response', 'N/A')
        summary_res = response[:500] + "..." if len(response) > 500 else response
        context_str += f"Entry {i+1} [Time: {ts}]:\n"
        context_str += f"User asked: {prompt}\n"
        context_str += f"AI previously answered: {summary_res}\n\n"
    context_str += "--- END OF HISTORICAL DATABASE ENTRIES ---"
    return context_str


def format_query_for_display(query_doc):
    timestamp = query_doc.get('timestamp', 'Unknown')
    if isinstance(timestamp, datetime):
        timestamp = timestamp.strftime('%Y-%m-%d %H:%M:%S')
    return {
        'ID': str(query_doc.get('_id', 'N/A')),
        'Timestamp': timestamp,
        'Provider': query_doc.get('ai_config', {}).get('provider', 'N/A'),
        'Model': query_doc.get('ai_config', {}).get('model', 'N/A'),
        'Prompt': (query_doc.get('user_prompt', 'N/A')[:100] + '...')
                  if len(query_doc.get('user_prompt', '')) > 100
                  else query_doc.get('user_prompt', 'N/A'),
    }

# ============================================================================
# KNOWLEDGE BASE FUNCTIONS
# ============================================================================

def get_knowledge_base_collection():
    client = get_mongodb_client()
    if client is None:
        return None
    db = client['ai_platform_knowledge_base']
    collection = db['qa_repository']
    return collection


def search_knowledge_base(query, category=None, limit=5):
    try:
        collection = get_knowledge_base_collection()
        if collection is None:
            return []
        search_filter = {}
        if category:
            search_filter['category'] = category
        search_filter['$or'] = [
            {'question': {'$regex': query, '$options': 'i'}},
            {'answer': {'$regex': query, '$options': 'i'}}
        ]
        results = collection.find(search_filter).limit(limit)
        return list(results)
    except Exception as e:
        print(f"Error searching knowledge base: {e}")
        return []


def get_all_kb_categories():
    try:
        collection = get_knowledge_base_collection()
        if collection is None:
            return []
        categories = collection.distinct('category')
        return sorted(categories)
    except Exception as e:
        print(f"Error getting categories: {e}")
        return []


def get_kb_by_category(category):
    try:
        collection = get_knowledge_base_collection()
        if collection is None:
            return []
        results = collection.find({'category': category}).sort('question', 1)
        return list(results)
    except Exception as e:
        print(f"Error getting category data: {e}")
        return []


def format_kb_context(kb_results, max_entries=3):
    if not kb_results:
        return ""
    context = "\n--- KNOWLEDGE BASE REFERENCE ---\n"
    context += "Here are relevant Q&As from the knowledge base:\n\n"
    for idx, entry in enumerate(kb_results[:max_entries], 1):
        context += f"KB Entry {idx}:\n"
        context += f"Category: {entry.get('category', 'N/A')}\n"
        context += f"Q: {entry.get('question', 'N/A')}\n"
        a = entry.get('answer', 'N/A')
        context += f"A: {a[:500]}{'...' if len(a) > 500 else ''}\n\n"
    context += "--- END KNOWLEDGE BASE REFERENCE ---\n"
    return context


def enhance_prompt_with_kb(user_prompt, module_name):
    category_keywords = {
        'General': ['inventory', 'alert', 'general'],
        'Switches': ['switch', 'switches', 'network'],
        'Firewall': ['firewall', 'security'],
        'Servers': ['server', 'os', 'operating system'],
        'Database': ['database', 'db', 'sql', 'query'],
        'UPS': ['ups', 'power', 'battery'],
        'Printers': ['printer', 'print']
    }
    detected_category = None
    prompt_lower = user_prompt.lower()
    for category, keywords in category_keywords.items():
        if any(keyword in prompt_lower for keyword in keywords):
            detected_category = category
            break
    kb_results = search_knowledge_base(user_prompt, category=detected_category, limit=3)
    if kb_results:
        kb_context = format_kb_context(kb_results)
        return kb_context, detected_category, len(kb_results)
    return "", None, 0

# ============================================================================
# DATA LOADING
# ============================================================================

def clean_column_name(column_name):
    name = str(column_name).replace('(RAW)', '').replace('(raw)', '').strip()
    name = ''.join(char if char.isalnum() else '_' for char in name)
    name = name.lower()
    while '__' in name:
        name = name.replace('__', '_')
    return name.strip('_')


@st.cache_data
def load_server_data():
    cpu_file = "historicdataCPU.csv"
    disk_file = "historicdataDISK.csv"
    try:
        cpu_df = load_and_process_data_source(cpu_file, "cpu")
        disk_df = load_and_process_data_source(disk_file, "disk")
        if cpu_df is not None and disk_df is not None:
            combined_df = pd.merge(
                cpu_df.reset_index(),
                disk_df.reset_index(),
                on=clean_column_name("Date Time"),
                how='inner',
                suffixes=('_cpu', '_disk')
            )
            if 'interval_start_dt_cpu' in combined_df.columns:
                combined_df['interval_start_dt'] = combined_df['interval_start_dt_cpu']
            if 'interval_start_dt_disk' in combined_df.columns:
                combined_df.drop(columns=['interval_start_dt_disk'], inplace=True)
            combined_df.set_index(clean_column_name("Date Time"), inplace=True, drop=False)
            return combined_df
        return None
    except Exception as e:
        st.error(f"Error loading server data: {e}")
        return None


def load_and_process_data_source(file_path, data_type_name):
    try:
        df_original = pd.read_csv(file_path)
    except FileNotFoundError:
        return None
    df_processed = pd.DataFrame()
    cleaned_date_time_key = clean_column_name("Date Time")
    if "Date Time" in df_original.columns:
        df_processed[cleaned_date_time_key] = df_original["Date Time"].astype(str).str.strip()
    else:
        return None
    for col in df_original.columns:
        if col in ("Date Time", "Date Time(RAW)"):
            continue
        if '(RAW)' in col or '(raw)' in col:
            base_name = col.replace('(RAW)', '').replace('(raw)', '').strip()
            final_name = clean_column_name(base_name)
            if final_name in ["total", "downtime", "coverage"]:
                final_name = f"{data_type_name.lower()}_{final_name}"
            series = df_original[col]
            if pd.api.types.is_string_dtype(series):
                series = series.str.replace(',', '', regex=False)
            df_processed[final_name] = pd.to_numeric(series, errors='coerce').fillna(0.0)
    try:
        df_processed['interval_start_dt'] = pd.to_datetime(
            df_processed[cleaned_date_time_key].str.split(' - ').str[0],
            errors='coerce'
        )
    except Exception:
        df_processed['interval_start_dt'] = pd.NaT
    df_processed.set_index(cleaned_date_time_key, inplace=True)
    return df_processed


# ─────────────────────────────────────────────────────────────────────────────
# CORE PARSING HELPERS  (used when New Tickets sheet is absent)
# ─────────────────────────────────────────────────────────────────────────────

def _parse_inv_code(msg):
    m = re.search(r'Inv Code\s*-\s*(\S+)', str(msg))
    return m.group(1).strip() if m else None

def _parse_device_type(msg):
    m = re.search(r'Device Type\s*-\s*([^,]+)', str(msg))
    return m.group(1).strip() if m else None

def _parse_device_error(msg):
    m = re.search(r'Device Error\s*-(.+?)(\s*,|$)', str(msg))
    return m.group(1).strip() if m else None

def _parse_table_frag(msg):
    m = re.search(r'Table Frag\s*=\s*([\d]+)%', str(msg))
    return m.group(1).strip() + '%' if m else 'N/A'

def _parse_query_resp_ms(device_error):
    """Extract numeric ms from 'Q.Resp =2000 ms' style strings."""
    m = re.search(r'Q\.Resp\s*=\s*([\d]+)\s*ms', str(device_error))
    return int(m.group(1)) if m else None

def _parse_table_frag_numeric(table_frag_str):
    """Convert '60%' → 60 (float), 'N/A' → None."""
    if str(table_frag_str) == 'N/A':
        return None
    m = re.search(r'([\d.]+)', str(table_frag_str))
    return float(m.group(1)) if m else None


@st.cache_data
def load_inventory_data(file_path: str = "testdata.xlsx"):
    """
    Loads inventory + ticket data.

    Priority:
      1. Use 'New Tickets for Analysis' sheet if it exists — it already has
         Inv_Code_Parsed, Device_Type_Parsed, Device_Error, Table_Frag columns.
      2. Fall back to 'Tickets for Analysis' and parse Error_Message on the fly.

    Returns: inventory_df, tickets_df, merged_df
    All DataFrames use lowercase snake_case column names.
    tickets_df always contains:
        ticket_id, timestamp, inv_code, device_type,
        inv_code_parsed, device_type_parsed, device_error, table_frag,
        month, hour, day_of_week,
        query_resp_ms (numeric, DB rows only),
        table_frag_numeric (float, DB rows only)
    """
    try:
        xl = pd.ExcelFile(file_path)
        sheet_names = xl.sheet_names

        # ── INVENTORY ──────────────────────────────────────────────────────
        inv_raw = pd.read_excel(file_path, sheet_name="Inventory Work Sheet", header=0)
        inv_raw.columns = [clean_column_name(c) for c in inv_raw.columns]
        inventory = inv_raw[['serial_no', 'inv_code', 'device_type']].dropna(subset=['inv_code']).copy()
        inventory['inv_code'] = inventory['inv_code'].astype(str).str.strip()
        inventory['device_type'] = inventory['device_type'].astype(str).str.strip()

        # ── TICKETS ────────────────────────────────────────────────────────
        NEW_SHEET = "New Tickets for Analysis"

        if NEW_SHEET in sheet_names:
            # ── Path A: pre-parsed sheet ────────────────────────────────
            tix_raw = pd.read_excel(file_path, sheet_name=NEW_SHEET, header=0)
            tix_raw.columns = [clean_column_name(c) for c in tix_raw.columns]

            # Expected cleaned names:
            # ticket_id, timestamp, inv_code, device_type,
            # inv_code_parsed, device_type_parsed, device_error, table_frag
            required = {'ticket_id', 'timestamp', 'inv_code', 'device_type',
                        'inv_code_parsed', 'device_type_parsed', 'device_error', 'table_frag'}
            missing = required - set(tix_raw.columns)
            if missing:
                st.warning(f"⚠️ '{NEW_SHEET}' is missing columns: {missing}. "
                           "Falling back to on-the-fly parsing.")
                raise ValueError("Missing columns in new sheet")

            tickets = tix_raw[list(required)].dropna(subset=['inv_code']).copy()
            tickets['inv_code']         = tickets['inv_code'].astype(str).str.strip()
            tickets['device_type']      = tickets['device_type'].astype(str).str.strip()
            tickets['inv_code_parsed']  = tickets['inv_code_parsed'].astype(str).str.strip()
            tickets['device_type_parsed'] = tickets['device_type_parsed'].astype(str).str.strip()
            tickets['device_error']     = tickets['device_error'].astype(str).str.strip()
            tickets['table_frag']       = tickets['table_frag'].astype(str).str.strip()

        else:
            # ── Path B: legacy sheet – parse Error_Message on the fly ───
            tix_raw = pd.read_excel(file_path, sheet_name="Tickets for Analysis", header=2)
            tix_raw.columns = [clean_column_name(c) for c in tix_raw.columns]
            tickets = tix_raw[['ticket_id', 'timestamp', 'inv_code',
                                'device_type', 'error_message']].dropna(subset=['inv_code']).copy()
            tickets['inv_code']   = tickets['inv_code'].astype(str).str.strip()
            tickets['device_type'] = tickets['device_type'].astype(str).str.strip()

            tickets['inv_code_parsed']    = tickets['error_message'].apply(_parse_inv_code)
            tickets['device_type_parsed'] = tickets['error_message'].apply(_parse_device_type)
            tickets['device_error']       = tickets['error_message'].apply(_parse_device_error)
            tickets['table_frag']         = tickets['error_message'].apply(_parse_table_frag)
            tickets.drop(columns=['error_message'], inplace=True)

        # ── Common enrichment ──────────────────────────────────────────────
        tickets['timestamp']   = pd.to_datetime(tickets['timestamp'], errors='coerce')
        tickets['month']       = tickets['timestamp'].dt.to_period('M').astype(str)
        tickets['hour']        = tickets['timestamp'].dt.hour
        tickets['day_of_week'] = tickets['timestamp'].dt.day_name()

        # Numeric columns for DB analysis
        tickets['query_resp_ms']      = tickets['device_error'].apply(_parse_query_resp_ms)
        tickets['table_frag_numeric'] = tickets['table_frag'].apply(_parse_table_frag_numeric)

        # ── Merged (inventory + ticket counts) ────────────────────────────
        ticket_count = (
            tickets.groupby('inv_code')
            .size()
            .reset_index(name='ticket_count')
        )
        merged = inventory.merge(ticket_count, on='inv_code', how='left').fillna({'ticket_count': 0})
        merged['ticket_count'] = merged['ticket_count'].astype(int)

        return inventory, tickets, merged

    except Exception as e:
        st.error(f"Error loading inventory data: {e}")
        return None, None, None


# ============================================================================
# BUILD RICH DATA SUMMARY FOR AI PROMPT
# ============================================================================

_SWITCH_TYPES = {"Access Switch", "Dist Switch", "Core Switch", "Int-Link"}
_SERVER_TYPES = {"Windows Server", "Linux Server"}


def build_inventory_summary(inventory: pd.DataFrame,
                             tickets: pd.DataFrame,
                             merged: pd.DataFrame) -> str:
    """
    Creates a detailed, structured text summary of the dataset that is
    injected into every AI prompt so the model can answer accurately.
    All numbers come directly from the data — no guessing.
    """
    lines = []
    lines.append("=== INVENTORY & ALERT DATASET SUMMARY ===")
    lines.append(f"Total inventory items : {len(inventory):,}")
    lines.append(f"Total alert tickets   : {len(tickets):,}")
    lines.append(f"Date range            : {tickets['timestamp'].min().date()} → {tickets['timestamp'].max().date()}")
    lines.append("")

    # ── Alert distribution by device type ──────────────────────────────────
    lines.append("--- ALERTS BY DEVICE TYPE ---")
    by_type = tickets.groupby('device_type').size().sort_values(ascending=False)
    for dtype, cnt in by_type.items():
        lines.append(f"  {dtype}: {cnt:,} alerts")
    lines.append("")

    # ── General: top 10 items, no-alert items, crunch hours ────────────────
    lines.append("--- GENERAL ---")
    top_type = by_type.index[0]
    lines.append(f"Type with most alerts : {top_type} ({by_type.iloc[0]:,})")

    top10 = (
        merged[merged['ticket_count'] > 0]
        .sort_values('ticket_count', ascending=False)
        .head(10)[['inv_code', 'device_type', 'ticket_count']]
    )
    lines.append("Top 10 items with most alerts:")
    for _, row in top10.iterrows():
        lines.append(f"  {row['inv_code']} ({row['device_type']}): {int(row['ticket_count'])}")

    no_alert_items = merged[merged['ticket_count'] == 0]
    lines.append(f"Items with no alerts  : {len(no_alert_items):,}")

    hourly = tickets.groupby('hour').size()
    p95 = np.percentile(hourly, 95)
    crunch = sorted(hourly[hourly >= p95].index.tolist())
    lines.append(f"95th-pct crunch hours : {[f'{h:02d}:00' for h in crunch]}")
    lines.append("")

    # ── Monthly trend ───────────────────────────────────────────────────────
    monthly = tickets.groupby('month').size()
    first_m, last_m = monthly.iloc[0], monthly.iloc[-1]
    pct = (last_m - first_m) / first_m * 100 if first_m else 0
    trend = "increasing" if pct > 5 else ("decreasing" if pct < -5 else "stable")
    lines.append("--- MONTHLY TREND ---")
    lines.append(f"Trend       : {trend} ({pct:+.1f}% from first to last month)")
    lines.append(f"Peak month  : {monthly.idxmax()} ({monthly.max()} alerts)")
    lines.append(f"Lowest month: {monthly.idxmin()} ({monthly.min()} alerts)")
    lines.append("")

    # ── Switches ────────────────────────────────────────────────────────────
    sw = tickets[tickets['device_type'].isin(_SWITCH_TYPES)]
    sw_inv = inventory[inventory['device_type'].isin(_SWITCH_TYPES)]
    lines.append("--- SWITCHES ---")
    lines.append(f"Total switch alerts: {len(sw):,}")
    top5_sw = (
        sw.groupby(['inv_code', 'device_type']).size()
        .sort_values(ascending=False).head(5)
    )
    lines.append("Top 5 switches with most alerts:")
    for (inv_code, dtype), cnt in top5_sw.items():
        lines.append(f"  {inv_code} ({dtype}): {cnt}")
    top_sw_err = sw['device_error'].value_counts().head(5)
    lines.append("Most common switch errors:")
    for err, cnt in top_sw_err.items():
        lines.append(f"  '{err}': {cnt}")
    alerted_sw = set(sw['inv_code'].unique())
    pf_sw = sw_inv[~sw_inv['inv_code'].isin(alerted_sw)]
    lines.append(f"Problem-free switches: {len(pf_sw)} of {len(sw_inv)}")
    sw_hourly = sw.groupby('hour').size()
    if len(sw_hourly) > 0:
        sw_p95 = np.percentile(sw_hourly, 95)
        sw_crunch = sorted(sw_hourly[sw_hourly >= sw_p95].index.tolist())
        lines.append(f"Switch crunch hours (95th pct): {[f'{h:02d}:00' for h in sw_crunch]}")
    lines.append("")

    # ── Firewall ─────────────────────────────────────────────────────────────
    fw = tickets[tickets['device_type'] == 'Firewall']
    lines.append("--- FIREWALL ---")
    lines.append(f"Total firewall alerts: {len(fw):,}")
    top_fw_err = fw['device_error'].value_counts().head(5)
    lines.append("Most common firewall errors:")
    for err, cnt in top_fw_err.items():
        lines.append(f"  '{err}': {cnt}")
    fw_by_inv = fw.groupby('inv_code').size().sort_values(ascending=False)
    lines.append("Alerts per firewall:")
    for inv_code, cnt in fw_by_inv.items():
        lines.append(f"  {inv_code}: {cnt}")
    fw_peak_hour = int(fw.groupby('hour').size().idxmax()) if len(fw) > 0 else 'N/A'
    fw_peak_day  = fw.groupby('day_of_week').size().idxmax() if len(fw) > 0 else 'N/A'
    lines.append(f"Peak firewall alert hour: {fw_peak_hour:02d}:00" if isinstance(fw_peak_hour, int) else f"Peak hour: {fw_peak_hour}")
    lines.append(f"Peak firewall alert day : {fw_peak_day}")
    lines.append("")

    # ── Servers ──────────────────────────────────────────────────────────────
    srv = tickets[tickets['device_type'].isin(_SERVER_TYPES)]
    srv_inv = inventory[inventory['device_type'].isin(_SERVER_TYPES)]
    lines.append("--- SERVERS ---")
    lines.append(f"Total server alerts: {len(srv):,}")
    by_os = srv.groupby('device_type').size().sort_values(ascending=False)
    for os_type, cnt in by_os.items():
        lines.append(f"  {os_type}: {cnt:,} alerts")
    top_srv_err = srv['device_error'].value_counts().head(10)
    lines.append("Most common server errors:")
    for err, cnt in top_srv_err.items():
        lines.append(f"  '{err}': {cnt}")
    alerted_srv = set(srv['inv_code'].unique())
    ef_srv = srv_inv[~srv_inv['inv_code'].isin(alerted_srv)]
    lines.append(f"Error-free servers: {len(ef_srv)} of {len(srv_inv)}")
    top10_srv = (
        srv.groupby(['inv_code', 'device_type']).size()
        .sort_values(ascending=False).head(10)
    )
    lines.append("Top 10 most problematic servers:")
    for (inv_code, dtype), cnt in top10_srv.items():
        lines.append(f"  {inv_code} ({dtype}): {cnt}")

    # Age proxy correlation (serial_no)
    srv_merged = srv_inv.copy()
    srv_merged['serial_no'] = pd.to_numeric(srv_merged['serial_no'], errors='coerce')
    tc = srv.groupby('inv_code').size().reset_index(name='ticket_count')
    srv_merged = srv_merged.merge(tc, on='inv_code', how='left').fillna({'ticket_count': 0})
    age_corr = srv_merged[['serial_no', 'ticket_count']].dropna().corr().iloc[0, 1]
    lines.append(f"Correlation (serial_no vs alerts, age proxy): r={age_corr:.3f} "
                 f"({'no meaningful correlation' if abs(age_corr) < 0.1 else 'some correlation'})")
    lines.append("")

    # ── Database ─────────────────────────────────────────────────────────────
    db = tickets[tickets['device_type'] == 'Database'].dropna(subset=['query_resp_ms'])
    lines.append("--- DATABASE ---")
    lines.append(f"Total DB alerts: {len(db):,}")
    top5_db_err = db['device_error'].value_counts().head(5)
    lines.append("Top 5 DB error messages:")
    for err, cnt in top5_db_err.items():
        lines.append(f"  '{err}': {cnt}")
    if len(db) > 2:
        db_corr = db[['query_resp_ms', 'table_frag_numeric']].dropna().corr().iloc[0, 1]
        lines.append(f"Correlation (Query Response ms vs Table Frag %): r={db_corr:.3f} "
                     f"({'very strong' if abs(db_corr) > 0.9 else 'moderate'} positive correlation)")
        lines.append(f"  → Higher table fragmentation strongly predicts slower query response.")
        lines.append(f"  Query resp stats: min={db['query_resp_ms'].min():.0f} ms, "
                     f"avg={db['query_resp_ms'].mean():.0f} ms, max={db['query_resp_ms'].max():.0f} ms")
        lines.append(f"  Table frag stats: min={db['table_frag_numeric'].min():.0f}%, "
                     f"avg={db['table_frag_numeric'].mean():.1f}%, max={db['table_frag_numeric'].max():.0f}%")
    top5_db_inst = (
        db.groupby('inv_code').size()
        .sort_values(ascending=False).head(5)
    )
    lines.append("Top 5 DB instances with most alerts:")
    for inv_code, cnt in top5_db_inst.items():
        lines.append(f"  {inv_code}: {cnt}")
    lines.append("")

    # ── UPS ──────────────────────────────────────────────────────────────────
    ups = tickets[tickets['device_type'] == 'UPS']
    lines.append("--- UPS ---")
    lines.append(f"Total UPS alerts: {len(ups):,}")
    top_ups = ups.groupby('inv_code').size().sort_values(ascending=False)
    lines.append(f"UPS with most alerts: {top_ups.index[0]} ({top_ups.iloc[0]} alerts)")
    top3_ups_msg = ups['device_error'].value_counts().head(3)
    lines.append("Top 3 UPS messages:")
    for err, cnt in top3_ups_msg.items():
        lines.append(f"  '{err}': {cnt}")
    lines.append("")

    # ── Printers ─────────────────────────────────────────────────────────────
    pri = tickets[tickets['device_type'] == 'Printer']
    if len(pri) > 0:
        lines.append("--- PRINTERS ---")
        lines.append(f"Total printer alerts: {len(pri):,}")
        top_pri_err = pri['device_error'].value_counts().head(5)
        lines.append("Most common printer errors:")
        for err, cnt in top_pri_err.items():
            lines.append(f"  '{err}': {cnt}")
        lines.append("")

    lines.append("=== END OF SUMMARY ===")
    return "\n".join(lines)


# ============================================================================
# SERVER HEALTH FUNCTIONS
# ============================================================================

def format_server_data_for_llm(data_df, start_dt, end_dt):
    if data_df is None or data_df.empty:
        return f"No data found for period: {start_dt} to {end_dt}"
    num_points = len(data_df)
    lines = [f"Server Health Metrics ({num_points} data points)", "---"]
    if 'cpu_total' in data_df.columns:
        lines.append(f"[CPU] Avg: {data_df['cpu_total'].mean():.2f}%, Max: {data_df['cpu_total'].max():.2f}%")
    if 'disk_total' in data_df.columns:
        lines.append(f"[Disk] Avg free: {data_df['disk_total'].mean()/(1024**3):.2f} GB")
    return "\n".join(lines)


def analyze_with_ai(data_snapshot, context, prompt, provider, model,
                    temperature, max_tokens, api_key,
                    db_context="", kb_context=""):
    try:
        system_msg = (
            f"You are an expert server health monitoring AI.\n"
            f"Current System Context: {context}\n\n"
            f"{db_context}\n{kb_context}\n"
            "When answering:\n"
            "1. Consider historical entries from the database to identify trends.\n"
            "2. Reference the knowledge base when relevant.\n"
            "3. Provide actionable insights."
        )
        user_msg = f"Current Live Data Snapshot:\n{data_snapshot}\n\nNew Request:\n{prompt}"

        if provider == "OpenAI":
            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": system_msg},
                           {"role": "user", "content": user_msg}],
                temperature=temperature, max_tokens=max_tokens
            )
            return response.choices[0].message.content

        elif provider == "Google Gemini":
            genai.configure(api_key=api_key)
            gen_model = genai.GenerativeModel(model)
            generation_config = genai.types.GenerationConfig(
                temperature=temperature, max_output_tokens=max_tokens
            )
            response = gen_model.generate_content(
                f"{system_msg}\n\n{user_msg}", generation_config=generation_config
            )
            return response.text

        elif provider == "Mistral AI":
            client = Mistral(api_key=api_key)
            response = client.chat.complete(
                model=model,
                messages=[{"role": "system", "content": system_msg},
                           {"role": "user", "content": user_msg}],
                temperature=temperature, max_tokens=max_tokens
            )
            return response.choices[0].message.content

    except Exception as e:
        return f"Error during {provider} analysis: {e}"


# ============================================================================
# STRUCTURED ANALYTICS  (pre-computed, 100% data-driven)
# ============================================================================

def render_structured_analytics(inventory, tickets, merged):
    """
    Renders the full pre-computed Q&A analytics panel.
    Called inside the '📋 Structured Analytics' tab of Inventory Intelligence.
    All numbers are computed directly from the dataframes — no LLM involved.
    """
    st.markdown("### 📋 Pre-Computed Structured Analytics")
    st.caption("All answers are computed **directly from the dataset** — exact figures, no AI guessing.")

    section = st.selectbox(
        "Jump to section:",
        [
            "📈 Overview — Alert Patterns Over Time",
            "🗂️  General",
            "🔀 Switches",
            "🔥 Firewall",
            "🖥️  Servers",
            "🗄️  Database",
            "🔋 UPS",
        ],
        key="analytics_section"
    )

    # ── OVERVIEW ──────────────────────────────────────────────────────────────
    if section == "📈 Overview — Alert Patterns Over Time":
        monthly = tickets.groupby('month').size().reset_index(name='count')
        first_v, last_v = monthly['count'].iloc[0], monthly['count'].iloc[-1]
        pct = (last_v - first_v) / first_v * 100 if first_v else 0
        trend = "increasing" if pct > 5 else ("decreasing" if pct < -5 else "stable")

        st.subheader("Alert message pattern over time")
        c1, c2 = st.columns([2, 1])
        with c1:
            fig = px.line(monthly, x='month', y='count', markers=True,
                          title="Monthly Alert Volume", template="plotly_dark",
                          labels={"count": "Alerts", "month": "Month"})
            fig.update_traces(line_color='#00d4ff')
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            st.metric("Total Alerts", f"{len(tickets):,}")
            st.metric("Monthly Avg", f"{monthly['count'].mean():.0f}")
            st.metric("Trend", trend.capitalize())
            st.info(
                f"**Verdict:** Alert volume is **{trend}** "
                f"({pct:+.1f}% from first to last month). "
                f"Peak: **{monthly.loc[monthly['count'].idxmax(),'month']}** "
                f"({monthly['count'].max():,}). "
                f"Lowest: **{monthly.loc[monthly['count'].idxmin(),'month']}** "
                f"({monthly['count'].min():,})."
            )

        st.subheader("Alerts by Device Type")
        tpd = tickets.groupby('device_type').size().reset_index(name='alert_count').sort_values('alert_count', ascending=False)
        fig2 = px.bar(tpd, x='device_type', y='alert_count', color='alert_count',
                      color_continuous_scale='Blues', template='plotly_dark',
                      title='Total Alerts per Device Type',
                      labels={'alert_count': 'Alerts', 'device_type': 'Device Type'})
        st.plotly_chart(fig2, use_container_width=True)

    # ── GENERAL ───────────────────────────────────────────────────────────────
    elif section == "🗂️  General":
        st.subheader("General")
        by_type = merged.groupby('device_type')['ticket_count'].sum().sort_values(ascending=False)

        st.markdown("#### Q1 — Type of inventory with most alerts")
        c1, c2 = st.columns([1, 2])
        with c1:
            st.success(f"**{by_type.index[0]}** — {int(by_type.iloc[0]):,} alerts")
            df_bt = by_type.reset_index()
            df_bt.columns = ['Device Type', 'Alerts']
            st.dataframe(df_bt, use_container_width=True)
        with c2:
            fig = px.pie(df_bt, names='Device Type', values='Alerts',
                         template='plotly_dark', title='Alert Share by Device Type',
                         color_discrete_sequence=px.colors.sequential.Blues_r)
            st.plotly_chart(fig, use_container_width=True)

        st.divider()
        st.markdown("#### Q2 — Top 10 inventory items with most alerts")
        top10 = (
            merged[merged['ticket_count'] > 0]
            .sort_values('ticket_count', ascending=False)
            .head(10)[['inv_code', 'device_type', 'ticket_count']]
            .reset_index(drop=True)
        )
        top10.columns = ['Inv Code', 'Device Type', 'Alert Count']
        st.dataframe(top10, use_container_width=True)
        fig2 = px.bar(top10, x='Inv Code', y='Alert Count', color='Device Type',
                      template='plotly_dark', title='Top 10 Items by Alert Count')
        st.plotly_chart(fig2, use_container_width=True)

        st.divider()
        st.markdown("#### Q3 — Inventory items with no alerts")
        no_alerts = merged[merged['ticket_count'] == 0][['inv_code', 'device_type']].reset_index(drop=True)
        st.metric("Items with ZERO alerts", f"{len(no_alerts):,} of {len(merged):,}")
        with st.expander(f"View all {len(no_alerts)} items with no alerts"):
            no_alerts.columns = ['Inv Code', 'Device Type']
            st.dataframe(no_alerts, use_container_width=True)

        st.divider()
        st.markdown("#### Q4 — Crunch time: 95th percentile alert hours")
        hourly = tickets.groupby('hour').size().reset_index(name='Alerts')
        hourly.columns = ['Hour', 'Alerts']
        p95 = np.percentile(hourly['Alerts'], 95)
        fig3 = px.bar(hourly, x='Hour', y='Alerts', template='plotly_dark',
                      color='Alerts', color_continuous_scale='Blues',
                      title='Alert Distribution by Hour of Day')
        fig3.add_hline(y=p95, line_dash='dash', line_color='red',
                       annotation_text=f'95th pct ({p95:.0f})')
        st.plotly_chart(fig3, use_container_width=True)
        crunch = sorted(hourly[hourly['Alerts'] >= p95]['Hour'].tolist())
        st.info(f"⏰ 95th pct crunch hours: **{', '.join(f'{h:02d}:00' for h in crunch)}**")

    # ── SWITCHES ──────────────────────────────────────────────────────────────
    elif section == "🔀 Switches":
        st.subheader("Switches")
        sw = tickets[tickets['device_type'].isin(_SWITCH_TYPES)]
        sw_inv = inventory[inventory['device_type'].isin(_SWITCH_TYPES)]

        st.markdown("#### Q1 — Top 5 switches with most alerts")
        top5 = (
            sw.groupby(['inv_code', 'device_type']).size()
            .reset_index(name='Alert Count')
            .sort_values('Alert Count', ascending=False)
            .head(5).reset_index(drop=True)
        )
        top5.columns = ['Inv Code', 'Type', 'Alert Count']
        c1, c2 = st.columns([1, 2])
        with c1:
            st.dataframe(top5, use_container_width=True)
        with c2:
            fig = px.bar(top5, x='Inv Code', y='Alert Count', color='Type',
                         template='plotly_dark', title='Top 5 Switches by Alert Count')
            st.plotly_chart(fig, use_container_width=True)

        st.divider()
        st.markdown("#### Q2 — Most common alert messages")
        top_msg = sw['device_error'].value_counts().reset_index()
        top_msg.columns = ['Error Message', 'Count']
        st.dataframe(top_msg, use_container_width=True)

        st.divider()
        st.markdown("#### Q3 — Problem-free switches")
        alerted = set(sw['inv_code'].unique())
        pf = sw_inv[~sw_inv['inv_code'].isin(alerted)][['inv_code', 'device_type']].reset_index(drop=True)
        st.metric("Problem-free switches", f"{len(pf)} of {len(sw_inv)}")
        if len(pf) > 0:
            with st.expander("View problem-free switches"):
                pf.columns = ['Inv Code', 'Type']
                st.dataframe(pf, use_container_width=True)
        else:
            st.warning("All switches have at least one recorded alert.")

        st.divider()
        st.markdown("#### Q4 — Crunch time: 95th percentile for switch alerts")
        sw_hourly = sw.groupby('hour').size().reset_index(name='Alerts')
        sw_hourly.columns = ['Hour', 'Alerts']
        sw_p95 = np.percentile(sw_hourly['Alerts'], 95)
        fig2 = px.bar(sw_hourly, x='Hour', y='Alerts', template='plotly_dark',
                      color='Alerts', color_continuous_scale='Blues',
                      title='Switch Alerts by Hour of Day')
        fig2.add_hline(y=sw_p95, line_dash='dash', line_color='red',
                       annotation_text=f'95th pct ({sw_p95:.0f})')
        st.plotly_chart(fig2, use_container_width=True)
        sw_crunch = sorted(sw_hourly[sw_hourly['Alerts'] >= sw_p95]['Hour'].tolist())
        st.info(f"⏰ Switch crunch hours: **{', '.join(f'{h:02d}:00' for h in sw_crunch)}**")

    # ── FIREWALL ──────────────────────────────────────────────────────────────
    elif section == "🔥 Firewall":
        st.subheader("Firewall")
        fw = tickets[tickets['device_type'] == 'Firewall']
        st.info("ℹ️ Only 2 firewalls in this dataset — volume ranking is less meaningful; focus is on error patterns and timing.")

        st.markdown("#### Q1 — Most common errors")
        top_err = fw['device_error'].value_counts().reset_index()
        top_err.columns = ['Error', 'Count']
        c1, c2 = st.columns([1, 2])
        with c1:
            st.dataframe(top_err, use_container_width=True)
        with c2:
            fig = px.bar(top_err, x='Error', y='Count', template='plotly_dark',
                         color='Count', color_continuous_scale='Reds',
                         title='Firewall Error Frequency')
            st.plotly_chart(fig, use_container_width=True)

        st.divider()
        st.markdown("#### Q2 — Most problematic firewalls")
        fw_rank = fw.groupby('inv_code').size().reset_index(name='Alert Count').sort_values('Alert Count', ascending=False)
        fw_rank.columns = ['Inv Code', 'Alert Count']
        st.dataframe(fw_rank, use_container_width=True)

        st.divider()
        st.markdown("#### Q3 — Correlation with time of day & day of week")
        ca, cb = st.columns(2)
        with ca:
            by_h = fw.groupby('hour').size().reset_index(name='Alerts')
            by_h.columns = ['Hour', 'Alerts']
            fig2 = px.bar(by_h, x='Hour', y='Alerts', template='plotly_dark',
                          color='Alerts', color_continuous_scale='Oranges',
                          title='Firewall Alerts by Hour of Day')
            st.plotly_chart(fig2, use_container_width=True)
            peak_h = int(by_h.loc[by_h['Alerts'].idxmax(), 'Hour'])
            st.caption(f"Peak hour: **{peak_h:02d}:00**")
        with cb:
            day_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
            by_d = fw.groupby('day_of_week').size().reset_index(name='Alerts')
            by_d.columns = ['Day', 'Alerts']
            by_d['Day'] = pd.Categorical(by_d['Day'], categories=day_order, ordered=True)
            by_d = by_d.sort_values('Day')
            fig3 = px.bar(by_d, x='Day', y='Alerts', template='plotly_dark',
                          color='Alerts', color_continuous_scale='Oranges',
                          title='Firewall Alerts by Day of Week')
            st.plotly_chart(fig3, use_container_width=True)
            peak_d = by_d.loc[by_d['Alerts'].idxmax(), 'Day']
            st.caption(f"Peak day: **{peak_d}**")
        st.info(
            f"📊 Total firewall alerts: **{len(fw)}**. "
            f"Peak activity at **{peak_h:02d}:00** and on **{peak_d}s**. "
            "Given the small sample (2 firewalls), treat timing as directional."
        )

    # ── SERVERS ───────────────────────────────────────────────────────────────
    elif section == "🖥️  Servers":
        st.subheader("Servers")
        srv = tickets[tickets['device_type'].isin(_SERVER_TYPES)]
        srv_inv = inventory[inventory['device_type'].isin(_SERVER_TYPES)]

        st.markdown("#### Q1 — Most common alert messages")
        top_err = srv['device_error'].value_counts().reset_index().head(10)
        top_err.columns = ['Error', 'Count']
        c1, c2 = st.columns([1, 2])
        with c1:
            st.dataframe(top_err, use_container_width=True)
        with c2:
            fig = px.bar(top_err, x='Error', y='Count', template='plotly_dark',
                         color='Count', color_continuous_scale='Blues',
                         title='Top 10 Server Alert Messages')
            st.plotly_chart(fig, use_container_width=True)

        st.divider()
        st.markdown("#### Q2 — OS type with relatively higher alert volume")
        by_os = srv.groupby('device_type').size().reset_index(name='Alert Count').sort_values('Alert Count', ascending=False)
        by_os.columns = ['OS Type', 'Alert Count']
        st.dataframe(by_os, use_container_width=True)
        top_os = by_os.iloc[0]
        st.success(
            f"**{top_os['OS Type']}** generates more alerts ({top_os['Alert Count']:,}) "
            f"than {by_os.iloc[1]['OS Type']} ({by_os.iloc[1]['Alert Count']:,})."
        )

        st.divider()
        st.markdown("#### Q3 — Correlation of alerts with installation age (Serial No proxy)")
        srv_m = srv_inv.copy()
        srv_m['serial_no'] = pd.to_numeric(srv_m['serial_no'], errors='coerce')
        tc = srv.groupby('inv_code').size().reset_index(name='ticket_count')
        srv_m = srv_m.merge(tc, on='inv_code', how='left').fillna({'ticket_count': 0})
        corr = srv_m[['serial_no','ticket_count']].dropna().corr().iloc[0,1]
        st.metric("Pearson r  (Serial No vs Alert Count)", f"{corr:.3f}")
        if abs(corr) < 0.1:
            verdict = "No meaningful correlation — installation age does not predict alert frequency."
        elif corr > 0:
            verdict = f"Weak positive (r={corr:.2f}): slightly newer devices tend to generate more alerts."
        else:
            verdict = f"Weak negative (r={corr:.2f}): older devices generate slightly more alerts."
        st.info(f"📊 {verdict}")
        fig2 = px.scatter(srv_m.dropna(subset=['serial_no']),
                          x='serial_no', y='ticket_count', color='device_type',
                          template='plotly_dark',
                          title='Serial No. vs Alert Count (age proxy)',
                          labels={'serial_no': 'Serial No (lower = older)', 'ticket_count': 'Alerts'},
                          opacity=0.5)
        st.plotly_chart(fig2, use_container_width=True)

        st.divider()
        st.markdown("#### Q4 — Error-free servers")
        alerted = set(srv['inv_code'].unique())
        ef = srv_inv[~srv_inv['inv_code'].isin(alerted)][['inv_code','device_type']].reset_index(drop=True)
        st.metric("Error-free servers", f"{len(ef):,} of {len(srv_inv):,}")
        if len(ef) > 0:
            with st.expander(f"View all {len(ef)} error-free servers"):
                ef.columns = ['Inv Code', 'OS Type']
                st.dataframe(ef, use_container_width=True)

        st.divider()
        st.markdown("#### Q5 — Top 10 most problematic servers")
        top10 = (
            srv.groupby(['inv_code','device_type']).size()
            .reset_index(name='Alert Count')
            .sort_values('Alert Count', ascending=False)
            .head(10).reset_index(drop=True)
        )
        top10.columns = ['Inv Code', 'OS Type', 'Alert Count']
        st.dataframe(top10, use_container_width=True)
        fig3 = px.bar(top10, x='Inv Code', y='Alert Count', color='OS Type',
                      template='plotly_dark', title='Top 10 Servers by Alert Count')
        st.plotly_chart(fig3, use_container_width=True)

    # ── DATABASE ──────────────────────────────────────────────────────────────
    elif section == "🗄️  Database":
        st.subheader("Database")
        db = tickets[tickets['device_type'] == 'Database'].copy()
        db_with_resp = db.dropna(subset=['query_resp_ms'])

        st.markdown("#### Q1 — Top 5 common messages")
        top5 = db['device_error'].value_counts().reset_index().head(5)
        top5.columns = ['Error Message', 'Count']
        c1, c2 = st.columns([1, 2])
        with c1:
            st.dataframe(top5, use_container_width=True)
        with c2:
            fig = px.bar(top5, x='Error Message', y='Count', template='plotly_dark',
                         color='Count', color_continuous_scale='Purples',
                         title='Top 5 DB Error Messages')
            st.plotly_chart(fig, use_container_width=True)

        st.divider()
        st.markdown("#### Q2 — Correlation: Table Fragmentation vs Query Response Time")
        st.caption(
            "The dataset does not contain a direct CPU load column. "
            "**Table Fragmentation %** (parsed from the `table_frag` column) is the closest proxy for database load. "
            "A high correlation with Query Response ms means fragmentation directly impacts performance."
        )
        if len(db_with_resp) > 2:
            corr_val = db_with_resp[['query_resp_ms','table_frag_numeric']].dropna().corr().iloc[0,1]
            st.metric("Pearson r  (Table Frag % vs Query Resp ms)", f"{corr_val:.3f}")
            st.success(
                f"**Very strong positive correlation (r = {corr_val:.3f})**: "
                "Higher table fragmentation is tightly associated with slower query response times. "
                "Regular index REBUILD / REORGANIZE would directly reduce query latency."
            )

            # Query response distribution
            fig2 = px.histogram(db_with_resp, x='query_resp_ms', nbins=20,
                                template='plotly_dark',
                                title='Distribution of Query Response Times (ms)',
                                labels={'query_resp_ms': 'Response Time (ms)'},
                                color_discrete_sequence=['#9b59b6'])
            st.plotly_chart(fig2, use_container_width=True)

            # Scatter with trendline
            try:
                fig3 = px.scatter(
                    db_with_resp.dropna(subset=['table_frag_numeric']),
                    x='table_frag_numeric', y='query_resp_ms',
                    template='plotly_dark',
                    title='Table Fragmentation % vs Query Response Time (ms)',
                    labels={'table_frag_numeric': 'Table Fragmentation (%)',
                            'query_resp_ms': 'Query Response (ms)'},
                    color='query_resp_ms',
                    color_continuous_scale='Purples',
                    opacity=0.5,
                    trendline='ols'
                )
            except Exception:
                fig3 = px.scatter(
                    db_with_resp.dropna(subset=['table_frag_numeric']),
                    x='table_frag_numeric', y='query_resp_ms',
                    template='plotly_dark',
                    title='Table Fragmentation % vs Query Response Time (ms)',
                    labels={'table_frag_numeric': 'Table Fragmentation (%)',
                            'query_resp_ms': 'Query Response (ms)'},
                    color='query_resp_ms',
                    color_continuous_scale='Purples',
                    opacity=0.5
                )
            st.plotly_chart(fig3, use_container_width=True)

            # Stats table
            stats_df = pd.DataFrame({
                'Metric': ['Query Resp (ms)', 'Table Frag (%)'],
                'Min': [db_with_resp['query_resp_ms'].min(), db_with_resp['table_frag_numeric'].min()],
                'Mean': [db_with_resp['query_resp_ms'].mean().round(1), db_with_resp['table_frag_numeric'].mean().round(1)],
                'Max': [db_with_resp['query_resp_ms'].max(), db_with_resp['table_frag_numeric'].max()],
            })
            st.dataframe(stats_df, use_container_width=True)

        st.divider()
        st.markdown("#### Q3 — Top 5 database instances with most alerts")
        top5_inst = (
            db.groupby('inv_code').size()
            .reset_index(name='Alert Count')
            .sort_values('Alert Count', ascending=False)
            .head(5).reset_index(drop=True)
        )
        top5_inst.columns = ['DB Instance', 'Alert Count']
        c1, c2 = st.columns([1, 2])
        with c1:
            st.dataframe(top5_inst, use_container_width=True)
        with c2:
            fig4 = px.bar(top5_inst, x='DB Instance', y='Alert Count',
                          template='plotly_dark', color='Alert Count',
                          color_continuous_scale='Purples',
                          title='Top 5 DB Instances by Alert Count')
            st.plotly_chart(fig4, use_container_width=True)

    # ── UPS ───────────────────────────────────────────────────────────────────
    elif section == "🔋 UPS":
        st.subheader("UPS")
        ups = tickets[tickets['device_type'] == 'UPS']

        st.markdown("#### Q1 — UPS units with most alert messages")
        top_ups = (
            ups.groupby('inv_code').size()
            .reset_index(name='Alert Count')
            .sort_values('Alert Count', ascending=False)
            .reset_index(drop=True)
        )
        top_ups.columns = ['UPS Unit', 'Alert Count']
        c1, c2 = st.columns([1, 2])
        with c1:
            st.dataframe(top_ups, use_container_width=True)
        with c2:
            fig = px.bar(top_ups.head(15), x='UPS Unit', y='Alert Count',
                         template='plotly_dark', color='Alert Count',
                         color_continuous_scale='Greens',
                         title='UPS Units by Alert Count')
            st.plotly_chart(fig, use_container_width=True)
        st.success(
            f"Most alerting UPS: **{top_ups.iloc[0]['UPS Unit']}** "
            f"with **{top_ups.iloc[0]['Alert Count']}** alerts."
        )

        st.divider()
        st.markdown("#### Q2 — Top 3 most frequent messages")
        top3 = ups['device_error'].value_counts().reset_index().head(3)
        top3.columns = ['Message', 'Count']
        ca, cb = st.columns([1, 2])
        with ca:
            st.dataframe(top3, use_container_width=True)
        with cb:
            fig2 = px.pie(top3, names='Message', values='Count',
                          template='plotly_dark', title='Top 3 UPS Alert Messages',
                          color_discrete_sequence=px.colors.sequential.Greens_r)
            st.plotly_chart(fig2, use_container_width=True)
        st.info(
            f"🔋 The dominant UPS alert is **'{top3.iloc[0]['Message']}'** "
            f"({top3.iloc[0]['Count']} occurrences), indicating battery life degradation "
            "is the primary UPS health concern."
        )


# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    st.markdown('<div class="main-header">🤖 AI Intelligence Platform V3</div>', unsafe_allow_html=True)
    st.markdown("*Combined Server Health Analysis & Inventory Intelligence*")

    with st.sidebar:
        st.title("🚀 Configuration")

        st.subheader("API Keys")
        openai_key  = st.text_input("OpenAI API Key",  type="password", value=os.getenv("OPENAI_API_KEY", ""))
        gemini_key  = st.text_input("Google API Key",  type="password", value=os.getenv("GOOGLE_API_KEY", ""))
        mistral_key = st.text_input("Mistral API Key", type="password", value=os.getenv("MISTRAL_API_KEY", ""))

        st.divider()
        st.subheader("Database Configuration")
        mongo_uri = st.text_input("MongoDB URI", type="password", value=os.getenv("MONGODB_URI", ""))
        if mongo_uri:
            os.environ["MONGODB_URI"] = mongo_uri
            client = get_mongodb_client()
            if client:
                st.success("✅ MongoDB Connected")
            else:
                st.error("❌ MongoDB Connection Failed")
        else:
            st.warning("⚠️ MongoDB URI not configured. Queries will not be saved.")

        st.divider()
        st.subheader("Module Selection")
        selected_module = st.radio(
            "Choose Analysis Module:",
            ["Server Health Analysis (V1)", "Inventory Intelligence (V2)", "Unified Dashboard"]
        )

    # =========================================================================
    # MODULE 1: SERVER HEALTH
    # =========================================================================
    if selected_module == "Server Health Analysis (V1)":
        st.markdown('<div class="section-header">📊 Server Health Analysis</div>', unsafe_allow_html=True)

        hc1, hc2 = st.columns(2)
        with hc1:
            use_history = st.checkbox("🔗 Include historical database records?", value=False)
        with hc2:
            history_limit = st.number_input("Historical entries to include", 1, 50, 5, 1,
                                            disabled=not use_history)

        server_data = load_server_data()
        if server_data is None or server_data.empty:
            st.warning("⚠️ Server data files not found. Please ensure `historicdataCPU.csv` and `historicdataDISK.csv` are present.")
            return

        tab1, tab2 = st.tabs(["🔍 New Analysis", "📜 Query History"])

        with tab1:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("📅 Time Range")
                start_date = st.date_input("Start Date", datetime.now().date())
                start_time = st.time_input("Start Time", datetime.now().time())
                end_date   = st.date_input("End Date",   datetime.now().date())
                end_time   = st.time_input("End Time",   datetime.now().time())
            with col2:
                st.subheader("🔧 AI Configuration")
                ai_provider = st.selectbox("AI Provider", ["OpenAI", "Google Gemini", "Mistral AI"])
                if ai_provider == "OpenAI":
                    model = st.selectbox("Model", ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"])
                elif ai_provider == "Google Gemini":
                    model = st.selectbox("Model", ["gemini-2.0-flash-exp","gemini-1.5-pro","gemini-1.5-flash","gemini-1.5-flash-8b"])
                else:
                    model = st.selectbox("Model", ["mistral-large-latest","mistral-small-latest","open-mistral-7b","open-mixtral-8x7b"])
                temperature = st.slider("Temperature", 0.0, 1.0, 0.3, 0.1)
                max_tokens  = st.number_input("Max Tokens", 100, 8000, 1000, 100)

            st.subheader("💭 Prompt")
            context     = st.text_area("System Context", "Windows Server running SQL Server.", height=80)
            user_prompt = st.text_area("Your Question",  "Analyze for critical issues and provide top 3 recommendations.", height=100)

            if st.button("🔍 Analyze", type="primary", use_container_width=True):
                db_context_data = ""
                kb_context_data = ""
                if use_history:
                    with st.spinner("Fetching historical entries..."):
                        db_context_data = get_context_from_db('server_health', limit=history_limit)
                    st.info(f"📚 Using last **{history_limit}** historical entries.")
                    with st.spinner("Searching knowledge base..."):
                        kb_context_data, detected_cat, kb_count = enhance_prompt_with_kb(user_prompt, 'server_health')
                    if kb_count > 0:
                        st.info(f"📖 {kb_count} KB entries found" + (f" ({detected_cat})" if detected_cat else ""))

                key_map = {"OpenAI": openai_key, "Google Gemini": gemini_key, "Mistral AI": mistral_key}
                api_key = key_map[ai_provider]
                if not api_key:
                    st.error(f"Please provide a {ai_provider} API key in the sidebar.")
                    return

                with st.spinner("Analyzing..."):
                    start_dt = datetime.combine(start_date, start_time)
                    end_dt   = datetime.combine(end_date,   end_time)
                    filtered = server_data[
                        (server_data['interval_start_dt'] >= start_dt) &
                        (server_data['interval_start_dt'] <= end_dt)
                    ].copy()
                    if filtered.empty:
                        st.warning("No data found in the selected time range.")
                        return
                    snapshot = format_server_data_for_llm(filtered, start_dt, end_dt)
                    st.subheader("📊 Data Summary")
                    st.text(snapshot)
                    st.subheader("🤖 AI Analysis")
                    result = analyze_with_ai(snapshot, context, user_prompt,
                                             ai_provider, model, temperature, max_tokens, api_key,
                                             db_context=db_context_data, kb_context=kb_context_data)
                    st.markdown(result)
                    saved_id = save_query_to_mongodb('server_health', {
                        'time_range': {'start': start_dt, 'end': end_dt},
                        'ai_config': {'provider': ai_provider, 'model': model,
                                      'temperature': temperature, 'max_tokens': max_tokens},
                        'context': context, 'user_prompt': user_prompt,
                        'data_summary': snapshot, 'ai_response': result,
                        'data_points_analyzed': len(filtered),
                        'history_entries_used': history_limit if use_history else 0
                    })
                    if saved_id:
                        st.success(f"✅ Saved to database: {saved_id}")

        with tab2:
            st.subheader("📜 Recent Query History")
            history = get_query_history('server_health', limit=20)
            if not history:
                st.info("No history yet. Run some analyses first.")
            else:
                for query in history:
                    fmt = format_query_for_display(query)
                    with st.expander(f"🕐 {fmt['Timestamp']} — {fmt['Provider']} ({fmt['Model']})"):
                        ca, cb = st.columns(2)
                        with ca:
                            cfg = query.get('ai_config', {})
                            st.write(f"**Provider:** {cfg.get('provider','N/A')}  |  **Model:** {cfg.get('model','N/A')}")
                            st.write(f"**Temperature:** {cfg.get('temperature','N/A')}  |  **Max Tokens:** {cfg.get('max_tokens','N/A')}")
                        with cb:
                            tr = query.get('time_range', {})
                            st.write(f"**Start:** {tr.get('start','N/A')}  |  **End:** {tr.get('end','N/A')}")
                            st.write(f"**Data Points:** {query.get('data_points_analyzed','N/A')}  |  **History Used:** {query.get('history_entries_used','N/A')}")
                        st.write("**Prompt:**")
                        st.text(query.get('user_prompt', 'N/A'))
                        st.write("**AI Response:**")
                        st.markdown(query.get('ai_response', 'N/A'))

    # =========================================================================
    # MODULE 2: INVENTORY INTELLIGENCE
    # =========================================================================
    elif selected_module == "Inventory Intelligence (V2)":
        st.markdown('<div class="section-header">📦 Inventory Intelligence Hub</div>', unsafe_allow_html=True)

        hc1, hc2 = st.columns(2)
        with hc1:
            use_history_inv = st.checkbox("📚 Include previous DB queries for context?", value=False)
        with hc2:
            history_limit_inv = st.number_input("Historical entries to include", 1, 50, 5, 1,
                                                key="inv_history_limit", disabled=not use_history_inv)

        inventory, tickets, merged = load_inventory_data()
        if inventory is None:
            st.warning("⚠️ Could not load inventory data. Ensure `testdata.xlsx` is present.")
            return

        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Visual Dashboard",
            "📋 Structured Analytics",
            "🤖 AI Data Assistant",
            "📜 Query History",
            "📚 Knowledge Base",
        ])

        # ── TAB 1: Visual Dashboard ──────────────────────────────────────────
        with tab1:
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Inventory Items", len(inventory))
            c2.metric("Total Alert Tickets",   len(tickets))
            c3.metric("Device Types",          tickets['device_type'].nunique())

            ca, cb = st.columns(2)
            with ca:
                st.subheader("Alerts by Device Type")
                tpd = tickets.groupby('device_type').size().reset_index(name='count')
                fig1 = px.bar(tpd, x='device_type', y='count', template='plotly_dark',
                              color='count', color_continuous_scale='blues',
                              labels={'count': 'Alerts', 'device_type': 'Device Type'})
                st.plotly_chart(fig1, use_container_width=True)
            with cb:
                st.subheader("Alert Trends Over Time")
                trend = tickets.groupby('month').size().reset_index(name='count')
                fig2 = px.line(trend, x='month', y='count', markers=True, template='plotly_dark',
                               labels={'count': 'Alerts', 'month': 'Month'})
                st.plotly_chart(fig2, use_container_width=True)

            st.subheader("Top 10 Most Alerted Items")
            top10 = merged.nlargest(10, 'ticket_count')[['inv_code','ticket_count','device_type']]
            fig3 = px.bar(top10, x='inv_code', y='ticket_count', color='device_type',
                          template='plotly_dark',
                          labels={'ticket_count': 'Alerts', 'inv_code': 'Asset Code'})
            st.plotly_chart(fig3, use_container_width=True)

        # ── TAB 2: Structured Analytics ──────────────────────────────────────
        with tab2:
            render_structured_analytics(inventory, tickets, merged)

        # ── TAB 3: AI Data Assistant ─────────────────────────────────────────
        with tab3:
            st.subheader("💬 Chat with your Data")
            st.caption(
                "The AI receives a full pre-computed summary of the dataset (alerts by type, "
                "top items, error patterns, correlations, crunch hours) so it can answer "
                "accurately without guessing."
            )

            col1, col2 = st.columns(2)
            with col1:
                ai_provider = st.selectbox("AI Provider", ["OpenAI","Google Gemini","Mistral AI"], key="inv_provider")
                if ai_provider == "OpenAI":
                    model = st.selectbox("Model", ["gpt-4o","gpt-4o-mini","gpt-4-turbo","gpt-3.5-turbo"], key="inv_model")
                elif ai_provider == "Google Gemini":
                    model = st.selectbox("Model", ["gemini-2.0-flash-exp","gemini-1.5-pro","gemini-1.5-flash","gemini-1.5-flash-8b"], key="inv_model")
                else:
                    model = st.selectbox("Model", ["mistral-large-latest","mistral-small-latest","open-mistral-7b"], key="inv_model")
            with col2:
                temperature = st.slider("Temperature", 0.0, 1.0, 0.3, 0.1, key="inv_temp")
                max_tokens  = st.number_input("Max Tokens", 100, 8000, 2000, 100, key="inv_tokens")

            key_map = {"OpenAI": openai_key, "Google Gemini": gemini_key, "Mistral AI": mistral_key}
            api_key = key_map[ai_provider]

            if not api_key:
                st.warning(f"Please enter a {ai_provider} API Key in the sidebar.")
            else:
                user_input = st.chat_input("Ask a question about the inventory or alerts...")

                if user_input:
                    with st.chat_message("user"):
                        st.write(user_input)

                    with st.chat_message("assistant"):
                        with st.spinner("Analyzing..."):

                            # Build the rich pre-computed data summary
                            data_summary = build_inventory_summary(inventory, tickets, merged)

                            # Optional: fetch DB history context
                            db_ctx = ""
                            if use_history_inv:
                                db_ctx = get_context_from_db('inventory', limit=history_limit_inv)

                            # Optional: KB context
                            kb_ctx, _, _ = enhance_prompt_with_kb(user_input, 'inventory')

                            # ── SYSTEM PROMPT ────────────────────────────────
                            system_prompt = f"""You are a senior IT infrastructure analyst with full access to the pre-computed analytics summary below.
This summary was computed DIRECTLY from the dataset — all numbers are exact and accurate.
Use ONLY this summary to answer questions. Do not guess or fabricate numbers.

{data_summary}

{db_ctx}
{kb_ctx}

INSTRUCTIONS:
- Answer using the exact figures provided in the summary above.
- For questions about specific errors, use the parsed 'device_error' values listed.
- For database questions, use the query_resp_ms and table_frag_numeric correlations provided.
- If the answer is in the summary, state it directly and confidently.
- If a question asks for something NOT in the summary, say so honestly.
- Format your answer clearly with bullet points or tables where appropriate.
"""

                            try:
                                if ai_provider == "OpenAI":
                                    client = OpenAI(api_key=api_key)
                                    response = client.chat.completions.create(
                                        model=model,
                                        messages=[
                                            {"role": "system", "content": system_prompt},
                                            {"role": "user",   "content": user_input}
                                        ],
                                        temperature=temperature,
                                        max_tokens=max_tokens
                                    )
                                    full_text = response.choices[0].message.content

                                elif ai_provider == "Google Gemini":
                                    genai.configure(api_key=api_key)
                                    gen_model = genai.GenerativeModel(model)
                                    gen_cfg = genai.types.GenerationConfig(
                                        temperature=temperature, max_output_tokens=max_tokens
                                    )
                                    response = gen_model.generate_content(
                                        f"{system_prompt}\n\nUser question: {user_input}",
                                        generation_config=gen_cfg
                                    )
                                    full_text = response.text

                                else:
                                    client = Mistral(api_key=api_key)
                                    response = client.chat.complete(
                                        model=model,
                                        messages=[
                                            {"role": "system", "content": system_prompt},
                                            {"role": "user",   "content": user_input}
                                        ],
                                        temperature=temperature,
                                        max_tokens=max_tokens
                                    )
                                    full_text = response.choices[0].message.content

                                st.markdown(full_text)

                                # Save to MongoDB
                                save_query_to_mongodb('inventory', {
                                    'ai_config': {'provider': ai_provider, 'model': model,
                                                  'temperature': temperature, 'max_tokens': max_tokens},
                                    'user_prompt': user_input,
                                    'ai_response': full_text,
                                    'history_entries_used': history_limit_inv if use_history_inv else 0
                                })

                            except Exception as e:
                                st.error(f"Error: {e}")

        # ── TAB 4: Query History ─────────────────────────────────────────────
        with tab4:
            st.subheader("📜 Recent Query History")
            history = get_query_history('inventory', limit=20)
            if not history:
                st.info("No history yet. Start chatting to build history.")
            else:
                for query in history:
                    fmt = format_query_for_display(query)
                    with st.expander(f"🕐 {fmt['Timestamp']} — {fmt['Provider']} ({fmt['Model']})"):
                        cfg = query.get('ai_config', {})
                        st.write(f"**Provider:** {cfg.get('provider','N/A')}  |  **Model:** {cfg.get('model','N/A')}")
                        st.write(f"**History Used:** {query.get('history_entries_used','N/A')}")
                        st.write("**Prompt:**")
                        st.text(query.get('user_prompt','N/A'))
                        st.write("**AI Response:**")
                        st.markdown(query.get('ai_response','N/A'))

        # ── TAB 5: Knowledge Base ────────────────────────────────────────────
        with tab5:
            st.subheader("📚 Knowledge Base Repository")
            st.markdown("Pre-generated answers the AI references automatically when answering questions.")

            col1, col2 = st.columns([1, 3])
            with col1:
                categories = get_all_kb_categories()
                if categories:
                    selected_cat = st.selectbox("Filter by Category", ["All"] + categories)
                else:
                    st.warning("⚠️ Knowledge base empty.")
                    selected_cat = "All"
            with col2:
                search_term = st.text_input("🔍 Search knowledge base", "")

            if categories:
                if search_term:
                    results = search_knowledge_base(
                        search_term,
                        category=None if selected_cat == "All" else selected_cat,
                        limit=20
                    )
                    st.info(f"Found {len(results)} matching entries")
                else:
                    results = []
                    cats = categories if selected_cat == "All" else [selected_cat]
                    for cat in cats:
                        results.extend(get_kb_by_category(cat))

                for entry in results:
                    with st.expander(f"📌 {entry['category']}: {entry['question']}"):
                        st.markdown(f"**Category:** {entry['category']}")
                        st.markdown(f"**Q:** {entry['question']}")
                        st.markdown("**A:**")
                        st.markdown(entry['answer'])
                        st.divider()
                        ca, cb = st.columns(2)
                        ca.caption(f"Generated: {entry.get('timestamp','N/A')}")
                        cb.caption(f"Model: {entry.get('ai_provider','N/A')} — {entry.get('ai_model','N/A')}")

    # =========================================================================
    # MODULE 3: UNIFIED DASHBOARD
    # =========================================================================
    else:
        st.markdown('<div class="section-header">🎯 Unified Intelligence Dashboard</div>', unsafe_allow_html=True)

        server_data = load_server_data()
        inventory, tickets, merged = load_inventory_data()

        st.subheader("📊 System Overview")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Server Data Points", len(server_data) if server_data is not None else "N/A")
        if server_data is not None and not server_data.empty:
            c2.metric("Latest CPU", f"{server_data.iloc[-1].get('cpu_total', 0):.1f}%")
        else:
            c2.metric("Latest CPU", "N/A")
        c3.metric("Total Assets",  len(inventory) if inventory is not None else "N/A")
        c4.metric("Total Tickets", len(tickets)   if tickets   is not None else "N/A")

        st.divider()

        if server_data is not None and not server_data.empty:
            st.subheader("🖥️ Server Health Metrics")
            ca, cb = st.columns(2)
            with ca:
                if 'cpu_total' in server_data.columns:
                    fig_cpu = px.line(server_data.tail(50), x='interval_start_dt', y='cpu_total',
                                      title='CPU Usage (Last 50 Points)', template='plotly_dark',
                                      labels={'cpu_total': 'CPU %', 'interval_start_dt': 'Time'})
                    fig_cpu.update_traces(line_color='#00d4ff')
                    st.plotly_chart(fig_cpu, use_container_width=True)
            with cb:
                if 'disk_total' in server_data.columns:
                    rd = server_data.tail(50).copy()
                    rd['disk_gb'] = rd['disk_total'] / (1024**3)
                    fig_disk = px.line(rd, x='interval_start_dt', y='disk_gb',
                                       title='Disk Free Space (Last 50 Points)', template='plotly_dark',
                                       labels={'disk_gb': 'Free (GB)', 'interval_start_dt': 'Time'})
                    fig_disk.update_traces(line_color='#58a6ff')
                    st.plotly_chart(fig_disk, use_container_width=True)

        st.divider()

        if inventory is not None and tickets is not None:
            st.subheader("📦 Inventory & Alert Analytics")
            ca, cb = st.columns(2)
            with ca:
                dt = tickets.groupby('device_type').size().reset_index(name='count')
                fig_pie = px.pie(dt, values='count', names='device_type',
                                 title='Alerts by Device Type', template='plotly_dark',
                                 color_discrete_sequence=px.colors.sequential.Blues_r)
                st.plotly_chart(fig_pie, use_container_width=True)
            with cb:
                top10 = merged.nlargest(10,'ticket_count')[['inv_code','ticket_count','device_type']]
                fig_top = px.bar(top10, x='inv_code', y='ticket_count', color='device_type',
                                 title='Top 10 Assets by Alert Count', template='plotly_dark',
                                 labels={'ticket_count': 'Alerts', 'inv_code': 'Asset'})
                st.plotly_chart(fig_top, use_container_width=True)

            st.subheader("📈 Monthly Alert Trends")
            mt = tickets.groupby('month').size().reset_index(name='ticket_count')
            fig_trend = px.area(mt, x='month', y='ticket_count',
                                title='Monthly Alert Volume', template='plotly_dark',
                                labels={'ticket_count': 'Alerts', 'month': 'Month'})
            fig_trend.update_traces(fill='tozeroy', line_color='#00d4ff')
            st.plotly_chart(fig_trend, use_container_width=True)

        st.divider()
        st.info("💡 Select a specific module from the sidebar for detailed AI-powered analysis.")


if __name__ == "__main__":
    main()