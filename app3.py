import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from openai import OpenAI
from mistralai.client import Mistral
import anthropic
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

# Custom cohesive CSS theme for a professional developer aesthetic
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    .main-header {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1.5rem 0 0.2rem 0;
        letter-spacing: -0.04em;
    }
    
    .sub-header {
        text-align: center;
        font-size: 1.1rem;
        color: #64748b;
        margin-bottom: 2rem;
        font-weight: 500;
    }
    
    .section-header {
        font-size: 1.6rem;
        font-weight: 700;
        color: #4f46e5;
        border-bottom: 2px solid #e2e8f0;
        padding-bottom: 0.5rem;
        margin-top: 1.5rem;
        margin-bottom: 1.5rem;
        letter-spacing: -0.02em;
    }
    
    /* Subtle theme overrides for cards */
    .metric-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.25rem;
        text-align: center;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);
    }
    
    @media (prefers-color-scheme: dark) {
        .metric-card {
            background-color: #1e293b;
            border-color: #334155;
        }
        .section-header {
            border-bottom-color: #334155;
            color: #818cf8;
        }
    }
    
    /* Styled code containers */
    div.stCodeBlock {
        border-radius: 8px;
        border: 1px solid #e2e8f0;
    }
    @media (prefers-color-scheme: dark) {
        div.stCodeBlock {
            border: 1px solid #334155;
        }
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
# DATA LOADING & PARSING
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
# CORE PARSING HELPERS
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
    m = re.search(r'Q\.Resp\s*=\s*([\d]+)\s*ms', str(device_error))
    return int(m.group(1)) if m else None

def _parse_cpu_load(device_error):
    m = re.search(r'CPU\s*Load\s*=\s*([\d]+)%', str(device_error))
    return int(m.group(1)) if m else None

def _parse_table_frag_numeric(table_frag_str):
    if str(table_frag_str) == 'N/A':
        return None
    m = re.search(r'([\d.]+)', str(table_frag_str))
    return float(m.group(1)) if m else None


@st.cache_data
def load_inventory_data(file_path: str = "testdata.xlsx"):
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
            tix_raw = pd.read_excel(file_path, sheet_name=NEW_SHEET, header=0)
            tix_raw.columns = [clean_column_name(c) for c in tix_raw.columns]

            required = {'ticket_id', 'timestamp', 'inv_code', 'device_type',
                        'inv_code_parsed', 'device_type_parsed', 'device_error', 'table_frag'}
            missing = required - set(tix_raw.columns)
            if missing:
                st.warning(f"⚠️ '{NEW_SHEET}' is missing columns: {missing}. Falling back to legacy parsing.")
                raise ValueError("Missing columns")

            tickets = tix_raw[list(required)].dropna(subset=['inv_code']).copy()
            tickets['inv_code']         = tickets['inv_code'].astype(str).str.strip()
            tickets['device_type']      = tickets['device_type'].astype(str).str.strip()
            tickets['inv_code_parsed']  = tickets['inv_code_parsed'].astype(str).str.strip()
            tickets['device_type_parsed'] = tickets['device_type_parsed'].astype(str).str.strip()
            tickets['device_error']     = tickets['device_error'].astype(str).str.strip()
            tickets['table_frag']       = tickets['table_frag'].astype(str).str.strip()

        else:
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

        # Numeric columns
        tickets['query_resp_ms']      = tickets['device_error'].apply(_parse_query_resp_ms)
        tickets['cpu_load_numeric']   = tickets['device_error'].apply(_parse_cpu_load)
        tickets['table_frag_numeric'] = tickets['table_frag'].apply(_parse_table_frag_numeric)

        # ── Merged ────────────────────────────────────────────────────────
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
    lines = []
    lines.append("=== INVENTORY & ALERT DATASET SUMMARY ===")
    lines.append(f"Total inventory items : {len(inventory):,}")
    lines.append(f"Total alert tickets   : {len(tickets):,}")
    if not tickets.empty:
        lines.append(f"Date range            : {tickets['timestamp'].min().date()} to {tickets['timestamp'].max().date()}")
    lines.append("")

    # ── Alert distribution by device type ──────────────────────────────────
    lines.append("--- ALERTS BY DEVICE TYPE ---")
    by_type = tickets.groupby('device_type').size().sort_values(ascending=False)
    for dtype, cnt in by_type.items():
        lines.append(f"  {dtype}: {cnt:,} alerts")
    lines.append("")

    # ── General ────────────────────────────────────────────────────────────
    lines.append("--- GENERAL ---")
    if not by_type.empty:
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
    if not hourly.empty:
        p95 = np.percentile(hourly, 95)
        crunch = sorted(hourly[hourly >= p95].index.tolist())
        lines.append(f"95th-pct crunch hours : {[f'{h:02d}:00' for h in crunch]}")
    lines.append("")

    # ── Monthly trend ───────────────────────────────────────────────────────
    monthly = tickets.groupby('month').size()
    if not monthly.empty:
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

    srv_merged = srv_inv.copy()
    srv_merged['serial_no'] = pd.to_numeric(srv_merged['serial_no'], errors='coerce')
    tc = srv.groupby('inv_code').size().reset_index(name='ticket_count')
    srv_merged = srv_merged.merge(tc, on='inv_code', how='left').fillna({'ticket_count': 0})
    age_corr = srv_merged[['serial_no', 'ticket_count']].dropna().corr().iloc[0, 1] if len(srv_merged) > 1 else 0
    lines.append(f"Correlation (serial_no vs alerts, age proxy): r={age_corr:.3f} "
                 f"({'no correlation' if abs(age_corr) < 0.1 else 'moderate correlation'})")
    lines.append("")

    # ── Database ─────────────────────────────────────────────────────────────
    db = tickets[tickets['device_type'] == 'Database']
    lines.append("--- DATABASE ---")
    lines.append(f"Total DB alerts: {len(db):,}")
    top5_db_err = db['device_error'].value_counts().head(5)
    lines.append("Top 5 DB error messages:")
    for err, cnt in top5_db_err.items():
        lines.append(f"  '{err}': {cnt}")
    
    db_valid_resp = db.dropna(subset=['query_resp_ms'])
    if len(db_valid_resp) > 2:
        # Check table frag correlation
        if 'table_frag_numeric' in db_valid_resp.columns:
            db_frag_corr = db_valid_resp[['query_resp_ms', 'table_frag_numeric']].dropna().corr().iloc[0, 1]
            lines.append(f"Correlation (Query Response ms vs Table Frag %): r={db_frag_corr:.3f} "
                         f"({'strong' if abs(db_frag_corr) > 0.7 else 'moderate'} correlation)")
        # Check cpu load correlation if parsed
        if 'cpu_load_numeric' in db_valid_resp.columns and db_valid_resp['cpu_load_numeric'].notna().sum() > 2:
            db_cpu_corr = db_valid_resp[['query_resp_ms', 'cpu_load_numeric']].dropna().corr().iloc[0, 1]
            lines.append(f"Correlation (Query Response ms vs CPU Load %): r={db_cpu_corr:.3f} "
                         f"({'strong' if abs(db_cpu_corr) > 0.7 else 'moderate'} correlation)")
            
        lines.append(f"  Query resp stats: min={db_valid_resp['query_resp_ms'].min():.0f} ms, "
                     f"avg={db_valid_resp['query_resp_ms'].mean():.0f} ms, max={db_valid_resp['query_resp_ms'].max():.0f} ms")
    top5_db_inst = db.groupby('inv_code').size().sort_values(ascending=False).head(5)
    lines.append("Top 5 DB instances with most alerts:")
    for inv_code, cnt in top5_db_inst.items():
        lines.append(f"  {inv_code}: {cnt}")
    lines.append("")

    # ── UPS ──────────────────────────────────────────────────────────────────
    ups = tickets[tickets['device_type'] == 'UPS']
    lines.append("--- UPS ---")
    lines.append(f"Total UPS alerts: {len(ups):,}")
    if len(ups) > 0:
        top_ups = ups.groupby('inv_code').size().sort_values(ascending=False)
        lines.append(f"UPS with most alerts: {top_ups.index[0]} ({top_ups.iloc[0]} alerts)")
    top3_ups_msg = ups['device_error'].value_counts().head(3)
    lines.append("Top 3 UPS messages:")
    for err, cnt in top3_ups_msg.items():
        lines.append(f"  '{err}': {cnt}")
    lines.append("")

    # ── Printers ─────────────────────────────────────────────────────────────
    pri = tickets[tickets['device_type'] == 'Printer']
    lines.append("--- PRINTERS ---")
    lines.append(f"Total printer alerts: {len(pri):,}")
    if len(pri) > 0:
        top5_pri = pri.groupby('inv_code').size().sort_values(ascending=False).head(5)
        lines.append("Top 5 printers with most alerts:")
        for inv_code, cnt in top5_pri.items():
            lines.append(f"  {inv_code}: {cnt}")
        top_pri_err = pri['device_error'].value_counts().head(3)
        lines.append("Top 3 printer messages:")
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
            f"You are an expert infrastructure analyst.\n"
            f"Current System Context: {context}\n\n"
            f"{db_context}\n{kb_context}\n"
            "Answering guidelines:\n"
            "1. Reference the data directly.\n"
            "2. Note past database query configurations if relevant.\n"
            "3. Be specific and clear."
        )
        user_msg = f"Data Snapshot:\n{data_snapshot}\n\nUser Request:\n{prompt}"

        if provider == "OpenAI":
            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": system_msg},
                           {"role": "user", "content": user_msg}],
                temperature=temperature, max_tokens=max_tokens
            )
            return response.choices[0].message.content

        elif provider == "Anthropic":
            client = anthropic.Anthropic(api_key=api_key)
            response = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_msg,
                messages=[{"role": "user", "content": user_msg}]
            )
            return response.content[0].text

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
# STRUCTURED ANALYTICS  (Pre-computed)
# ============================================================================

def render_structured_analytics(inventory, tickets, merged):
    st.markdown("### 📋 Direct Computed Analytics")
    st.caption("Exact values computed straight from source data.")

    section = st.selectbox(
        "Jump to section:",
        [
            "📈 Overview — Alert Patterns Over Time",
            "🗂️ General Overview",
            "🔀 Switches",
            "🔥 Firewall",
            "🖥️ Servers",
            "🗄️ Database",
            "🔋 UPS",
            "🖨️ Printers"
        ],
        key="analytics_section"
    )

    # ── OVERVIEW ──────────────────────────────────────────────────────────────
    if section == "📈 Overview — Alert Patterns Over Time":
        monthly = tickets.groupby('month').size().reset_index(name='count')
        first_v, last_v = monthly['count'].iloc[0], monthly['count'].iloc[-1]
        pct = (last_v - first_v) / first_v * 100 if first_v else 0
        trend = "increasing" if pct > 5 else ("decreasing" if pct < -5 else "stable")

        st.subheader("Alert trend")
        c1, c2 = st.columns([2, 1])
        with c1:
            fig = px.line(monthly, x='month', y='count', markers=True,
                          title="Monthly Alerts Timeline", template="plotly_dark")
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_family="Plus Jakarta Sans",
                margin=dict(l=20, r=20, t=40, b=20)
            )
            fig.update_traces(line=dict(color='#6366f1', width=3), marker=dict(size=8, color='#f43f5e'))
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            with st.container(border=True):
                st.metric("Total Tickets Analyzed", f"{len(tickets):,}")
                st.metric("Timeline Trend Status", trend.capitalize())
                st.info(f"Timeline spans **{monthly['month'].min()}** to **{monthly['month'].max()}**.")

    # ── GENERAL ───────────────────────────────────────────────────────────────
    elif section == "🗂️ General Overview":
        st.subheader("General Summary")
        by_type = merged.groupby('device_type')['ticket_count'].sum().sort_values(ascending=False)

        with st.container(border=True):
            st.markdown("#### Q1 — Type of inventory with most alerts")
            st.write(f"Winner: **{by_type.index[0]}** with {by_type.iloc[0]} alerts.")

        st.markdown("#### Q2 — Top 10 items with most alerts")
        top10 = merged.sort_values('ticket_count', ascending=False).head(10)[['inv_code', 'device_type', 'ticket_count']]
        st.dataframe(top10, use_container_width=True)

        st.markdown("#### Q3 — Inventory items with no alerts")
        no_alerts = merged[merged['ticket_count'] == 0][['inv_code', 'device_type']]
        st.metric("Total Items with No Alerts", len(no_alerts))
        with st.expander("View entire list"):
            st.dataframe(no_alerts, use_container_width=True)

        with st.container(border=True):
            st.markdown("#### Q4 — Crunch hours (95th percentile)")
            hourly = tickets.groupby('hour').size().reset_index(name='count')
            p95 = np.percentile(hourly['count'], 95)
            crunch = sorted(hourly[hourly['count'] >= p95]['hour'].tolist())
            st.write(f"95th percentile window hours: {', '.join(f'{h:02d}:00' for h in crunch)}")

    # ── SWITCHES ──────────────────────────────────────────────────────────────
    elif section == "🔀 Switches":
        sw = tickets[tickets['device_type'].isin(_SWITCH_TYPES)]
        sw_inv = inventory[inventory['device_type'].isin(_SWITCH_TYPES)]

        st.markdown("#### Q1 — Top 5 Switches")
        top5 = sw.groupby('inv_code').size().sort_values(ascending=False).head(5)
        st.dataframe(top5, use_container_width=True)

        st.markdown("#### Q2 — Most common alert message")
        st.dataframe(sw['device_error'].value_counts().head(1))

        with st.container(border=True):
            st.markdown("#### Q3 — Problem-free switches")
            alerted = set(sw['inv_code'].unique())
            pf = sw_inv[~sw_inv['inv_code'].isin(alerted)]
            st.write(f"Count: {len(pf)} out of {len(sw_inv)}")
            st.dataframe(pf, use_container_width=True)

        with st.container(border=True):
            st.markdown("#### Q4 — Switch Crunch Time (95th percentile)")
            sw_hourly = sw.groupby('hour').size().reset_index(name='count')
            if not sw_hourly.empty:
                p95 = np.percentile(sw_hourly['count'], 95)
                crunch = sorted(sw_hourly[sw_hourly['count'] >= p95]['hour'].tolist())
                st.write(f"Hours: {', '.join(f'{h:02d}:00' for h in crunch)}")

    # ── FIREWALL ──────────────────────────────────────────────────────────────
    elif section == "🔥 Firewall":
        fw = tickets[tickets['device_type'] == 'Firewall']
        st.markdown("#### Q1 — Most common error")
        st.write(fw['device_error'].value_counts().head(1))

        st.markdown("#### Q2 — Problematic Firewalls")
        st.dataframe(fw.groupby('inv_code').size().sort_values(ascending=False), use_container_width=True)

        with st.container(border=True):
            st.markdown("#### Q3 — Correlation with day & hour")
            h_peak = fw.groupby('hour').size().idxmax() if len(fw) > 0 else "N/A"
            d_peak = fw.groupby('day_of_week').size().idxmax() if len(fw) > 0 else "N/A"
            st.write(f"Peak Hour: {h_peak}:00. Peak Day of Week: {d_peak}.")

    # ── SERVERS ───────────────────────────────────────────────────────────────
    elif section == "🖥️ Servers":
        srv = tickets[tickets['device_type'].isin(_SERVER_TYPES)]
        srv_inv = inventory[inventory['device_type'].isin(_SERVER_TYPES)]

        st.markdown("#### Q1 — Common alert messages")
        st.dataframe(srv['device_error'].value_counts().head(10), use_container_width=True)

        st.markdown("#### Q2 — OS platform distribution")
        st.dataframe(srv.groupby('device_type').size(), use_container_width=True)

        with st.container(border=True):
            st.markdown("#### Q3 — Correlation with installation age")
            srv_m = srv_inv.copy()
            srv_m['serial_no'] = pd.to_numeric(srv_m['serial_no'], errors='coerce')
            tc = srv.groupby('inv_code').size().reset_index(name='cnt')
            srv_m = srv_m.merge(tc, on='inv_code', how='left').fillna(0)
            st.write(srv_m[['serial_no', 'cnt']].corr().iloc[0,1])

        st.markdown("#### Q4 — Error-free servers")
        alerted = set(srv['inv_code'].unique())
        st.write(f"Total error free: {len(srv_inv[~srv_inv['inv_code'].isin(alerted)])}")

    # ── DATABASE ──────────────────────────────────────────────────────────────
    elif section == "🗄️ Database":
        db = tickets[tickets['device_type'] == 'Database']
        st.markdown("#### Q1 — Top 5 common messages")
        st.dataframe(db['device_error'].value_counts().head(5), use_container_width=True)

        with st.container(border=True):
            st.markdown("#### Q2 — Correlations")
            db_valid = db.dropna(subset=['query_resp_ms'])
            if len(db_valid) > 2:
                if 'table_frag_numeric' in db_valid.columns:
                    corr = db_valid[['query_resp_ms', 'table_frag_numeric']].corr().iloc[0, 1]
                    st.write(f"Query Response vs Table Frag correlation: **{corr:.4f}**")
                if 'cpu_load_numeric' in db_valid.columns and db_valid['cpu_load_numeric'].notna().sum() > 2:
                    corr_cpu = db_valid[['query_resp_ms', 'cpu_load_numeric']].corr().iloc[0, 1]
                    st.write(f"Query Response vs CPU Load correlation: **{corr_cpu:.4f}**")

        st.markdown("#### Q3 — Top 5 Database Instances")
        st.dataframe(db.groupby('inv_code').size().sort_values(ascending=False).head(5), use_container_width=True)

    # ── UPS ───────────────────────────────────────────────────────────────────
    elif section == "🔋 UPS":
        ups = tickets[tickets['device_type'] == 'UPS']
        st.markdown("#### Q1 — Top UPS instances with most alerts")
        st.dataframe(ups.groupby('inv_code').size().sort_values(ascending=False).head(5), use_container_width=True)

        st.markdown("#### Q2 — Top 3 frequent messages")
        st.dataframe(ups['device_error'].value_counts().head(3), use_container_width=True)

    # ── PRINTERS ──────────────────────────────────────────────────────────────
    elif section == "🖨️ Printers":
        pri = tickets[tickets['device_type'] == 'Printer']
        st.markdown("#### Q1 — Top 5 printers with the most messages")
        st.dataframe(pri.groupby('inv_code').size().sort_values(ascending=False).head(5), use_container_width=True)

        st.markdown("#### Q2 — Top 3 Frequent Messages")
        st.dataframe(pri['device_error'].value_counts().head(3), use_container_width=True)


# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    st.markdown('<div class="main-header">🤖 AI Intelligence Platform V3</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Combined Server Health Analysis & Inventory Intelligence</div>', unsafe_allow_html=True)

    with st.sidebar:
        st.title("🚀 Configuration")

        with st.container(border=True):
            st.subheader("🔑 API Credentials")
            openai_key    = st.text_input("OpenAI API Key", type="password", value=os.getenv("OPENAI_API_KEY", ""))
            anthropic_key = st.text_input("Anthropic API Key", type="password", value=os.getenv("ANTHROPIC_API_KEY", ""))
            mistral_key   = st.text_input("Mistral API Key", type="password", value=os.getenv("MISTRAL_API_KEY", ""))

        with st.container(border=True):
            st.subheader("🌐 Database Connection")
            mongo_uri = st.text_input("MongoDB URI", type="password", value=os.getenv("MONGODB_URI", ""))
            if mongo_uri:
                os.environ["MONGODB_URI"] = mongo_uri
                client = get_mongodb_client()
                if client:
                    st.success("MongoDB Connected")
                else:
                    st.error("MongoDB Connection Failed")
            else:
                st.warning("MongoDB URI not configured.")

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

        with st.container(border=True):
            hc1, hc2 = st.columns(2)
            with hc1:
                use_history = st.checkbox("Include historical database records?", value=False)
            with hc2:
                history_limit = st.number_input("Historical entries to include", 1, 50, 5, 1, disabled=not use_history)

        server_data = load_server_data()
        if server_data is None or server_data.empty:
            st.warning("⚠️ Server data files not found. Ensure historicdataCPU.csv and historicdataDISK.csv are present.")
            return

        tab1, tab2 = st.tabs(["🔍 Analysis Workspace", "📜 Saved History"])

        with tab1:
            col1, col2 = st.columns(2)
            with col1:
                with st.container(border=True):
                    st.subheader("📅 Time Range Filter")
                    start_date = st.date_input("Start Date", datetime.now().date())
                    start_time = st.time_input("Start Time", datetime.now().time())
                    end_date   = st.date_input("End Date",   datetime.now().date())
                    end_time   = st.time_input("End Time",   datetime.now().time())
            with col2:
                with st.container(border=True):
                    st.subheader("🔧 AI Model Setup")
                    ai_provider = st.selectbox("AI Provider", ["OpenAI", "Anthropic", "Mistral AI"])
                    if ai_provider == "OpenAI":
                        model = st.selectbox("Model", ["gpt-4o", "gpt-4o-mini", "o1-mini", "o3-mini"])
                    elif ai_provider == "Anthropic":
                        model = st.selectbox("Model", ["claude-3-5-sonnet-latest", "claude-3-5-haiku-latest", "claude-3-opus-latest"])
                    else:
                        model = st.selectbox("Model", ["mistral-large-latest", "mistral-medium-latest", "mistral-small-latest"])
                    temperature = st.slider("Temperature", 0.0, 1.0, 0.3, 0.1)
                    max_tokens  = st.number_input("Max Tokens", 100, 8000, 1000, 100)

            st.subheader("💭 Prompt Configuration")
            context     = st.text_area("System Context", "Database workload operations environment.", height=80)
            user_prompt = st.text_area("Your Question",  "Analyze for critical trends and issue suggestions.", height=100)

            if st.button("🔍 Run Health Analysis", type="primary", use_container_width=True):
                db_context_data = ""
                kb_context_data = ""
                if use_history:
                    db_context_data = get_context_from_db('server_health', limit=history_limit)
                    kb_context_data, _, _ = enhance_prompt_with_kb(user_prompt, 'server_health')

                key_map = {"OpenAI": openai_key, "Anthropic": anthropic_key, "Mistral AI": mistral_key}
                api_key = key_map[ai_provider]
                if not api_key:
                    st.error(f"Please provide {ai_provider} API key.")
                    return

                with st.spinner("Analyzing server indicators..."):
                    start_dt = datetime.combine(start_date, start_time)
                    end_dt   = datetime.combine(end_date,   end_time)
                    filtered = server_data[
                        (server_data['interval_start_dt'] >= start_dt) &
                        (server_data['interval_start_dt'] <= end_dt)
                    ].copy()
                    if filtered.empty:
                        st.warning("No metrics found in range.")
                        return
                    snapshot = format_server_data_for_llm(filtered, start_dt, end_dt)
                    
                    st.subheader("📊 Data Summary Snapshot")
                    st.text(snapshot)
                    
                    st.subheader("🤖 AI Analysis Output")
                    result = analyze_with_ai(snapshot, context, user_prompt,
                                             ai_provider, model, temperature, max_tokens, api_key,
                                             db_context=db_context_data, kb_context=kb_context_data)
                    st.markdown(result)
                    save_query_to_mongodb('server_health', {
                        'time_range': {'start': start_dt, 'end': end_dt},
                        'ai_config': {'provider': ai_provider, 'model': model,
                                      'temperature': temperature, 'max_tokens': max_tokens},
                        'context': context, 'user_prompt': user_prompt,
                        'data_summary': snapshot, 'ai_response': result,
                        'data_points_analyzed': len(filtered),
                        'history_entries_used': history_limit if use_history else 0
                    })

        with tab2:
            st.subheader("📜 Saved Query History")
            history = get_query_history('server_health', limit=20)
            if not history:
                st.info("No queries logged.")
            else:
                for query in history:
                    fmt = format_query_for_display(query)
                    with st.expander(f"🕐 {fmt['Timestamp']} — {fmt['Provider']} ({fmt['Model']})"):
                        st.markdown(query.get('ai_response', 'N/A'))

    # =========================================================================
    # MODULE 2: INVENTORY INTELLIGENCE
    # =========================================================================
    elif selected_module == "Inventory Intelligence (V2)":
        st.markdown('<div class="section-header">📦 Inventory Intelligence Hub</div>', unsafe_allow_html=True)

        with st.container(border=True):
            hc1, hc2 = st.columns(2)
            with hc1:
                use_history_inv = st.checkbox("Include previous context records?", value=False)
            with hc2:
                history_limit_inv = st.number_input("Entries to look back", 1, 50, 5, 1, key="inv_history_limit", disabled=not use_history_inv)

        inventory, tickets, merged = load_inventory_data()
        if inventory is None:
            st.warning("⚠️ Inventory dataset files missing. Ensure testdata.xlsx is present.")
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
            st.markdown("### Operational Metrics")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Total Unique Assets", f"{len(inventory):,}")
            with c2:
                st.metric("Total Incident Tickets", f"{len(tickets):,}")
            with c3:
                st.metric("Covered Asset Families", tickets['device_type'].nunique())

            st.markdown("---")
            ca, cb = st.columns(2)
            with ca:
                with st.container(border=True):
                    st.subheader("Distribution by Type")
                    tpd = tickets.groupby('device_type').size().reset_index(name='count')
                    fig1 = px.bar(tpd, x='device_type', y='count', color='count', 
                                  color_continuous_scale='indigo', template='plotly_dark')
                    fig1.update_layout(
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font_family="Plus Jakarta Sans",
                        margin=dict(l=10, r=10, t=30, b=10)
                    )
                    st.plotly_chart(fig1, use_container_width=True)
            with cb:
                with st.container(border=True):
                    st.subheader("Monthly Incident Levels")
                    trend = tickets.groupby('month').size().reset_index(name='count')
                    fig2 = px.line(trend, x='month', y='count', markers=True, template='plotly_dark')
                    fig2.update_layout(
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font_family="Plus Jakarta Sans",
                        margin=dict(l=10, r=10, t=30, b=10)
                    )
                    fig2.update_traces(line=dict(color='#818cf8', width=3), marker=dict(size=8, color='#f43f5e'))
                    st.plotly_chart(fig2, use_container_width=True)

        # ── TAB 2: Structured Analytics ──────────────────────────────────────
        with tab2:
            render_structured_analytics(inventory, tickets, merged)

        # ── TAB 3: AI Data Assistant ─────────────────────────────────────────
        with tab3:
            st.subheader("💬 Chat Assistant")
            st.caption("AI generates exact responses grounded completely in data summaries.")

            with st.container(border=True):
                st.markdown("### 💡 Presentation Quick-Select Questions")
                preset_category = st.selectbox("Select exact scenario questions to run:", [
                    "Select a preset question...",
                    "General: Type of Inventory with most alerts",
                    "General: Top 10 inventory items with most alerts",
                    "General: Inventory items with no alerts",
                    "General: Crunch time interval (95th percentile alerts)",
                    "Switches: Top Five Switches with most alerts",
                    "Switches: Most common alert message",
                    "Switches: Problem free switches",
                    "Switches: Crunch time interval (95th percentile alerts)",
                    "Firewall: Most common error",
                    "Firewall: List most problematic Firewalls",
                    "Firewall: Correlation with time of day and day of week",
                    "Servers: Most common alert messages",
                    "Servers: Specific OS with relatively higher messages",
                    "Servers: Correlation of messages to installation age",
                    "Servers: Error-free servers",
                    "Servers: Top 10 most problematic servers",
                    "Database: Top five common messages",
                    "Database: Correlation between CPU load/Table fragmentation and Query response",
                    "Database: Top five database instances with the most messages",
                    "UPS: Top UPS with the most alert messages",
                    "UPS: Top three frequent messages",
                    "Printers: Top five printers with the most messages",
                    "Printers: Top three Frequent Messages",
                    "Overall Trend: Any alert message pattern over time indicating decrease, increase or stable volume"
                ])

            col1, col2 = st.columns(2)
            with col1:
                with st.container(border=True):
                    ai_provider = st.selectbox("AI Provider", ["OpenAI", "Anthropic", "Mistral AI"], key="inv_provider")
                    if ai_provider == "OpenAI":
                        model = st.selectbox("Model", ["gpt-4o", "gpt-4o-mini", "o1-mini", "o3-mini"], key="inv_model")
                    elif ai_provider == "Anthropic":
                        model = st.selectbox("Model", ["claude-3-5-sonnet-latest", "claude-3-5-haiku-latest", "claude-3-opus-latest"], key="inv_model")
                    else:
                        model = st.selectbox("Model", ["mistral-large-latest", "mistral-medium-latest", "mistral-small-latest"], key="inv_model")
            with col2:
                with st.container(border=True):
                    temperature = st.slider("Temperature", 0.0, 1.0, 0.1, 0.1, key="inv_temp")
                    max_tokens  = st.number_input("Max Tokens", 100, 8000, 2000, 100, key="inv_tokens")

            key_map = {"OpenAI": openai_key, "Anthropic": anthropic_key, "Mistral AI": mistral_key}
            api_key = key_map[ai_provider]

            user_input = ""
            run_preset = False

            if preset_category != "Select a preset question...":
                question_mapping = {
                    "General: Type of Inventory with most alerts": "Which type of inventory has generated the most alerts and how many?",
                    "General: Top 10 inventory items with most alerts": "List the top 10 inventory items that have the most alerts.",
                    "General: Inventory items with no alerts": "Identify which inventory items have generated no alerts.",
                    "General: Crunch time interval (95th percentile alerts)": "What is the crunch time hour interval during which 95th percentile alerts occurred?",
                    "Switches: Top Five Switches with most alerts": "List the top five switches with the most alerts.",
                    "Switches: Most common alert message": "What is the most common alert message among switches?",
                    "Switches: Problem free switches": "Which switches are problem-free (no alerts)?",
                    "Switches: Crunch time interval (95th percentile alerts)": "What is the crunch time hour interval for switches?",
                    "Firewall: Most common error": "What is the most common error for Firewalls?",
                    "Firewall: List most problematic Firewalls": "List the most problematic firewalls.",
                    "Firewall: Correlation with time of day and day of week": "Explain any correlation between Firewall messages and time of day or day of week.",
                    "Servers: Most common alert messages": "What are the most common alert messages for servers?",
                    "Servers: Specific OS with relatively higher messages": "Identify the specific OS platform with relatively higher alerts.",
                    "Servers: Correlation of messages to installation age": "Is there a correlation between server alerts and their installation age (Serial No proxy)?",
                    "Servers: Error-free servers": "Identify error-free servers.",
                    "Servers: Top 10 most problematic servers": "What are the top 10 most problematic servers?",
                    "Database: Top five common messages": "List the top 5 common database message errors.",
                    "Database: Correlation between CPU load/Table fragmentation and Query response": "Explain the correlation between Database CPU Load / Table Fragmentation and Query Response Time.",
                    "Database: Top five database instances with the most messages": "Identify top five database instances with the most alerts.",
                    "UPS: Top UPS with the most alert messages": "Identify the top UPS with the most alert messages.",
                    "UPS: Top three frequent messages": "List top three frequent messages for UPS units.",
                    "Printers: Top five printers with the most messages": "Identify top five printers with the most messages.",
                    "Printers: Top three Frequent Messages": "List top three frequent messages for printer units.",
                    "Overall Trend: Any alert message pattern over time indicating decrease, increase or stable volume": "Are there patterns indicating an increase, decrease, or stable number of alert messages over time?"
                }
                preset_query = question_mapping.get(preset_category, "")
                st.info(f"📍 **Target Prompt:** {preset_query}")
                if st.button("🚀 Run Selected Preset Query", use_container_width=True):
                    user_input = preset_query
                    run_preset = True

            # Standard chat input fallback
            chat_input_val = st.chat_input("Or enter a custom question...")
            if chat_input_val:
                user_input = chat_input_val

            if user_input:
                if not api_key:
                    st.warning(f"Provide a valid {ai_provider} API Key configuration.")
                else:
                    with st.chat_message("user"):
                        st.write(user_input)

                    with st.chat_message("assistant"):
                        with st.spinner("Processing analysis prompt..."):
                            data_summary = build_inventory_summary(inventory, tickets, merged)
                            db_ctx = get_context_from_db('inventory', limit=history_limit_inv) if use_history_inv else ""
                            kb_ctx, _, _ = enhance_prompt_with_kb(user_input, 'inventory')

                            system_prompt = f"""You are an expert IT operations system analyst with full access to the pre-computed analytics summary below.
This data is exact. Rely only on this context. Do not invent numbers.

{data_summary}

{db_ctx}
{kb_ctx}

INSTRUCTIONS:
1. Provide exact responses using values explicitly stated above.
2. For correlations, cite the Pearson coefficient values provided in the database section.
3. Keep layout clear and formatted.
"""
                            try:
                                if ai_provider == "OpenAI":
                                    client = OpenAI(api_key=api_key)
                                    response = client.chat.completions.create(
                                        model=model,
                                        messages=[
                                            {"role": "system", "content": system_prompt},
                                            {"role": "user", "content": user_input}
                                        ],
                                        temperature=temperature,
                                        max_tokens=max_tokens
                                    )
                                    full_text = response.choices[0].message.content

                                elif ai_provider == "Anthropic":
                                    client = anthropic.Anthropic(api_key=api_key)
                                    response = client.messages.create(
                                        model=model,
                                        max_tokens=max_tokens,
                                        temperature=temperature,
                                        system=system_prompt,
                                        messages=[{"role": "user", "content": user_input}]
                                    )
                                    full_text = response.content[0].text

                                else:
                                    client = Mistral(api_key=api_key)
                                    response = client.chat.complete(
                                        model=model,
                                        messages=[
                                            {"role": "system", "content": system_prompt},
                                            {"role": "user", "content": user_input}
                                        ],
                                        temperature=temperature,
                                        max_tokens=max_tokens
                                    )
                                    full_text = response.choices[0].message.content

                                st.markdown(full_text)

                                save_query_to_mongodb('inventory', {
                                    'ai_config': {'provider': ai_provider, 'model': model,
                                                  'temperature': temperature, 'max_tokens': max_tokens},
                                    'user_prompt': user_input,
                                    'ai_response': full_text,
                                    'history_entries_used': history_limit_inv if use_history_inv else 0
                                })

                            except Exception as e:
                                st.error(f"Error executing analysis: {e}")

        # ── TAB 4: Query History ─────────────────────────────────────────────
        with tab4:
            st.subheader("📜 Recent Query History")
            history = get_query_history('inventory', limit=20)
            if not history:
                st.info("No query logs saved yet.")
            else:
                for query in history:
                    fmt = format_query_for_display(query)
                    with st.expander(f"🕐 {fmt['Timestamp']} — {fmt['Provider']} ({fmt['Model']})"):
                        st.markdown(query.get('ai_response', 'N/A'))

        # ── TAB 5: Knowledge Base ────────────────────────────────────────────
        with tab5:
            st.subheader("📚 Knowledge Base Reference")
            categories = get_all_kb_categories()
            if categories:
                selected_cat = st.selectbox("Category Filter", ["All"] + categories)
                results = []
                if selected_cat == "All":
                    for cat in categories:
                        results.extend(get_kb_by_category(cat))
                else:
                    results = get_kb_by_category(selected_cat)

                for entry in results:
                    with st.expander(f"📌 {entry['category']}: {entry['question']}"):
                        st.markdown(entry['answer'])
            else:
                st.info("Knowledge base repository is empty.")

    # =========================================================================
    # MODULE 3: UNIFIED DASHBOARD
    # =========================================================================
    else:
        st.markdown('<div class="section-header">🎯 Unified Dashboard Overview</div>', unsafe_allow_html=True)

        server_data = load_server_data()
        inventory, tickets, merged = load_inventory_data()

        st.subheader("📊 System Overview")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            with st.container(border=True):
                st.metric("Server Data Points", len(server_data) if server_data is not None else "N/A")
        with c2:
            with st.container(border=True):
                if server_data is not None and not server_data.empty:
                    st.metric("Latest CPU", f"{server_data.iloc[-1].get('cpu_total', 0):.1f}%")
                else:
                    st.metric("Latest CPU", "N/A")
        with c3:
            with st.container(border=True):
                st.metric("Total Assets",  len(inventory) if inventory is not None else "N/A")
        with c4:
            with st.container(border=True):
                st.metric("Total Tickets", len(tickets)   if tickets   is not None else "N/A")


if __name__ == "__main__":
    main()