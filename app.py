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

def get_context_from_db(module_name, limit=5):
    """
    Fetches historical data from MongoDB and formats it for the AI 
    so the AI can 'remember' previous analyses.
    """
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

# Page Configuration
st.set_page_config(
    page_title="AI Intelligence Platform V3",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
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
# MONGODB CONNECTION AND FUNCTIONS
# ============================================================================

@st.cache_resource
def get_mongodb_client():
    """Initialize MongoDB client with connection pooling"""
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
    """Get the appropriate database and collection based on module"""
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
    """Save user query and AI response to MongoDB"""
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
    """Retrieve recent query history from MongoDB"""
    try:
        db, collection = get_database_collection(module_name)
        if collection is None:
            return []
        
        queries = collection.find().sort('timestamp', -1).limit(limit)
        return list(queries)
    except Exception as e:
        st.error(f"Error retrieving history: {e}")
        return []

def format_query_for_display(query_doc):
    """Format a query document for display"""
    timestamp = query_doc.get('timestamp', 'Unknown')
    if isinstance(timestamp, datetime):
        timestamp = timestamp.strftime('%Y-%m-%d %H:%M:%S')
    
    return {
        'ID': str(query_doc.get('_id', 'N/A')),
        'Timestamp': timestamp,
        'Provider': query_doc.get('ai_config', {}).get('provider', 'N/A'),
        'Model': query_doc.get('ai_config', {}).get('model', 'N/A'),
        'Prompt': query_doc.get('user_prompt', 'N/A')[:100] + '...' if len(query_doc.get('user_prompt', '')) > 100 else query_doc.get('user_prompt', 'N/A'),
    }

"""
Add these functions to your main application (app.py)
Place them after your existing MongoDB functions
"""

# ============================================================================
# KNOWLEDGE BASE FUNCTIONS - ADD TO YOUR EXISTING app.py
# ============================================================================

def get_knowledge_base_collection():
    """Get knowledge base collection"""
    client = get_mongodb_client()
    if client is None:
        return None
    
    db = client['ai_platform_knowledge_base']
    collection = db['qa_repository']
    return collection

def search_knowledge_base(query, category=None, limit=5):
    """
    Search knowledge base for relevant Q&As
    Returns matching questions and answers
    """
    try:
        collection = get_knowledge_base_collection()
        if collection is None:
            return []
        
        # Build search filter
        search_filter = {}
        if category:
            search_filter['category'] = category
        
        # Search in both question and answer fields
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
    """Get all unique categories from knowledge base"""
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
    """Get all Q&As from a specific category"""
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
    """
    Format knowledge base results into context string for AI
    """
    if not kb_results:
        return ""
    
    context = "\n--- KNOWLEDGE BASE REFERENCE ---\n"
    context += "Here are relevant Q&As from the knowledge base that may help answer the current query:\n\n"
    
    for idx, entry in enumerate(kb_results[:max_entries], 1):
        context += f"KB Entry {idx}:\n"
        context += f"Category: {entry.get('category', 'N/A')}\n"
        context += f"Q: {entry.get('question', 'N/A')}\n"
        context += f"A: {entry.get('answer', 'N/A')[:500]}...\n\n"
    
    context += "--- END KNOWLEDGE BASE REFERENCE ---\n"
    return context

def enhance_prompt_with_kb(user_prompt, module_name):
    """
    Enhance user prompt with relevant knowledge base context
    """
    # Determine category based on keywords in prompt
    category_keywords = {
        'General': ['inventory', 'alert', 'general'],
        'Switches': ['switch', 'switches', 'network'],
        'Firewall': ['firewall', 'security'],
        'Servers': ['server', 'os', 'operating system'],
        'Database': ['database', 'db', 'sql', 'query'],
        'UPS': ['ups', 'power', 'battery'],
        'Printers': ['printer', 'print']
    }
    
    # Find matching category
    detected_category = None
    prompt_lower = user_prompt.lower()
    
    for category, keywords in category_keywords.items():
        if any(keyword in prompt_lower for keyword in keywords):
            detected_category = category
            break
    
    # Search knowledge base
    kb_results = search_knowledge_base(user_prompt, category=detected_category, limit=3)
    
    if kb_results:
        kb_context = format_kb_context(kb_results)
        return kb_context, detected_category, len(kb_results)
    
    return "", None, 0


# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

def clean_column_name(column_name):
    """Cleans column names by removing (RAW), spaces, making lowercase."""
    name = str(column_name).replace('(RAW)', '').replace('(raw)', '').strip()
    name = ''.join(char if char.isalnum() else '_' for char in name)
    name = name.lower()
    while '__' in name:
        name = name.replace('__', '_')
    return name.strip('_')

@st.cache_data
def load_server_data():
    """Load and process server health data (CPU and Disk)"""
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
    """Process individual server data source"""
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
        if col == "Date Time" or col == "Date Time(RAW)":
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
    except:
        df_processed['interval_start_dt'] = pd.NaT
    
    df_processed.set_index(cleaned_date_time_key, inplace=True)
    return df_processed

@st.cache_data
def load_inventory_data():
    """Load and clean inventory and tickets data"""
    try:
        file_path = "testdata.xlsx"

        # read sheets
        inventory = pd.read_excel(
            file_path,
            sheet_name="Inventory Work Sheet",
            header=0
        )

        tickets = pd.read_excel(
            file_path,
            sheet_name="Tickets for Analysis",
            header=2
        )

        # standardize column names
        inventory.columns = [clean_column_name(c) for c in inventory.columns]
        tickets.columns = [clean_column_name(c) for c in tickets.columns]

        # expected columns after cleaning
        # inventory → serial_no, inv_code, device_type
        # tickets → ticket_id, timestamp, inv_code, device_type, error_message

        # convert timestamp
        tickets["timestamp"] = pd.to_datetime(
            tickets["timestamp"],
            errors="coerce"
        )

        # create useful time column
        tickets["month"] = tickets["timestamp"].dt.to_period("M").astype(str)

        # ticket count per inventory item
        ticket_count = (
            tickets
            .groupby("inv_code")
            .size()
            .reset_index(name="ticket_count")
        )

        # merge inventory with ticket count
        merged = (
            inventory
            .merge(ticket_count, on="inv_code", how="left")
            .fillna(0)
        )

        return inventory, tickets, merged

    except Exception as e:
        st.error(f"Error loading inventory data: {e}")
        return None, None, None

# ============================================================================
# SERVER HEALTH ANALYSIS FUNCTIONS
# ============================================================================

def format_server_data_for_llm(data_df, start_dt, end_dt):
    """Format server data for LLM analysis"""
    if data_df is None or data_df.empty:
        return f"No data found for period: {start_dt} to {end_dt}"
    
    num_points = len(data_df)
    lines = [f"Server Health Metrics ({num_points} data points)", "---", ""]
    
    lines.append("[CPU Metrics]")
    if 'cpu_total' in data_df.columns:
        lines.append(f"- Total CPU: Avg {data_df['cpu_total'].mean():.2f}%, Max {data_df['cpu_total'].max():.2f}%")
    
    lines.append("\n[Disk Metrics]")
    if 'disk_total' in data_df.columns:
        lines.append(f"- Total Free Space: Avg {(data_df['disk_total'].mean()/(1024**3)):.2f} GB")
    
    return "\n".join(lines)

def analyze_with_ai(data_snapshot, context, prompt, provider, model, temperature, max_tokens, api_key, db_context="", kb_context=""):
    """Universal AI analysis function supporting multiple providers"""
    try:
        system_msg = f"""You are an expert server health monitoring AI. 
        Current System Context: {context}
        
        {db_context}
        {kb_context}
        When answering:
        1. Consider historical entries from the database to identify trends
        2. Reference the knowledge base when relevant questions have been answered before
        3. Provide actionable insights based on both current data and historical context"""
        
        user_msg = f"Current Live Data Snapshot:\n{data_snapshot}\n\nNew Request:\n{prompt}"
        
        if provider == "OpenAI":
            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
        
        elif provider == "Google Gemini":
            genai.configure(api_key=api_key)
            gen_model = genai.GenerativeModel(model)
            full_prompt = f"{system_msg}\n\n{user_msg}"
            generation_config = genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens
            )
            response = gen_model.generate_content(full_prompt, generation_config=generation_config)
            return response.text
        
        elif provider == "Mistral AI":
            client = Mistral(api_key=api_key)
            response = client.chat.complete(
                model=model,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
    
    except Exception as e:
        return f"Error during {provider} analysis: {e}"

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    st.markdown('<div class="main-header">🤖 AI Intelligence Platform V3</div>', unsafe_allow_html=True)
    st.markdown("*Combined Server Health Analysis & Inventory Intelligence*")
    
    # Sidebar Configuration
    with st.sidebar:
        st.title("🚀 Configuration")
        
        st.subheader("API Keys")
        openai_key = st.text_input("OpenAI API Key", type="password", value=os.getenv("OPENAI_API_KEY", ""))
        gemini_key = st.text_input("Google API Key", type="password", value=os.getenv("GOOGLE_API_KEY", ""))
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
    
    # ======================================================================== 
    # MODULE 1: SERVER HEALTH ANALYSIS
    # ========================================================================
    if selected_module == "Server Health Analysis (V1)":
        st.markdown('<div class="section-header">📊 Server Health Analysis</div>', unsafe_allow_html=True)

        # --- HISTORY TOGGLE + ENTRY COUNT (side by side) ---
        hist_col1, hist_col2 = st.columns([1, 1])
        with hist_col1:
            use_history = st.checkbox("🔗 Include historical database records in this analysis?", value=False)
        with hist_col2:
            if use_history:
                history_limit = st.number_input(
                    "Number of historical entries to include",
                    min_value=1,
                    max_value=50,
                    value=5,
                    step=1,
                    help="How many past database entries should the AI consider when generating its response?"
                )
            else:
                history_limit = 5  # default, won't be used

        server_data = load_server_data()
        
        if server_data is None or server_data.empty:
            st.warning("⚠️ Server data files not found. Please ensure `historicdataCPU.csv` and `historicdataDISK.csv` are in the same directory.")
            return
        
        tab1, tab2 = st.tabs(["🔍 New Analysis", "📜 Query History"])
        
        with tab1:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📅 Time Range Selection")
                start_date = st.date_input("Start Date", datetime.now().date())
                start_time = st.time_input("Start Time", datetime.now().time())
                end_date = st.date_input("End Date", datetime.now().date())
                end_time = st.time_input("End Time", datetime.now().time())
            
            with col2:
                st.subheader("🔧 AI Configuration")
                ai_provider = st.selectbox("AI Provider", ["OpenAI", "Google Gemini", "Mistral AI"])
                
                if ai_provider == "OpenAI":
                    model = st.selectbox("Model", ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"])
                elif ai_provider == "Google Gemini":
                    model = st.selectbox("Model", [
                        "gemini-2.0-flash-exp",
                        "gemini-1.5-pro",
                        "gemini-1.5-flash",
                        "gemini-1.5-flash-8b"
                    ])
                else:
                    model = st.selectbox("Model", [
                        "mistral-large-latest",
                        "mistral-small-latest",
                        "open-mistral-7b",
                        "open-mixtral-8x7b"
                    ])
                
                temperature = st.slider("Temperature", 0.0, 1.0, 0.3, 0.1)
                max_tokens = st.number_input("Max Tokens", 100, 8000, 1000, 100)
            
            st.subheader("💭 Analysis Context & Prompt")
            context = st.text_area("System Context (Optional)", 
                                   "Windows Server running SQL Server. CPU core 3 handles indexing.", 
                                   height=80)
            user_prompt = st.text_area("Your Question/Prompt", 
                                       "Analyze for critical issues and provide top 3 recommendations.", 
                                       height=100)
            
            if st.button("🔍 Analyze", type="primary", use_container_width=True):
                db_context_data = ""
                if use_history:
                    with st.spinner(f"Fetching {history_limit} historical entries from MongoDB..."):
                        db_context_data = get_context_from_db('server_health', limit=history_limit)
                    st.info(f"📚 AI is considering the last **{history_limit}** historical database entries.")
                    # Search knowledge base
                    kb_context_data = ""
                    kb_count = 0
                    with st.spinner("Searching knowledge base for relevant information..."):
                        kb_context_data, detected_category, kb_count = enhance_prompt_with_kb(user_prompt, 'server_health')
                    
                    if kb_count > 0:
                        st.info(f"📖 Found {kb_count} relevant knowledge base entries" + 
                                (f" in category: {detected_category}" if detected_category else ""))
                if ai_provider == "OpenAI" and not openai_key:
                    st.error("Please provide OpenAI API key in the sidebar.")
                    return
                elif ai_provider == "Google Gemini" and not gemini_key:
                    st.error("Please provide Google API key in the sidebar.")
                    return
                elif ai_provider == "Mistral AI" and not mistral_key:
                    st.error("Please provide Mistral API key in the sidebar.")
                    return
                
                api_key = openai_key if ai_provider == "OpenAI" else (gemini_key if ai_provider == "Google Gemini" else mistral_key)
                
                with st.spinner("Analyzing server health data..."):
                    start_dt = datetime.combine(start_date, start_time)
                    end_dt = datetime.combine(end_date, end_time)
                    
                    filtered_data = server_data[
                        (server_data['interval_start_dt'] >= start_dt) & 
                        (server_data['interval_start_dt'] <= end_dt)
                    ].copy()
                    
                    if filtered_data.empty:
                        st.warning("No data found in the selected time range.")
                        return
                    
                    data_snapshot = format_server_data_for_llm(filtered_data, start_dt, end_dt)
                    
                    st.subheader("📊 Data Summary")
                    st.text(data_snapshot)
                    
                    st.subheader("🤖 AI Analysis")
                    result = analyze_with_ai(data_snapshot, context, user_prompt, 
                                ai_provider, model, temperature, max_tokens, api_key,
                                db_context=db_context_data, kb_context=kb_context_data)
                    st.markdown(result)
                    
                    query_data = {
                        'time_range': {
                            'start': start_dt,
                            'end': end_dt
                        },
                        'ai_config': {
                            'provider': ai_provider,
                            'model': model,
                            'temperature': temperature,
                            'max_tokens': max_tokens
                        },
                        'context': context,
                        'user_prompt': user_prompt,
                        'data_summary': data_snapshot,
                        'ai_response': result,
                        'data_points_analyzed': len(filtered_data),
                        'history_entries_used': history_limit if use_history else 0
                    }
                    
                    saved_id = save_query_to_mongodb('server_health', query_data)
                    if saved_id:
                        st.success(f"✅ Query saved to database with ID: {saved_id}")
        
        with tab2:
            st.subheader("📜 Recent Query History")
            
            history = get_query_history('server_health', limit=20)
            
            if not history:
                st.info("No query history found. Run some analyses to build your history!")
            else:
                for idx, query in enumerate(history):
                    formatted = format_query_for_display(query)
                    
                    with st.expander(f"🕐 {formatted['Timestamp']} - {formatted['Provider']} ({formatted['Model']})"):
                        col_a, col_b = st.columns(2)
                        
                        with col_a:
                            st.write("**Configuration:**")
                            st.write(f"- Provider: {query.get('ai_config', {}).get('provider', 'N/A')}")
                            st.write(f"- Model: {query.get('ai_config', {}).get('model', 'N/A')}")
                            st.write(f"- Temperature: {query.get('ai_config', {}).get('temperature', 'N/A')}")
                            st.write(f"- Max Tokens: {query.get('ai_config', {}).get('max_tokens', 'N/A')}")
                        
                        with col_b:
                            st.write("**Time Range:**")
                            time_range = query.get('time_range', {})
                            st.write(f"- Start: {time_range.get('start', 'N/A')}")
                            st.write(f"- End: {time_range.get('end', 'N/A')}")
                            st.write(f"- Data Points: {query.get('data_points_analyzed', 'N/A')}")
                            st.write(f"- History Entries Used: {query.get('history_entries_used', 'N/A')}")
                        
                        st.write("**Context:**")
                        st.text(query.get('context', 'N/A'))
                        
                        st.write("**User Prompt:**")
                        st.text(query.get('user_prompt', 'N/A'))
                        
                        st.write("**AI Response:**")
                        st.markdown(query.get('ai_response', 'N/A'))
    
    # ======================================================================== 
    # MODULE 2: INVENTORY INTELLIGENCE
    # ========================================================================
    elif selected_module == "Inventory Intelligence (V2)":
        st.markdown('<div class="section-header">📦 Inventory Intelligence Hub</div>', unsafe_allow_html=True)

        # --- HISTORY TOGGLE + ENTRY COUNT (side by side) ---
        hist_col1, hist_col2 = st.columns([1, 1])
        with hist_col1:
            use_history_inv = st.checkbox("📚 Let AI access previous database queries for context?", value=False)
        with hist_col2:
            if use_history_inv:
                history_limit_inv = st.number_input(
                    "Number of historical entries to include",
                    min_value=1,
                    max_value=50,
                    value=5,
                    step=1,
                    key="inv_history_limit",
                    help="How many past database entries should the AI consider when generating its response?"
                )
            else:
                history_limit_inv = 5  # default, won't be used

        inventory, tickets, merged = load_inventory_data()
        
        if inventory is None:
            st.warning("⚠️ Inventory data file not found. Please ensure `testdata.xlsx` is in the same directory.")
            return
        
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Visual Dashboard", "🤖 AI Data Assistant", "📜 Query History", "📚 Knowledge Base"])
        
        with tab1:
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Items", len(inventory))
            c2.metric("Total Tickets", len(tickets))
            c3.metric("Active Devices", tickets["device_type"].nunique())
            
            col_a, col_b = st.columns(2)
            
            with col_a:
                st.subheader("Tickets by Item Type")
                type_data = merged.groupby("device_type")["ticket_count"].sum().reset_index()
                fig1 = px.bar(type_data, x="device_type", y="ticket_count", 
                            template="plotly_dark", color="ticket_count", 
                            color_continuous_scale="blues")
                st.plotly_chart(fig1, use_container_width=True)
            
            with col_b:
                st.subheader("Ticket Trends Over Time")
                trend = tickets.groupby("month").size().reset_index(name="count")
                fig2 = px.line(trend, x="month", y="count", markers=True, template="plotly_dark")
                st.plotly_chart(fig2, use_container_width=True)
        
        with tab2:
            st.subheader("💬 Chat with your Data")
            
            col1, col2 = st.columns(2)
            
            with col1:
                ai_provider = st.selectbox("AI Provider", ["OpenAI", "Google Gemini", "Mistral AI"], key="inv_provider")
                
                if ai_provider == "OpenAI":
                    model = st.selectbox("Model", ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"], key="inv_model")
                elif ai_provider == "Google Gemini":
                    model = st.selectbox("Model", [
                        "gemini-2.0-flash-exp",
                        "gemini-1.5-pro",
                        "gemini-1.5-flash",
                        "gemini-1.5-flash-8b"
                    ], key="inv_model")
                else:
                    model = st.selectbox("Model", [
                        "mistral-large-latest",
                        "mistral-small-latest",
                        "open-mistral-7b"
                    ], key="inv_model")
            
            with col2:
                temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.1, key="inv_temp")
                max_tokens = st.number_input("Max Tokens", 100, 8000, 2000, 100, key="inv_tokens")
            
            if ai_provider == "OpenAI" and not openai_key:
                st.warning("Please enter OpenAI API Key in the sidebar.")
            elif ai_provider == "Google Gemini" and not gemini_key:
                st.warning("Please enter Google API Key in the sidebar.")
            elif ai_provider == "Mistral AI" and not mistral_key:
                st.warning("Please enter Mistral API Key in the sidebar.")
            else:
                user_input = st.chat_input("Ask me a question or request a chart...")
                
                if user_input:
                    with st.chat_message("user"):
                        st.write(user_input)
                    
                    with st.chat_message("assistant"):
                        with st.spinner("Analyzing..."):
                            # 1. Create a data sample to help the AI understand values
                            inv_sample = inventory.head(3).to_string()
                            tickets_sample = tickets.head(3).to_string()

                            # 2. Build the enhanced prompt
                            prompt = f"""
You are a senior data analyst. You must answer the user's question using the following pre-loaded DataFrames:

DATAFRAMES:
1. `inventory`: Main asset list. Columns: {list(inventory.columns)}
2. `tickets`: All reported issues. Columns: {list(tickets.columns)}
3. `merged`: Inventory merged with ticket counts. Columns: {list(merged.columns)}

DATA SAMPLES (Top 3 rows):
Inventory:
{inv_sample}
Tickets:
{tickets_sample}

USER QUESTION: {user_input}

INSTRUCTIONS:
- For ALL numerical, statistical, or filtering questions, you MUST generate Python code.
- Store your final answer in a variable named `result`.
- Use `tickets` or `merged` to find counts/failures. 
- Example for "most failures": `result = tickets['device_type'].value_counts().idxmax()` or a dataframe showing counts.
- Output ONLY Python code inside ```python``` fences. No explanation before or after.
"""
                            
                            api_key = openai_key if ai_provider == "OpenAI" else (gemini_key if ai_provider == "Google Gemini" else mistral_key)
                            
                            try:
                                if ai_provider == "OpenAI":
                                    client = OpenAI(api_key=api_key)
                                    response = client.chat.completions.create(
                                        model=model,
                                        messages=[{"role": "user", "content": prompt}],
                                        temperature=temperature,
                                        max_tokens=max_tokens
                                    )
                                    full_text = response.choices[0].message.content
                                
                                elif ai_provider == "Google Gemini":
                                    genai.configure(api_key=api_key)
                                    gen_model = genai.GenerativeModel(model)
                                    generation_config = genai.types.GenerationConfig(
                                        temperature=temperature,
                                        max_output_tokens=max_tokens
                                    )
                                    response = gen_model.generate_content(prompt, generation_config=generation_config)
                                    full_text = response.text
                                
                                else:
                                    client = Mistral(api_key=api_key)
                                    response = client.chat.complete(
                                        model=model,
                                        messages=[{"role": "user", "content": prompt}],
                                        temperature=temperature,
                                        max_tokens=max_tokens
                                    )
                                    full_text = response.choices[0].message.content
                                
                                code_match = re.search(r"```python\n(.*?)```", full_text, re.DOTALL)
                                
                                if code_match:
                                    code = code_match.group(1)
                                    local_env = {
                                        "pd": pd,
                                        "np": np,
                                        "px": px,
                                        "inventory": inventory,
                                        "tickets": tickets,
                                        "merged": merged
                                    }
                                    exec(code, {}, local_env)
                                    result = local_env.get("result")
                                    
                                    if isinstance(result, (pd.DataFrame, pd.Series)):
                                        st.dataframe(result, use_container_width=True)
                                    elif "plotly" in str(type(result)):
                                        st.plotly_chart(result, use_container_width=True)
                                    else:
                                        st.write(result)
                                    
                                    with st.expander("View Generated Code"):
                                        st.code(code, language="python")
                                else:
                                    st.write(full_text)
                                
                                query_data = {
                                    'ai_config': {
                                        'provider': ai_provider,
                                        'model': model,
                                        'temperature': temperature,
                                        'max_tokens': max_tokens
                                    },
                                    'user_prompt': user_input,
                                    'ai_response': full_text,
                                    'code_generated': code_match is not None,
                                    'code': code_match.group(1) if code_match else None,
                                    'history_entries_used': history_limit_inv if use_history_inv else 0
                                }
                                
                                saved_id = save_query_to_mongodb('inventory', query_data)
                                if saved_id:
                                    st.success(f"✅ Query saved to database")
                            
                            except Exception as e:
                                st.error(f"Error: {e}")
        
        with tab3:
            st.subheader("📜 Recent Query History")
            
            history = get_query_history('inventory', limit=20)
            
            if not history:
                st.info("No query history found. Start chatting to build your history!")
            else:
                for idx, query in enumerate(history):
                    formatted = format_query_for_display(query)
                    
                    with st.expander(f"🕐 {formatted['Timestamp']} - {formatted['Provider']} ({formatted['Model']})"):
                        col_a, col_b = st.columns(2)
                        
                        with col_a:
                            st.write("**Configuration:**")
                            st.write(f"- Provider: {query.get('ai_config', {}).get('provider', 'N/A')}")
                            st.write(f"- Model: {query.get('ai_config', {}).get('model', 'N/A')}")
                            st.write(f"- Temperature: {query.get('ai_config', {}).get('temperature', 'N/A')}")
                            st.write(f"- Max Tokens: {query.get('ai_config', {}).get('max_tokens', 'N/A')}")
                        
                        with col_b:
                            st.write("**Code Generated:**")
                            st.write(f"- {query.get('code_generated', False)}")
                            st.write(f"- History Entries Used: {query.get('history_entries_used', 'N/A')}")
                        
                        st.write("**User Prompt:**")
                        st.text(query.get('user_prompt', 'N/A'))
                        
                        st.write("**AI Response:**")
                        st.markdown(query.get('ai_response', 'N/A'))
                        
                        if query.get('code'):
                            st.write("**Generated Code:**")
                            st.code(query.get('code'), language='python')
        with tab4:
            st.subheader("📚 Knowledge Base Repository")
            
            st.markdown("""
            This knowledge base contains pre-generated answers to common questions about your infrastructure.
            The AI automatically references this when answering your questions.
            """)
            
            # Category filter
            col1, col2 = st.columns([1, 3])
            
            with col1:
                categories = get_all_kb_categories()
                if categories:
                    selected_category = st.selectbox(
                        "Filter by Category",
                        ["All"] + categories
                    )
                else:
                    st.warning("⚠️ Knowledge base is empty. Run `kb_builder_script.py` first.")
                    selected_category = "All"
            
            with col2:
                search_term = st.text_input("🔍 Search knowledge base", "")
            
            # Display knowledge base entries
            if categories:
                if search_term:
                    # Search mode
                    results = search_knowledge_base(search_term, 
                                                   category=None if selected_category == "All" else selected_category,
                                                   limit=20)
                    st.info(f"Found {len(results)} matching entries")
                else:
                    # Category browsing mode
                    if selected_category == "All":
                        results = []
                        for cat in categories:
                            results.extend(get_kb_by_category(cat))
                    else:
                        results = get_kb_by_category(selected_category)
                
                # Display results
                if results:
                    for entry in results:
                        with st.expander(f"📌 {entry['category']}: {entry['question']}"):
                            st.markdown(f"**Category:** {entry['category']}")
                            st.markdown(f"**Question:** {entry['question']}")
                            st.markdown("**Answer:**")
                            st.markdown(entry['answer'])
                            
                            st.divider()
                            col_a, col_b = st.columns(2)
                            with col_a:
                                st.caption(f"Generated: {entry.get('timestamp', 'N/A')}")
                            with col_b:
                                st.caption(f"Model: {entry.get('ai_provider', 'N/A')} - {entry.get('ai_model', 'N/A')}")
                else:
                    st.info("No entries found matching your criteria.")
    # ======================================================================== 
    # MODULE 3: UNIFIED DASHBOARD
    # ========================================================================
    else:
        st.markdown('<div class="section-header">🎯 Unified Intelligence Dashboard</div>', unsafe_allow_html=True)
        
        server_data = load_server_data()
        inventory, tickets, merged = load_inventory_data()
        
        st.subheader("📊 System Overview")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if server_data is not None and not server_data.empty:
                st.metric("Server Data Points", len(server_data))
            else:
                st.metric("Server Data Points", "N/A")
        
        with col2:
            if server_data is not None and not server_data.empty:
                latest_cpu = server_data.iloc[-1].get('cpu_total', 0)
                st.metric("Latest CPU Usage", f"{latest_cpu:.1f}%")
            else:
                st.metric("Latest CPU Usage", "N/A")
        
        with col3:
            if inventory is not None:
                st.metric("Total Assets", len(inventory))
            else:
                st.metric("Total Assets", "N/A")
        
        with col4:
            if tickets is not None:
                st.metric("Total Tickets", len(tickets))
            else:
                st.metric("Total Tickets", "N/A")
        
        st.divider()
        
        if server_data is not None and not server_data.empty:
            st.subheader("🖥️ Server Health Metrics")
            col_a, col_b = st.columns(2)
            
            with col_a:
                if 'cpu_total' in server_data.columns and 'interval_start_dt' in server_data.columns:
                    recent_data = server_data.tail(50).copy()
                    fig_cpu = px.line(
                        recent_data,
                        x='interval_start_dt',
                        y='cpu_total',
                        title='CPU Usage Over Time (Last 50 Points)',
                        template='plotly_dark',
                        labels={'cpu_total': 'CPU %', 'interval_start_dt': 'Time'}
                    )
                    fig_cpu.update_traces(line_color='#00d4ff')
                    st.plotly_chart(fig_cpu, use_container_width=True)
            
            with col_b:
                if 'disk_total' in server_data.columns and 'interval_start_dt' in server_data.columns:
                    recent_data = server_data.tail(50).copy()
                    recent_data['disk_gb'] = recent_data['disk_total'] / (1024**3)
                    fig_disk = px.line(
                        recent_data,
                        x='interval_start_dt',
                        y='disk_gb',
                        title='Disk Free Space Over Time (Last 50 Points)',
                        template='plotly_dark',
                        labels={'disk_gb': 'Free Space (GB)', 'interval_start_dt': 'Time'}
                    )
                    fig_disk.update_traces(line_color='#58a6ff')
                    st.plotly_chart(fig_disk, use_container_width=True)
        
        st.divider()
        
        if inventory is not None and tickets is not None:
            st.subheader("📦 Inventory & Ticket Analytics")
            col_c, col_d = st.columns(2)
            
            with col_c:
                device_tickets = tickets.groupby('device_type').size().reset_index(name='count')
                fig_devices = px.pie(
                    device_tickets,
                    values='count',
                    names='device_type',
                    title='Tickets by Device Type',
                    template='plotly_dark',
                    color_discrete_sequence=px.colors.sequential.Blues_r
                )
                st.plotly_chart(fig_devices, use_container_width=True)
            
            with col_d:
                top_assets = merged.nlargest(10, 'ticket_count')[['inv_code', 'ticket_count', 'device_type']]
                fig_top = px.bar(
                    top_assets,
                    x='inv_code',
                    y='ticket_count',
                    color='device_type',
                    title='Top 10 Assets by Ticket Count',
                    template='plotly_dark',
                    labels={'ticket_count': 'Tickets', 'inv_code': 'Asset Code'}
                )
                st.plotly_chart(fig_top, use_container_width=True)
            
            st.subheader("📈 Ticket Trends")
            monthly_trend = tickets.groupby('month').size().reset_index(name='ticket_count')
            fig_trend = px.area(
                monthly_trend,
                x='month',
                y='ticket_count',
                title='Monthly Ticket Volume',
                template='plotly_dark',
                labels={'ticket_count': 'Number of Tickets', 'month': 'Month'}
            )
            fig_trend.update_traces(fill='tozeroy', line_color='#00d4ff')
            st.plotly_chart(fig_trend, use_container_width=True)
        
        st.divider()
        st.info("💡 Select a specific module from the sidebar to perform detailed AI-powered analysis.")

if __name__ == "__main__":
    main()