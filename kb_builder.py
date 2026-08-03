"""
Knowledge Base Builder - Generates and stores answers to predefined questions
Run this script once to build your knowledge base from your data
"""

import pandas as pd
from pymongo import MongoClient
from datetime import datetime
import os
from dotenv import load_dotenv
import google.generativeai as genai
from openai import OpenAI
# from mistralai import Mistral

load_dotenv()

# ============================================================================
# PREDEFINED QUESTIONS BY CATEGORY
# ============================================================================

KNOWLEDGE_BASE_QUESTIONS = {
    "General": [
        "Type of Inventory with most alerts",
        "Top 10 inventory items with most alerts",
        "Inventory items with no alerts",
        "Crunch time interval during which 95 percentile alerts occurred"
    ],
    "Switches": [
        "Top Five Switches with most alerts",
        "Most common alert message for switches",
        "Problem free switches",
        "Crunch time interval during which 95 percentile alerts occurred for switches"
    ],
    "Firewall": [
        "Most common firewall error",
        "List most problematic Firewalls",
        "Is there a correlation between Firewall messages and time of the day and day of the week"
    ],
    "Servers": [
        "What are the most common alert messages for servers",
        "Specific OS with relatively higher messages",
        "Any correlation of messages to installation age",
        "Error free servers",
        "Top 10 most problematic servers"
    ],
    "Database": [
        "Top five common database messages",
        "Any correlation between CPU load and Query response",
        "Top five database instances with the most messages"
    ],
    "UPS": [
        "Top UPS with the most alert messages",
        "Top three frequent UPS messages"
    ],
    "Printers": [
        "Top five printers with the most messages",
        "Top three frequent printer messages"
    ]
}

# ============================================================================
# MONGODB FUNCTIONS
# ============================================================================

def get_mongodb_client():
    """Connect to MongoDB"""
    mongo_uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
    client = MongoClient(mongo_uri)
    return client

def initialize_knowledge_base():
    """Create knowledge base database and collection"""
    client = get_mongodb_client()
    db = client['ai_platform_knowledge_base']
    collection = db['qa_repository']
    
    # Create indexes for efficient querying
    collection.create_index([("category", 1)])
    collection.create_index([("question", 1)])
    collection.create_index([("timestamp", -1)])
    
    return collection

# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

def load_inventory_data():
    """Load inventory and tickets data"""
    try:
        file_path = "testdata.xlsx"
        inventory = pd.read_excel(file_path, sheet_name="Inventory Work Sheet", header=0)
        tickets = pd.read_excel(file_path, sheet_name="Tickets for Analysis", header=2)
        
        inventory.columns = inventory.columns.str.strip().str.lower()
        tickets.columns = tickets.columns.str.strip().str.lower()
        
        inventory.rename(columns={
            "inv code": "inv_code",
            "inv item type": "inv_item_type"
        }, inplace=True)
        
        tickets.rename(columns={
            "inv code": "inv_code",
            "device type": "device_type",
            "date & time": "date_time"
        }, inplace=True)
        
        tickets["date_time"] = pd.to_datetime(tickets["date_time"], errors="coerce")
        tickets["month"] = tickets["date_time"].dt.to_period("M").astype(str)
        
        return inventory, tickets
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None

# ============================================================================
# AI ANSWER GENERATION
# ============================================================================

def generate_answer_with_ai(question, category, inventory, tickets, provider="OpenAI", model="gpt-4o"):
    """Generate answer using AI based on actual data"""
    
    # Prepare data summary
    data_summary = f"""
    Dataset Summary:
    - Total Inventory Items: {len(inventory)}
    - Total Tickets: {len(tickets)}
    - Unique Device Types: {tickets['device_type'].nunique() if 'device_type' in tickets.columns else 'N/A'}
    - Date Range: {tickets['date_time'].min()} to {tickets['date_time'].max() if 'date_time' in tickets.columns else 'N/A'}
    
    Available Columns in Inventory: {', '.join(inventory.columns.tolist())}
    Available Columns in Tickets: {', '.join(tickets.columns.tolist())}
    """
    
    prompt = f"""
    You are an expert data analyst analyzing IT infrastructure inventory and alert data.
    
    Category: {category}
    Question: {question}
    
    {data_summary}
    
    Based on the data structure provided, give a comprehensive answer to the question.
    Include:
    1. Direct answer to the question
    2. Key insights and patterns
    3. Actionable recommendations
    4. Any caveats or data limitations
    
    If you need to perform calculations, explain the methodology.
    Format your response in a clear, professional manner with bullet points where appropriate.
    """
    
    try:
        if provider == "OpenAI":
            api_key = os.getenv("OPENAI_API_KEY")
            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=2000
            )
            return response.choices[0].message.content
        
        elif provider == "Google Gemini":
            api_key = os.getenv("GOOGLE_API_KEY")
            genai.configure(api_key=api_key)
            gen_model = genai.GenerativeModel(model)
            response = gen_model.generate_content(prompt)
            return response.text
        
        elif provider == "Mistral AI":
            api_key = os.getenv("MISTRAL_API_KEY")
            client = Mistral(api_key=api_key)
            response = client.chat.complete(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=2000
            )
            return response.choices[0].message.content
            
    except Exception as e:
        return f"Error generating answer: {e}"

# ============================================================================
# KNOWLEDGE BASE BUILDING
# ============================================================================

def build_knowledge_base(provider="OpenAI", model="gpt-4o"):
    """Build complete knowledge base by generating answers to all questions"""
    
    print("=" * 80)
    print("KNOWLEDGE BASE BUILDER")
    print("=" * 80)
    
    # Load data
    print("\n📁 Loading inventory and tickets data...")
    inventory, tickets = load_inventory_data()
    
    if inventory is None or tickets is None:
        print("❌ Failed to load data. Please ensure testdata.xlsx is available.")
        return
    
    print(f"✅ Data loaded successfully!")
    print(f"   - Inventory items: {len(inventory)}")
    print(f"   - Tickets: {len(tickets)}")
    
    # Initialize MongoDB
    print("\n💾 Connecting to MongoDB...")
    collection = initialize_knowledge_base()
    print("✅ Connected to knowledge base database")
    
    # Generate answers for all questions
    total_questions = sum(len(questions) for questions in KNOWLEDGE_BASE_QUESTIONS.values())
    current = 0
    
    print(f"\n🤖 Generating answers for {total_questions} questions using {provider} ({model})...")
    print("-" * 80)
    
    for category, questions in KNOWLEDGE_BASE_QUESTIONS.items():
        print(f"\n📂 Category: {category}")
        
        for question in questions:
            current += 1
            print(f"   [{current}/{total_questions}] Processing: {question[:60]}...")
            
            # Generate answer
            answer = generate_answer_with_ai(
                question, category, inventory, tickets, 
                provider=provider, model=model
            )
            
            # Create document
            document = {
                "category": category,
                "question": question,
                "answer": answer,
                "timestamp": datetime.now(),
                "ai_provider": provider,
                "ai_model": model,
                "data_summary": {
                    "inventory_count": len(inventory),
                    "tickets_count": len(tickets),
                    "device_types": tickets['device_type'].nunique() if 'device_type' in tickets.columns else 0
                }
            }
            
            # Check if question already exists
            existing = collection.find_one({"category": category, "question": question})
            
            if existing:
                # Update existing
                collection.update_one(
                    {"_id": existing["_id"]},
                    {"$set": document}
                )
                print(f"      ✏️  Updated existing entry")
            else:
                # Insert new
                collection.insert_one(document)
                print(f"      ✅ Saved to knowledge base")
    
    print("\n" + "=" * 80)
    print("✅ KNOWLEDGE BASE BUILT SUCCESSFULLY!")
    print(f"📊 Total entries: {collection.count_documents({})}")
    print("=" * 80)
    
    # Display summary
    print("\n📋 Summary by Category:")
    for category in KNOWLEDGE_BASE_QUESTIONS.keys():
        count = collection.count_documents({"category": category})
        print(f"   - {category}: {count} questions")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Build Knowledge Base from Predefined Questions')
    parser.add_argument('--provider', type=str, default='OpenAI', 
                       choices=['OpenAI', 'Google Gemini', 'Mistral AI'],
                       help='AI provider to use')
    parser.add_argument('--model', type=str, default='gpt-4o',
                       help='AI model to use')
    
    args = parser.parse_args()
    
    print(f"\n🚀 Starting Knowledge Base Builder")
    print(f"   Provider: {args.provider}")
    print(f"   Model: {args.model}")
    
    build_knowledge_base(provider=args.provider, model=args.model)