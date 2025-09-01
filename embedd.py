import google.generativeai as genai
from supabase import create_client
import json
from dotenv import load_dotenv
import os
import uuid
import time

# Load environment variables
load_dotenv()

# --- Config ---
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Validate environment variables
if not all([SUPABASE_URL, SUPABASE_KEY, GEMINI_API_KEY]):
    raise ValueError("Missing required environment variables. Check your .env file.")

# Init clients
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
genai.configure(api_key=GEMINI_API_KEY)

# Load JSON chunks
try:
    with open("hospital_chunks.json", "r", encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"✅ Loaded {len(chunks)} chunks from hospital_chunks.json")
except FileNotFoundError:
    print("❌ Error: hospital_chunks.json file not found.")
    exit(1)
except json.JSONDecodeError:
    print("❌ Error: Invalid JSON format in hospital_chunks.json.")
    exit(1)

# Validate chunks structure
if not isinstance(chunks, list):
    print("❌ Error: hospital_chunks.json should contain a list of chunks.")
    exit(1)

# Embedding model
embedding_model = "models/embedding-001"

print(f"🚀 Processing {len(chunks)} chunks...")
print(f"📊 Embedding model: {embedding_model}")
print("-" * 50)

# Track success/failure
success_count = 0
error_count = 0

for i, chunk in enumerate(chunks):
    # Validate chunk structure
    if not isinstance(chunk, dict) or "content" not in chunk:
        print(f"⚠️  Skipping invalid chunk at index {i} - missing 'content' field")
        error_count += 1
        continue
    
    # Skip if content is empty
    if not chunk["content"].strip():
        print(f"⚠️  Skipping empty chunk at index {i}")
        error_count += 1
        continue
        
    # Ensure chunk has an ID
    if "id" not in chunk or not chunk["id"]:
        chunk["id"] = str(uuid.uuid4())
        print(f"🔧 Generated ID for chunk {i}: {chunk['id']}")
    
    # Generate embedding from Gemini
    try:
        result = genai.embed_content(
            model=embedding_model,
            content=chunk["content"],
            task_type="retrieval_document"  # Specify task type for better embeddings
        )
        embedding = result["embedding"]
        
        # Validate embedding
        if not embedding or len(embedding) == 0:
            print(f"❌ Empty embedding for chunk {chunk['id']}")
            error_count += 1
            continue
            
    except Exception as e:
        print(f"❌ Error generating embedding for chunk {chunk.get('id', 'unknown')}: {e}")
        error_count += 1
        continue

    # Prepare data for insertion
    insert_data = {
        "id": chunk["id"],
        "content": chunk["content"].strip(),
        "section": chunk.get("section", "general"),
        "category": chunk.get("category", "general"),
        "page": chunk.get("page"),
        "embedding": embedding
    }

    # Insert into Supabase
    try:
        response = supabase.table("hospital_embeddings").upsert(insert_data).execute()
        
        # Check if insert was successful
        if response.data:
            print(f"✅ Processed chunk {i+1}/{len(chunks)}: {chunk['id']} ({chunk.get('section', 'unknown')})")
            success_count += 1
        else:
            print(f"⚠️  Warning: No data returned for chunk {chunk['id']}")
            
    except Exception as e:
        print(f"❌ Error inserting chunk {chunk['id']} into Supabase: {e}")
        error_count += 1
        continue
    
    # Add small delay to avoid rate limiting
    time.sleep(0.1)

print("-" * 50)
print(f"🎉 Processing complete!")
print(f"✅ Successfully processed: {success_count} chunks")
print(f"❌ Failed to process: {error_count} chunks")
print(f"📊 Success rate: {(success_count / len(chunks) * 100):.1f}%")

# Verify the data in Supabase
try:
    count_response = supabase.table("hospital_embeddings").select("id", count="exact").execute()
    total_records = count_response.count if hasattr(count_response, 'count') else len(count_response.data or [])
    print(f"📋 Total records in database: {total_records}")
except Exception as e:
    print(f"⚠️  Could not verify database count: {e}")

print("\n🚀 Ready to test your RAG chatbot!")