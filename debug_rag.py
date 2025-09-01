import os
import google.generativeai as genai
from supabase import create_client
from dotenv import load_dotenv
import json

# Load environment variables
load_dotenv()

# Config
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not all([SUPABASE_URL, SUPABASE_KEY, GEMINI_API_KEY]):
    print("❌ Missing environment variables")
    exit(1)

# Initialize clients
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
genai.configure(api_key=GEMINI_API_KEY)

def test_database_connection():
    """Test basic database connection"""
    print("🔍 Testing database connection...")
    try:
        response = supabase.table("hospital_embeddings").select("id", count="exact").execute()
        total_records = response.count if hasattr(response, 'count') else len(response.data or [])
        print(f"✅ Database connected. Total records: {total_records}")
        return True
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return False

def test_embeddings():
    """Test if embeddings exist in database"""
    print("\n🔍 Testing embeddings...")
    try:
        # Get a sample record with embedding
        response = supabase.table("hospital_embeddings").select("id,content,embedding").limit(1).execute()
        if response.data:
            record = response.data[0]
            embedding = record.get('embedding')
            if embedding:
                print(f"✅ Found embedding for record {record['id']}: {len(embedding)} dimensions")
                return True
            else:
                print(f"❌ Record {record['id']} has no embedding")
        else:
            print("❌ No records found in database")
    except Exception as e:
        print(f"❌ Error checking embeddings: {e}")
    return False

def test_vector_function():
    """Test the vector similarity function"""
    print("\n🔍 Testing vector similarity function...")
    try:
        # Generate a test embedding
        test_query = "when was hospital established"
        result = genai.embed_content(
            model="models/embedding-001",
            content=test_query,
            task_type="retrieval_query"
        )
        query_embedding = result["embedding"]
        print(f"✅ Generated query embedding: {len(query_embedding)} dimensions")
        
        # Test the RPC function
        response = supabase.rpc(
            "match_hospital_knowledge_debug",  # Use debug version without threshold
            {
                "query_embedding": query_embedding,
                "match_count": 10
            }
        ).execute()
        
        results = response.data or []
        print(f"✅ RPC function returned {len(results)} results")
        
        for i, result in enumerate(results[:3]):
            similarity = result.get('similarity', 0)
            content = result.get('content', '')[:100]
            print(f"  {i+1}. Similarity: {similarity:.4f} - {content}...")
            
        return len(results) > 0
        
    except Exception as e:
        print(f"❌ Vector function test failed: {e}")
        return False

def test_keyword_search():
    """Test keyword-based search as fallback"""
    print("\n🔍 Testing keyword search...")
    try:
        keywords = ["established", "1934", "hospital"]
        for keyword in keywords:
            response = supabase.table("hospital_embeddings").select("*").ilike("content", f"%{keyword}%").limit(3).execute()
            results = response.data or []
            print(f"  Keyword '{keyword}': {len(results)} matches")
            if results:
                for result in results[:1]:
                    content = result.get('content', '')[:100]
                    print(f"    - {result.get('id')}: {content}...")
        return True
    except Exception as e:
        print(f"❌ Keyword search failed: {e}")
        return False

def test_specific_query():
    """Test the specific query that's failing"""
    print("\n🔍 Testing specific failing query...")
    query = "When was the hospital started"
    
    try:
        # Generate embedding
        result = genai.embed_content(
            model="models/embedding-001",
            content=query,
            task_type="retrieval_query"
        )
        query_embedding = result["embedding"]
        
        # Test with very low threshold
        response = supabase.rpc(
            "match_hospital_knowledge",
            {
                "query_embedding": query_embedding,
                "match_count": 10,
                "similarity_threshold": 0.0  # No threshold
            }
        ).execute()
        
        results = response.data or []
        print(f"✅ Query '{query}' returned {len(results)} results")
        
        # Look for the specific chunk we expect
        for result in results:
            if "1934" in result.get('content', '') or "established" in result.get('content', '').lower():
                print(f"✅ Found relevant chunk: {result.get('id')} (similarity: {result.get('similarity', 0):.4f})")
                print(f"   Content: {result.get('content', '')}")
                break
        else:
            print("❌ Didn't find the expected chunk about 1934/establishment")
            
    except Exception as e:
        print(f"❌ Specific query test failed: {e}")

def show_sample_data():
    """Show sample data from database"""
    print("\n🔍 Sample data in database:")
    try:
        response = supabase.table("hospital_embeddings").select("id,content,section,category").limit(5).execute()
        data = response.data or []
        for i, record in enumerate(data, 1):
            print(f"{i}. {record.get('id')} - {record.get('section')} - {record.get('content', '')[:80]}...")
    except Exception as e:
        print(f"❌ Error fetching sample data: {e}")

def main():
    print("🚀 RAG Chatbot Diagnostics")
    print("=" * 50)
    
    # Run all tests
    db_ok = test_database_connection()
    if not db_ok:
        print("❌ Cannot proceed without database connection")
        return
    
    embeddings_ok = test_embeddings()
    vector_ok = test_vector_function()
    keyword_ok = test_keyword_search()
    
    show_sample_data()
    test_specific_query()
    
    print("\n" + "=" * 50)
    print("📊 SUMMARY:")
    print(f"Database Connection: {'✅' if db_ok else '❌'}")
    print(f"Embeddings Present: {'✅' if embeddings_ok else '❌'}")
    print(f"Vector Search: {'✅' if vector_ok else '❌'}")
    print(f"Keyword Search: {'✅' if keyword_ok else '❌'}")
    
    if not vector_ok:
        print("\n💡 RECOMMENDATIONS:")
        print("1. Check if pgvector extension is enabled: CREATE EXTENSION vector;")
        print("2. Verify the RPC function exists and has correct signature")
        print("3. Re-run the embedding script to ensure embeddings are properly stored")
        print("4. Check Supabase logs for any errors")

if __name__ == "__main__":
    main()