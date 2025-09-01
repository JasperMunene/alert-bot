import os
from typing import List, Dict, Any
import json

from flask import Flask, request, jsonify
from flask_restful import Api, Resource
from flask_cors import CORS

import google.generativeai as genai
from supabase import create_client, Client
from dotenv import load_dotenv

# ------------------------------
# Load Environment Variables
# ------------------------------
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "models/embedding-001")
GENERATION_MODEL = os.getenv("GEMINI_TEXT_MODEL", "models/gemini-2.5-flash")
TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "5"))
SIMILARITY_FLOOR = float(os.getenv("SIMILARITY_FLOOR", "0.25"))

if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("Supabase credentials missing. Set SUPABASE_URL and SUPABASE_KEY in .env")
if not GEMINI_API_KEY:
    raise RuntimeError("Gemini API key missing. Set GEMINI_API_KEY in .env")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
genai.configure(api_key=GEMINI_API_KEY)

CHAT_FILE = "chat_history.json"


# ------------------------------
# Chat History Persistence
# ------------------------------
def load_chat_history() -> Dict[str, List[Dict[str, str]]]:
    if os.path.exists(CHAT_FILE):
        with open(CHAT_FILE, "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return {}
    return {}

def save_chat_history(history: Dict[str, List[Dict[str, str]]]) -> None:
    with open(CHAT_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

CHAT_HISTORY: Dict[str, List[Dict[str, str]]] = load_chat_history()


# ------------------------------
# Retrieval Helpers
# ------------------------------
def embed_text(text: str) -> List[float]:
    """Generate embedding for query text using Gemini"""
    try:
        resp = genai.embed_content(
            model=EMBEDDING_MODEL, 
            content=text,
            task_type="retrieval_query"  # Use query task type for search queries
        )
        return resp["embedding"]
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return []

def retrieve_context(query: str, top_k: int = TOP_K) -> List[Dict[str, Any]]:
    """Retrieve relevant context from Supabase using vector similarity"""
    query_embedding = embed_text(query)
    
    if not query_embedding:
        print("Failed to generate query embedding")
        return []
    
    try:
        # First, try the RPC function for vector similarity search
        response = supabase.rpc(
            "match_hospital_knowledge",
            {
                "query_embedding": query_embedding, 
                "match_count": top_k,
                "similarity_threshold": SIMILARITY_FLOOR
            }
        ).execute()
        
        rows: List[Dict[str, Any]] = response.data or []
        print(f"RPC returned {len(rows)} rows")
        
        # If no results from vector search, fallback to keyword search
        if not rows:
            print("No vector results, trying keyword fallback...")
            keyword_results = fallback_keyword_search(query, top_k)
            if keyword_results:
                print(f"Keyword fallback returned {len(keyword_results)} results")
                return keyword_results
        
        # Process the results to ensure consistent structure
        processed_rows = []
        for row in rows:
            processed_row = {
                'id': row.get('id'),
                'content': row.get('content', ''),
                'similarity': row.get('similarity', 0),
                'section': row.get('section', 'General Info'),
                'metadata': {
                    'section': row.get('section', 'General Info'),
                    'category': row.get('category', 'general'),
                    'page': row.get('page')
                }
            }
            processed_rows.append(processed_row)
        
        # Lower the similarity threshold for debugging
        debug_threshold = max(0.1, SIMILARITY_FLOOR)
        filtered_rows = [row for row in processed_rows if row.get("similarity", 0) >= debug_threshold]
        
        print(f"Retrieved {len(filtered_rows)} relevant chunks (threshold: {debug_threshold}) for query: '{query}'")
        for row in filtered_rows:
            print(f"  - {row.get('id')}: {row.get('similarity', 0):.3f} - {row.get('content', '')[:100]}...")
        
        return filtered_rows
        
    except Exception as e:
        print(f"Error retrieving context: {e}")
        # Fallback to keyword search on error
        return fallback_keyword_search(query, top_k)

def fallback_keyword_search(query: str, top_k: int = TOP_K) -> List[Dict[str, Any]]:
    """Fallback keyword-based search when vector search fails"""
    try:
        # Split query into keywords
        keywords = [word.lower() for word in query.split() if len(word) > 2]
        print(f"Searching for keywords: {keywords}")
        
        # Use text search on content
        search_results = []
        for keyword in keywords:
            try:
                response = supabase.table("hospital_embeddings").select("*").ilike("content", f"%{keyword}%").limit(top_k).execute()
                if response.data:
                    search_results.extend(response.data)
            except Exception as e:
                print(f"Keyword search error for '{keyword}': {e}")
        
        # Remove duplicates and format
        seen_ids = set()
        unique_results = []
        for row in search_results:
            if row.get('id') not in seen_ids:
                seen_ids.add(row.get('id'))
                processed_row = {
                    'id': row.get('id'),
                    'content': row.get('content', ''),
                    'similarity': 0.5,  # Default similarity for keyword matches
                    'section': row.get('section', 'General Info'),
                    'metadata': {
                        'section': row.get('section', 'General Info'),
                        'category': row.get('category', 'general'),
                        'page': row.get('page')
                    }
                }
                unique_results.append(processed_row)
        
        print(f"Keyword search returned {len(unique_results)} unique results")
        return unique_results[:top_k]
        
    except Exception as e:
        print(f"Fallback keyword search failed: {e}")
        return []

def format_context(chunks: List[Dict[str, Any]]) -> str:
    """Format retrieved chunks into context string"""
    if not chunks:
        return "No relevant information found in the knowledge base."
    
    lines = []
    for i, chunk in enumerate(chunks, 1):
        section = chunk.get("section", "General Info")
        chunk_id = chunk.get("id", f"chunk_{i}")
        content = chunk.get("content", "").strip()
        similarity = chunk.get("similarity", 0)
        
        lines.append(f"[Source {i}: {section} | ID: {chunk_id} | Relevance: {similarity:.3f}]\n{content}")
    
    return "\n\n".join(lines)


# ------------------------------
# Prompt Engineering
# ------------------------------
SYSTEM_INSTRUCTIONS = """You are a professional, friendly virtual assistant for ALERT COMPREHENSIVE SPECIALIZED HOSPITAL.

IMPORTANT GUIDELINES:
- Answer questions based ONLY on the provided SOURCES about hospital services, departments, contact information, and policies
- If the answer cannot be found in the SOURCES, politely say: "I don't have that specific information in my knowledge base. Please contact the hospital directly at +251 113 47 16 32 for assistance."
- Never make up information about medical treatments, procedures, or hospital details
- Be concise, helpful, and empathetic in your responses
- If asked about medical advice, remind users to consult with healthcare professionals

RESPONSE FORMAT:
- Provide direct answers based on the sources
- Be conversational but professional
- Include relevant contact information when appropriate"""

def generate_answer(user_query: str, context_text: str, chat_id: str = None) -> str:
    """Generate response using Gemini with retrieved context"""
    try:
        model = genai.GenerativeModel(
            model_name=GENERATION_MODEL,
            system_instruction=SYSTEM_INSTRUCTIONS
        )
        
        # Build context-aware prompt
        prompt_parts = [
            "SOURCES:",
            context_text,
            "\n---\n"
        ]
        
        # Add recent conversation history for context
        if chat_id and chat_id in CHAT_HISTORY:
            recent_history = CHAT_HISTORY[chat_id][-3:]  # Last 3 exchanges
            if recent_history:
                prompt_parts.append("RECENT CONVERSATION:")
                for turn in recent_history:
                    prompt_parts.append(f"User: {turn['user']}")
                    prompt_parts.append(f"Assistant: {turn['assistant']}")
                prompt_parts.append("\n---\n")
        
        prompt_parts.extend([
            f"USER QUESTION: {user_query}",
            "\nPlease answer based on the SOURCES provided above. If the information is not available in the sources, direct the user to contact the hospital directly."
        ])
        
        full_prompt = "\n".join(prompt_parts)
        
        # Generate response
        response = model.generate_content(full_prompt)
        
        if response and hasattr(response, 'text') and response.text:
            return response.text.strip()
        else:
            return "I apologize, but I'm having difficulty processing your request right now. Please try again or contact the hospital directly."
            
    except Exception as e:
        print(f"Error generating response: {e}")
        return "I'm experiencing technical difficulties. Please try again shortly or contact the hospital at +251 113 47 16 32."


# ------------------------------
# Flask API
# ------------------------------
app = Flask(__name__)
CORS(app, origins=["*"])  # Configure CORS properly for production
api = Api(app)

class Health(Resource):
    def get(self):
        return {
            "status": "ok", 
            "service": "alert-hospital-assistant",
            "version": "1.0",
            "embedding_model": EMBEDDING_MODEL,
            "generation_model": GENERATION_MODEL
        }

class Chat(Resource):
    def post(self):
        try:
            payload = request.get_json(force=True) or {}
            user_query: str = (payload.get("message") or payload.get("query") or "").strip()
            chat_id: str = payload.get("chat_id") or "default"
            top_k: int = int(payload.get("top_k") or TOP_K)

            if not user_query:
                return jsonify({"error": "Missing 'message' or 'query' in JSON body."}), 400

            print(f"Processing query: {user_query}")

            # Retrieve hospital knowledge
            chunks = retrieve_context(user_query, top_k=top_k)
            context_text = format_context(chunks)

            print(f"Context retrieved: {len(chunks)} chunks")

            # Generate grounded answer
            answer = generate_answer(user_query, context_text, chat_id=chat_id)

            # Update chat history
            CHAT_HISTORY.setdefault(chat_id, []).append({
                "user": user_query,
                "assistant": answer,
            })
            save_chat_history(CHAT_HISTORY)

            # Prepare response
            response_data = {
                "answer": answer,
                "sources": [
                    {
                        "id": chunk.get("id"),
                        "section": chunk.get("section"),
                        "similarity": round(chunk.get("similarity", 0), 3),
                        "metadata": chunk.get("metadata", {})
                    }
                    for chunk in chunks
                ],
                "used_context": bool(chunks),
                "num_sources": len(chunks)
            }

            return jsonify(response_data)

        except Exception as e:
            print(f"Error in chat endpoint: {e}")
            return jsonify({
                "error": "Internal server error",
                "answer": "I apologize, but I'm experiencing technical difficulties. Please contact the hospital directly at +251 113 47 16 32."
            }), 500

class DebugRetrieval(Resource):
    """Debug endpoint to test retrieval without generation"""
    def post(self):
        try:
            payload = request.get_json(force=True) or {}
            user_query: str = (payload.get("message") or payload.get("query") or "").strip()
            top_k: int = int(payload.get("top_k") or TOP_K)

            if not user_query:
                return jsonify({"error": "Missing 'message' or 'query' in JSON body."}), 400

            print(f"Debug query: {user_query}")
            
            # Test embedding generation
            embedding = embed_text(user_query)
            print(f"Generated embedding: {len(embedding) if embedding else 0} dimensions")

            # Check database connection
            try:
                count_response = supabase.table("hospital_embeddings").select("id", count="exact").execute()
                total_records = count_response.count if hasattr(count_response, 'count') else len(count_response.data or [])
                print(f"Total records in database: {total_records}")
            except Exception as e:
                print(f"Database connection error: {e}")
                return jsonify({"error": f"Database connection failed: {str(e)}"}), 500

            # Try retrieval
            chunks = retrieve_context(user_query, top_k=top_k)
            context_text = format_context(chunks)

            # Also try direct database query for comparison
            try:
                all_records = supabase.table("hospital_embeddings").select("id,content,section,category").limit(5).execute()
                sample_data = all_records.data or []
            except Exception as e:
                sample_data = []
                print(f"Error fetching sample data: {e}")

            return jsonify({
                "query": user_query,
                "embedding_dimensions": len(embedding) if embedding else 0,
                "chunks_found": len(chunks),
                "context": context_text,
                "raw_chunks": chunks,
                "database_records": total_records if 'total_records' in locals() else "unknown",
                "sample_data": sample_data[:3],  # First 3 records for inspection
                "similarity_threshold": SIMILARITY_FLOOR
            })

        except Exception as e:
            print(f"Error in debug endpoint: {e}")
            import traceback
            traceback.print_exc()
            return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500

# Add resources to API
api.add_resource(Health, "/health")
api.add_resource(Chat, "/chat")
api.add_resource(DebugRetrieval, "/debug")

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8080"))
    debug_mode = os.getenv("DEBUG", "true").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug_mode)