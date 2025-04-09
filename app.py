import streamlit as st
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import os
import requests
from dotenv import load_dotenv

# Load environment variables (for local testing)
load_dotenv()

# Set page config
st.set_page_config(page_title="SHL Assessment Recommender", layout="centered")

# Load resources with caching
@st.cache_resource
def load_models():
    # Load embedding model
    model = SentenceTransformer("all-MiniLM-L6-v2")
    # Load FAISS index
    index = faiss.read_index("shl_index.faiss")
    # Load dataframe
    with open("shl_dataframe.pkl", "rb") as f:
        df = pickle.load(f)
    return model, index, df
# Load Groq API key (use Streamlit secrets in production)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama3-8b-8192"

model, index, df = load_models()

# Helper: retrieve candidates
def retrieve_candidates(query, top_k=20):
    q_emb = model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(q_emb)
    D, I = index.search(q_emb, top_k)
    candidates = []
    for score, idx in zip(D[0], I[0]):
        row = df.iloc[idx]
        candidates.append({
            "name": row["name"],
            "link": row["link"],
            "description": row["description"],
            "duration": row["duration"],
            "test_types": row["test_types"],
            "remote_testing": row["remote_testing"],
            "adaptive_irt": row["adaptive_irt"]
        })
    return candidates

# Helper: GROQ rerank
def groq_rerank(query, candidates, rerank_k=10):
    items = "\n".join(
        f"{i+1}. {c['name']} — {c['description'][:100]} (Duration: {c['duration']} mins)"
        for i, c in enumerate(candidates)
    )
    prompt = (
        f"You are an expert assessment recommender. The hiring need is:\n\n"
        f"“{query}”\n\n"
        f"Here are {len(candidates)} candidate assessments:\n{items}\n\n"
        f"Please rank the top {rerank_k} most relevant assessments by returning a comma-separated list "
        f"of their numbers in descending order of relevance (e.g., 3,1,5,...)."
    )

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant that ranks assessment tests from a given list."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.0
    }
    
    try:
        response = requests.post(GROQ_URL, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        text = result["choices"][0]["message"]["content"]
        indices = [int(x)-1 for x in text.split(",") if x.strip().isdigit()]
    except Exception as e:
        st.error(f"Error during reranking: {str(e)}")
        indices = list(range(min(rerank_k, len(candidates))))
    
    return [candidates[i] for i in indices[:rerank_k]]

# Streamlit UI
st.title("🧠 SHL Assessment Recommender")
st.markdown("Find the right SHL test by describing the job or skills you’re hiring for.")

query = st.text_input("Enter your role or test requirement:")
rerank_k = st.slider("Number of results to show", 1, 20, 5)

if st.button("🔍 Get Recommendations"):
    if not query:
        st.warning("Please enter a query.")
    else:
        with st.spinner("Searching assessments..."):
            try:
                # First-stage retrieval
                candidates = retrieve_candidates(query, top_k=20)
                
                if not candidates:
                    st.info("No initial candidates found.")
                    st.stop()
                
                # Second-stage reranking
                results = groq_rerank(query, candidates, rerank_k=rerank_k)
                
                if results:
                    st.success(f"Found {len(results)} recommendations!")
                    for item in results:
                        st.markdown("---")
                        st.markdown(f"### 🔹 [{item['name']}]({item['link']})")
                        st.markdown(f"**Test Type:** {item.get('test_types', 'N/A')}")
                        st.markdown(f"**Duration:** {item.get('duration', 'N/A')} minutes")
                        st.markdown(f"**Remote Testing Support:** {'✅ Yes' if item.get('remote_testing') else '❌ No'}")
                        st.markdown(f"**Adaptive/IRT Support:** {'✅ Yes' if item.get('adaptive_irt') else '❌ No'}")
                        st.markdown(f"**Description:** {item['description']}")
                else:
                    st.info("No recommendations found after reranking.")
            
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
