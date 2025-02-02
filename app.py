import streamlit as st
import pandas as pd
from rapidfuzz.fuzz import token_set_ratio
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util

# Load a better NLP model for topic comparison
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

def extract_first_words(text, num_words=4):
    """Get the first few words in lower case."""
    return " ".join(text.split()[:num_words]).lower()

def identify_merge_candidates(df, title_threshold=80, first_words_threshold=85, meta_threshold=80, topic_threshold=0.8):
    required_cols = ["url", "title_tag", "meta_description", "organic_sessions", "pageviews", "pubdate", "engagement_rate"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        st.error(f"Missing columns: {', '.join(missing_cols)}")
        return pd.DataFrame()
    
    df = df.dropna(subset=["url", "title_tag", "organic_sessions", "pageviews", "pubdate"])
    
    # Convert pubdate to a consistent format
    df["pubdate"] = pd.to_datetime(df["pubdate"], errors="coerce", format='%m/%d/%Y')
    
    df["title_tag"] = df["title_tag"].fillna("").astype(str).str.lower().str.strip()
    df["meta_description"] = df["meta_description"].fillna("").astype(str).str.lower().str.strip()
    df["first_words"] = df["title_tag"].apply(lambda x: extract_first_words(x, num_words=4))
    df["combined_text"] = df["title_tag"] + " " + df["meta_description"]

    # Generate embeddings for topic comparison
    embeddings = sbert_model.encode(df["combined_text"].tolist(), convert_to_tensor=True)
    
    merge_candidates = []
    n = len(df)
    total_comparisons = (n * (n - 1)) // 2
    progress_count = 0
    progress_bar = st.progress(0)

    for i in range(n):
        for j in range(i + 1, n):
            title_similarity = token_set_ratio(df.iloc[i]["title_tag"], df.iloc[j]["title_tag"])
            meta_similarity = token_set_ratio(df.iloc[i]["meta_description"], df.iloc[j]["meta_description"])
            first_words_similarity = token_set_ratio(df.iloc[i]["first_words"], df.iloc[j]["first_words"])
            topic_similarity = float(util.pytorch_cos_sim(embeddings[i], embeddings[j]))

            if first_words_similarity >= first_words_threshold and (
                title_similarity >= title_threshold or 
                meta_similarity >= meta_threshold) and topic_similarity >= topic_threshold:
                
                page1_score = df.iloc[i]["organic_sessions"] + df.iloc[i]["pageviews"]
                page2_score = df.iloc[j]["organic_sessions"] + df.iloc[j]["pageviews"]

                # If scores are too close, use engagement rate as a tiebreaker
                if abs(page1_score - page2_score) < 10:
                    page1_score += df.iloc[i]["engagement_rate"] * 100
                    page2_score += df.iloc[j]["engagement_rate"] * 100
                
                if page1_score > page2_score:
                    primary_page, secondary_page = df.iloc[i], df.iloc[j]
                else:
                    primary_page, secondary_page = df.iloc[j], df.iloc[i]
                
                merge_candidates.append({
                    "Primary Title": primary_page["title_tag"],
                    "Primary URL": primary_page["url"],
                    "Secondary Title": secondary_page["title_tag"],
                    "Secondary URL": secondary_page["url"],
                    "First Words Similarity": first_words_similarity,
                    "Title Similarity": title_similarity,
                    "Meta Description Similarity": meta_similarity,
                    "Topic Similarity": topic_similarity,
                    "Primary Page Score": page1_score,
                    "Secondary Page Score": page2_score
                })
            progress_count += 1
        if total_comparisons > 0:
            progress_bar.progress(progress_count / total_comparisons)
    
    return pd.DataFrame(merge_candidates)

st.title("🔍 Content Audit - Page Merge Tool")

uploaded_file = st.file_uploader("Upload your content audit CSV", type=["csv"])

title_threshold = st.slider("Title Similarity Threshold (%)", 50, 100, 80)
first_words_threshold = st.slider("First Words Similarity Threshold (%)", 50, 100, 85)
meta_threshold = st.slider("Meta Description Similarity Threshold (%)", 50, 100, 80)
topic_threshold = st.slider("Topic Similarity Threshold (0-1)", 0.0, 1.0, 0.8)

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
    except Exception:
        st.error("Error reading CSV file.")
        st.stop()
    
    with st.spinner("Processing merge candidates..."):
        merge_df = identify_merge_candidates(df, title_threshold, first_words_threshold, meta_threshold, topic_threshold)
    
    if not merge_df.empty:
        st.success("Merge candidates found!")
        st.write("### 📊 Potential Merge Candidates")
        st.dataframe(merge_df)
        
        output_csv = merge_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Merge Candidates CSV", output_csv, "merge_candidates.csv", "text/csv")
    else:
        st.info("No merge candidates found based on the selected thresholds.")
