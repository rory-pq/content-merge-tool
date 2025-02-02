import streamlit as st
import pandas as pd
from fuzzywuzzy import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def extract_first_words(title, num_words=4):
    return " ".join(title.split()[:num_words]).lower()

def identify_merge_candidates(df, title_threshold=80, first_words_threshold=85, topic_threshold=0.7):
    df = df.dropna(subset=["url", "title_tag", "organic_sessions", "pageviews", "pubdate"])
    df["pubdate"] = pd.to_datetime(df["pubdate"], errors="coerce")
    df["first_words"] = df["title_tag"].apply(lambda x: extract_first_words(x, num_words=4))
    
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(df["title_tag"].str.lower())
    cosine_sim = cosine_similarity(tfidf_matrix)
    
    merge_candidates = []
    for i in range(len(df)):
        for j in range(i + 1, len(df)):
            full_title_similarity = fuzz.ratio(df.iloc[i]["title_tag"], df.iloc[j]["title_tag"])
            first_words_similarity = fuzz.ratio(df.iloc[i]["first_words"], df.iloc[j]["first_words"])
            topic_similarity = cosine_sim[i, j]
            
            if first_words_similarity >= first_words_threshold and (full_title_similarity >= title_threshold or topic_similarity >= topic_threshold):
                page1_score = df.iloc[i]["organic_sessions"] + df.iloc[i]["pageviews"]
                page2_score = df.iloc[j]["organic_sessions"] + df.iloc[j]["pageviews"]
                
                if page1_score > page2_score:
                    primary_page, secondary_page = df.iloc[i], df.iloc[j]
                elif page2_score > page1_score:
                    primary_page, secondary_page = df.iloc[j], df.iloc[i]
                else:
                    primary_page, secondary_page = (df.iloc[i], df.iloc[j]) if df.iloc[i]["pubdate"] > df.iloc[j]["pubdate"] else (df.iloc[j], df.iloc[i])
                
                merge_candidates.append({
                    "Primary Title": primary_page["title_tag"],
                    "Primary URL": primary_page["url"],
                    "Secondary Title": secondary_page["title_tag"],
                    "Secondary URL": secondary_page["url"],
                    "First Words Similarity": first_words_similarity,
                    "Title Similarity": full_title_similarity,
                    "Topic Similarity": topic_similarity,
                    "Primary Page Score": page1_score,
                    "Secondary Page Score": page2_score
                })
    
    return pd.DataFrame(merge_candidates)

st.title("🔍 Content Audit - Page Merge Tool")
uploaded_file = st.file_uploader("Upload your content audit CSV", type=["csv"])

title_threshold = st.slider("Title Similarity Threshold (%)", 50, 100, 80)
first_words_threshold = st.slider("First Words Similarity Threshold (%)", 50, 100, 85)
topic_threshold = st.slider("Topic Similarity Threshold (0-1)", 0.0, 1.0, 0.7)

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    merge_df = identify_merge_candidates(df, title_threshold, first_words_threshold, topic_threshold)
    
    if not merge_df.empty:
        st.write("### 📊 Potential Merge Candidates")
        st.dataframe(merge_df)
        
        output_csv = merge_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Merge Candidates CSV", output_csv, "merge_candidates.csv", "text/csv")
    else:
        st.info("No merge candidates found based on the selected similarity thresholds.")
