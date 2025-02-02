import streamlit as st
import pandas as pd
from fuzzywuzzy import fuzz
from io import BytesIO

# Streamlit App Title
st.title("🔍 Content Audit - Page Merge Tool")

# File Upload
uploaded_file = st.file_uploader("Upload your content audit CSV", type=["csv"])

# Set similarity threshold slider
threshold = st.slider("Title Similarity Threshold (%)", 50, 100, 70)

# Function to extract the last part of the URL (slug)
def get_slug(url):
    return url.rstrip('/').split('/')[-1]

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Ensure necessary columns exist
    required_columns = ["url", "title_tag", "organic_sessions", "pageviews", "engagement_rate", "dofollow_backlinks", "pubdate"]
    missing_cols = [col for col in required_columns if col not in df.columns]

    if missing_cols:
        st.error(f"Missing columns: {', '.join(missing_cols)}. Please upload a valid CSV.")
    else:
        # Convert pubdate to datetime
        df["pubdate"] = pd.to_datetime(df["pubdate"], errors="coerce")

        # Add slug column
        df["slug"] = df["url"].apply(get_slug)

        # Store merge candidates
        merge_candidates = []

        # Compare each page with every other page
        for i, row1 in df.iterrows():
            for j, row2 in df.iterrows():
                if i >= j:  # Avoid duplicate comparisons
                    continue

                title_similarity = fuzz.ratio(str(row1["title_tag"]), str(row2["title_tag"]))
                url_similarity = fuzz.ratio(str(row1["slug"]), str(row2["slug"]))

                if title_similarity >= threshold or url_similarity >= threshold:
                    # Calculate performance score
                    page1_score = row1["organic_sessions"] + row1["pageviews"] + row1["engagement_rate"] + row1["dofollow_backlinks"]
                    page2_score = row2["organic_sessions"] + row2["pageviews"] + row2["engagement_rate"] + row2["dofollow_backlinks"]

                    # Choose the primary and secondary page
                    if page1_score > page2_score:
                        primary_page, secondary_page = row1, row2
                    elif page2_score > page1_score:
                        primary_page, secondary_page = row2, row1
                    else:  # If scores are equal, pick the newer pubdate
                        primary_page, secondary_page = (row1, row2) if row1["pubdate"] > row2["pubdate"] else (row2, row1)

                    merge_candidates.append({
                        "Primary Title": primary_page["title_tag"],
                        "Primary URL": primary_page["url"],
                        "Secondary Title": secondary_page["title_tag"],
                        "Secondary URL": secondary_page["url"],
                        "Title Similarity": title_similarity,
                        "URL Similarity": url_similarity,
                        "Primary Page Score": page1_score,
                        "Secondary Page Score": page2_score
                    })

        # Convert results to DataFrame
        merge_df = pd.DataFrame(merge_candidates)

        if not merge_df.empty:
            # Show results in Streamlit
            st.write("### 📊 Potential Merge Candidates")
            st.dataframe(merge_df)

            # Provide a download button for CSV
            output = BytesIO()
            merge_df.to_csv(output, index=False)
            output.seek(0)
            st.download_button("📥 Download Merge Candidates CSV", output, "merge_candidates.csv", "text/csv")
        else:
            st.info("No merge candidates found based on the selected similarity threshold.")
