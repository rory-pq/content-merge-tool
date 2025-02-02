import pandas as pd
from fuzzywuzzy import fuzz

# Load CSV file
file_path = "your_file.csv"  # Update with your actual file path
df = pd.read_csv(file_path)

# Ensure required columns exist
required_columns = ["url", "title_tag", "organic_sessions", "pageviews", "engagement_rate", "dofollow_backlinks", "pubdate"]
df = df[[col for col in required_columns if col in df.columns]].dropna()

# Convert pubdate to datetime (if available)
if "pubdate" in df.columns:
    df["pubdate"] = pd.to_datetime(df["pubdate"], errors="coerce")

# Function to extract the last part of the URL (slug)
def get_slug(url):
    return url.rstrip('/').split('/')[-1]

df["slug"] = df["url"].apply(get_slug)

# Store merge candidates
merge_candidates = []
threshold = 70  # Similarity threshold

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
                "Primary Page Score": page1_score if primary_page is row1 else page2_score,
                "Secondary Page Score": page2_score if primary_page is row1 else page1_score
            })

# Convert results to DataFrame and save
merge_df = pd.DataFrame(merge_candidates)
merge_df.to_csv("merge_candidates.csv", index=False)

print("✅ Merge analysis complete! Results saved in 'merge_candidates.csv'.")

