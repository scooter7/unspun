import streamlit as st
import requests
from bs4 import BeautifulSoup
import openai
from textblob import TextBlob
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN

# ----------------------------------------------------------------------
# 1. Configuration and Helper Functions
# ----------------------------------------------------------------------

# Set up OpenAI API key from Streamlit secrets.
openai.api_key = st.secrets["OPENAI"]["API_KEY"]

@st.cache_data(show_spinner=False)
def get_headlines(url: str, source: str) -> list:
    """
    Scrapes up to 10 headlines from a given URL using source-specific selectors.
    """
    headlines = []
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        if source == "CNN":
            # CNN: First try to select headlines using the span with 'cd__headline-text'
            elements = soup.select("span.cd__headline-text")
            # If not found, fallback to h3 elements with class 'cd__headline'
            if not elements:
                elements = soup.select("h3.cd__headline")
            headlines = [el.get_text(strip=True) for el in elements][:10]
        
        elif source == "Fox News":
            # Fox News: Use anchor tags with '/story/' in their href
            anchors = soup.find_all("a", href=True)
            filtered_headlines = []
            # Expanded list of unwanted text values
            unwanted = {
                "games hub", "america together", "entertainment", "personal finance", 
                "faith & values", "travel + outdoors", "food + drink", "fox weather",
                "full episodes", "latest wires", "antisemitism exposed", "fox around the world",
                "rss", "world", "opinion", "outkick", "digital originals", "economy",
                "fox news flash", "elections", "personal freedoms"
            }
            for a in anchors:
                href = a.get("href")
                if "/story/" in href:
                    text = a.get_text(strip=True)
                    # Skip if text is too short (likely not a news headline)
                    if len(text) < 15:
                        continue
                    # Skip if the text matches any unwanted phrase (case-insensitive)
                    if text.lower() in unwanted:
                        continue
                    if text not in filtered_headlines:
                        filtered_headlines.append(text)
                if len(filtered_headlines) >= 10:
                    break
            headlines = filtered_headlines[:10]
        
        elif source == "MSNBC":
            # MSNBC: try h2 tags as a starting point.
            elements = soup.find_all("h2")
            headlines = [el.get_text(strip=True) for el in elements][:10]
        
        elif source == "Breitbart":
            # Breitbart: sometimes headlines are in h1; fallback to h2 if needed.
            elements = soup.find_all("h1")
            if len(elements) < 10:
                elements = soup.find_all("h2")
            headlines = [el.get_text(strip=True) for el in elements][:10]
            
    except Exception as e:
        st.error(f"Error fetching headlines from {source} ({url}): {e}")
    return headlines

def perform_sentiment_analysis(text: str) -> float:
    """
    Returns the sentiment polarity of the text using TextBlob.
    """
    analysis = TextBlob(text)
    return analysis.sentiment.polarity

def measure_impact(text: str) -> int:
    """
    Estimates an impact score (0 to 100) based on key trigger words.
    """
    keywords = {
        "global": 100,
        "world": 90,
        "nationwide": 80,
        "government": 70,
        "policy": 60,
        "state": 50,
        "local": 30,
        "individual": 20,
        "family": 20,
        "community": 40,
        "outbreak": 100,
        "crisis": 90,
        "pandemic": 100,
    }
    score = 0
    text_lower = text.lower()
    for key, value in keywords.items():
        if key in text_lower:
            score = max(score, value)
    return score

def get_unbiased_summary(cluster_headlines: list) -> str:
    """
    Uses OpenAI's Chat Completions API with the gpt-4o-mini model
    to generate an unbiased summary from a cluster of headlines.
    """
    prompt = "The following are news headlines from various sources. " \
             "Summarize the overall story in a neutral, unbiased manner, " \
             "removing any partisan or biased language. Use concise language.\n\n"
    for headline in cluster_headlines:
        prompt += f"- {headline}\n"
    prompt += "\nSummary:"
    
    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=150,
            temperature=0.7,
            n=1
        )
        summary = response['choices'][0]['message']['content'].strip()
    except Exception as e:
        summary = "Error generating summary: " + str(e)
    return summary

def cluster_headlines(headlines: list) -> list:
    """
    Clusters headlines using TF-IDF vectorization and DBSCAN.
    Returns a list of clusters (each cluster is a list of headlines).
    Headlines not grouped (noise) are omitted.
    """
    if not headlines:
        return []
    vectorizer = TfidfVectorizer(stop_words="english")
    X = vectorizer.fit_transform(headlines)
    # DBSCAN with cosine distance metric (eps chosen heuristically)
    clustering = DBSCAN(eps=0.5, min_samples=2, metric="cosine").fit(X)
    clusters = {}
    for idx, label in enumerate(clustering.labels_):
        if label == -1:  # noise; ignore individual, non-overlapping headlines
            continue
        clusters.setdefault(label, []).append(headlines[idx])
    return list(clusters.values())

# ----------------------------------------------------------------------
# 2. Main App Function
# ----------------------------------------------------------------------
def main():
    st.title("Unbiased News Aggregator")
    st.write(
        "This app scrapes the latest headlines from CNN, Fox News, MSNBC, and Breitbart. "
        "It performs sentiment analysis, gauges the impact of each story, clusters overlapping "
        "stories, and generates an unbiased summary using the gpt-4o-mini model."
    )

    # Define the news sources and their URLs.
    news_sources = {
        "CNN": "https://www.cnn.com",
        "Fox News": "https://www.foxnews.com",
        "MSNBC": "https://www.msnbc.com",
        "Breitbart": "https://www.breitbart.com"
    }

    st.header("Scraping Headlines")
    all_headlines = {}
    for source, url in news_sources.items():
        with st.spinner(f"Scraping headlines from {source}..."):
            headlines = get_headlines(url, source)
            if not headlines:
                st.warning(f"No headlines fetched for {source}.")
            all_headlines[source] = headlines

    # Build a DataFrame with sentiment and impact metrics.
    data = []
    for source, headlines in all_headlines.items():
        for headline in headlines:
            sentiment = perform_sentiment_analysis(headline)
            impact = measure_impact(headline)
            data.append({
                "Source": source,
                "Headline": headline,
                "Sentiment": sentiment,
                "Impact": impact
            })
    df = pd.DataFrame(data)

    st.header("Headlines with Metrics")
    st.dataframe(df)

    st.subheader("Impact Gauges")
    st.write(
        "Below are individual gauges (0-100) representing the estimated impact score for each headline."
    )
    for idx, row in df.iterrows():
        st.write(f"**{row['Headline']}** (Source: {row['Source']})")
        gauge_value = int(row["Impact"])  # Impact score is in 0-100 range based on keywords.
        st.progress(gauge_value)
        st.write(f"Impact Score: {gauge_value}/100")
        st.write("---")

    # Combine all headlines and perform clustering for overlapping stories.
    st.header("Overlapping Stories & Unbiased Summaries")
    combined_headlines = []
    for headlines in all_headlines.values():
        combined_headlines.extend(headlines)

    clusters = cluster_headlines(combined_headlines)
    if clusters:
        for i, cluster in enumerate(clusters):
            st.markdown(f"### Overlapping Story Cluster {i+1}")
            st.write("**Headlines:**")
            for headline in cluster:
                st.write(f"- {headline}")
            summary = get_unbiased_summary(cluster)
            st.write("**Unbiased Summary:**")
            st.write(summary)
            st.write("---")
    else:
        st.write("No overlapping stories were detected.")

if __name__ == "__main__":
    main()
