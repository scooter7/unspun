import streamlit as st
import requests
from bs4 import BeautifulSoup
import openai
from textblob import TextBlob
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
import plotly.graph_objects as go
import datetime
import email.utils

# ----------------------------------------------------------------------
# 1. Configuration and Helper Functions
# ----------------------------------------------------------------------

# Set up OpenAI API key from Streamlit secrets.
openai.api_key = st.secrets["OPENAI"]["API_KEY"]

def filter_recent_items(items, max_age_hours=24):
    """Filter RSS items to include only those published within max_age_hours."""
    now = datetime.datetime.now(datetime.timezone.utc)
    threshold = datetime.timedelta(hours=max_age_hours)
    recent_items = []
    for item in items:
        if item.pubDate:
            try:
                pub_date = email.utils.parsedate_to_datetime(item.pubDate.get_text())
                if (now - pub_date) <= threshold:
                    recent_items.append(item)
            except Exception:
                recent_items.append(item)  # If date parsing fails, include the item.
        else:
            recent_items.append(item)
    return recent_items

@st.cache_data(show_spinner=False)
def get_headlines(url: str, source: str) -> list:
    """
    Fetches up to 10 headlines for a given source.
    For CNN and Fox News, uses their RSS feeds with the lxml-xml parser,
    and filters for stories published within the last 24 hours.
    For MSNBC and Breitbart, attempts to scrape from the provided URL.
    """
    headlines = []
    try:
        if source in ["CNN", "Fox News"]:
            # Use RSS feeds with the lxml-xml parser.
            rss_url = (
                "http://rss.cnn.com/rss/edition.rss"
                if source == "CNN"
                else "https://feeds.foxnews.com/foxnews/latest"
            )
            response = requests.get(rss_url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, "lxml-xml")
            items = soup.find_all("item")
            # Filter to include only recent items (last 24 hours)
            recent_items = filter_recent_items(items, max_age_hours=24)
            # Limit to 10 items.
            items = recent_items[:10]
            headlines = [item.title.get_text(strip=True) for item in items if item.title]
        
        elif source == "MSNBC":
            # Scrape from the main page using h2 tags.
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            elements = soup.find_all("h2")
            headlines = [el.get_text(strip=True) for el in elements][:10]
        
        elif source == "Breitbart":
            # Scrape from the main page: try h1 tags, fallback to h2.
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            elements = soup.find_all("h1")
            if len(elements) < 10:
                elements = soup.find_all("h2")
            headlines = [el.get_text(strip=True) for el in elements][:10]
            
    except Exception as e:
        st.error(f"Error fetching headlines from {source}: {e}")
    return headlines

def perform_sentiment_analysis(text: str) -> float:
    """Returns the sentiment polarity of the text using TextBlob."""
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
            messages=[{"role": "user", "content": prompt}],
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
    clustering = DBSCAN(eps=0.5, min_samples=2, metric="cosine").fit(X)
    clusters = {}
    for idx, label in enumerate(clustering.labels_):
        if label == -1:
            continue
        clusters.setdefault(label, []).append(headlines[idx])
    return list(clusters.values())

# ----------------------------------------------------------------------
# 2. Main App Function
# ----------------------------------------------------------------------
def main():
    st.title("Unbiased News Aggregator")
    st.write(
        "This app fetches the latest headlines from CNN, Fox News, MSNBC, and Breitbart. "
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

    st.header("Fetching Headlines")
    all_headlines = {}
    for source, url in news_sources.items():
        with st.spinner(f"Fetching headlines from {source}..."):
            headlines = get_headlines(url, source)
            if not headlines:
                st.warning(f"No headlines fetched for {source}.")
            all_headlines[source] = headlines

    # Process headlines to compute sentiment and impact.
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
    # Display each headline with its sentiment gauge and impact gauge.
    for idx, row in df.iterrows():
        st.markdown(f"**{row['Headline']}**  _(Source: {row['Source']})_")
        col1, col2 = st.columns(2)
        # Sentiment gauge in the first column.
        with col1:
            # Transform sentiment (-1 to 1) to a 0 to 100 scale.
            transformed_sentiment = (row["Sentiment"] + 1) * 50
            sentiment_fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=transformed_sentiment,
                title={"text": "Sentiment"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "black"},
                    "steps": [
                        {"range": [0, 33], "color": "red"},
                        {"range": [33, 66], "color": "yellow"},
                        {"range": [66, 100], "color": "green"}
                    ],
                    "threshold": {
                        "line": {"color": "black", "width": 4},
                        "thickness": 0.75,
                        "value": transformed_sentiment
                    }
                }
            ))
            st.plotly_chart(sentiment_fig, use_container_width=True, key=f"sentiment_{idx}")
        # Impact gauge in the second column (using a progress bar).
        with col2:
            impact_value = int(row["Impact"])
            st.progress(impact_value / 100)
            st.write(f"Impact Score: {impact_value}/100")
        st.markdown("---")
    
    # Cluster headlines for overlapping stories.
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
