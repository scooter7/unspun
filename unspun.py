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
import re

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
    For CNN and Fox News, uses their RSS feeds (returning a dict with title and link)
    and filters for stories published within the last 24 hours.
    For MSNBC and Breitbart, attempts to scrape from the provided URL (link is set to None).
    """
    headlines = []
    try:
        if source in ["CNN", "Fox News"]:
            rss_url = "http://rss.cnn.com/rss/edition.rss" if source == "CNN" else "https://feeds.foxnews.com/foxnews/latest"
            response = requests.get(rss_url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, "lxml-xml")
            items = soup.find_all("item")
            recent_items = filter_recent_items(items, max_age_hours=24)
            items = recent_items[:10]
            # Return list of dicts with 'title' and 'link'
            headlines = [{"title": item.title.get_text(strip=True), "link": item.link.get_text(strip=True)}
                         for item in items if item.title and item.link]
        elif source in ["MSNBC", "Breitbart"]:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            if source == "Breitbart":
                elements = soup.find_all("h1")
                if len(elements) < 10:
                    elements = soup.find_all("h2")
            else:
                elements = soup.find_all("h2")
            headlines = [{"title": el.get_text(strip=True), "link": None} for el in elements][:10]
    except Exception as e:
        st.error(f"Error fetching headlines from {source}: {e}")
    return headlines

def get_article_content(url: str) -> str:
    """
    Fetches the article content from the given URL by extracting all <p> tags.
    """
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        paragraphs = soup.find_all("p")
        content = " ".join(p.get_text(strip=True) for p in paragraphs)
        return content
    except Exception as e:
        st.error(f"Error fetching article content from {url}: {e}")
        return ""

def perform_sentiment_analysis(text: str) -> float:
    """Returns the sentiment polarity of the text using TextBlob."""
    analysis = TextBlob(text)
    return analysis.sentiment.polarity

def measure_impact(text: str, link: str = None) -> int:
    """
    Uses OpenAI's Chat API to provide a human-like judgment of impact.
    If a link is available and the headline is brief, a snippet of the article content is appended.
    The prompt instructs the model to rate the potential impact on a scale from 0 to 100,
    where the rating considers not just literal numbers but broader implications.
    """
    # Use the headline as the base content.
    content = text
    if link:
        article_text = get_article_content(link)
        # If the article text is long, take a snippet.
        if article_text and len(article_text) > 200:
            content += "\n\n" + article_text[:500]  # first 500 characters
    prompt = (
        "You are an expert news analyst. Given the following news story, "
        "rate its potential impact on a scale from 0 to 100. Consider both direct effects and broader implications. "
        "Even if only one person is mentioned, think about the ripple effects on society. "
        "Provide only a single integer as your answer.\n\n"
        "News Story:\n" + content + "\n\nImpact Score:"
    )
    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
            temperature=0.0,
            n=1
        )
        result = response['choices'][0]['message']['content'].strip()
        # Extract the first integer found in the response.
        numbers = re.findall(r'\d+', result)
        if numbers:
            impact_score = int(numbers[0])
        else:
            impact_score = 20  # fallback baseline
    except Exception as e:
        st.error(f"Error calculating impact score: {e}")
        impact_score = 20
    return impact_score

def get_unbiased_summary(cluster_headlines: list) -> str:
    """
    Uses OpenAI's Chat API with the gpt-4o-mini model to generate an unbiased summary
    from a cluster of headlines.
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
    Returns a list of clusters (each cluster is a list of headline texts).
    Headlines not grouped (noise) are omitted.
    """
    if not headlines:
        return []
    texts = [h["title"] for h in headlines]
    vectorizer = TfidfVectorizer(stop_words="english")
    X = vectorizer.fit_transform(texts)
    clustering = DBSCAN(eps=0.5, min_samples=2, metric="cosine").fit(X)
    clusters = {}
    for idx, label in enumerate(clustering.labels_):
        if label == -1:
            continue
        clusters.setdefault(label, []).append(texts[idx])
    return list(clusters.values())

# ----------------------------------------------------------------------
# 2. Main App Function
# ----------------------------------------------------------------------
def main():
    st.title("Unbiased News Aggregator")
    st.write(
        "This app fetches the latest headlines from CNN, Fox News, MSNBC, and Breitbart. "
        "It performs sentiment analysis, gauges the impact of each story (using human-like judgment), "
        "clusters overlapping stories, and generates an unbiased summary using the gpt-4o-mini model."
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
        for item in headlines:
            title = item["title"]
            link = item.get("link")
            sentiment = perform_sentiment_analysis(title)
            impact = measure_impact(title, link)
            data.append({
                "Source": source,
                "Headline": title,
                "Link": link,
                "Sentiment": sentiment,
                "Impact": impact
            })
    df = pd.DataFrame(data)
    
    st.header("Headlines with Metrics")
    # Display each headline with its sentiment gauge and impact gauge.
    for idx, row in df.iterrows():
        if row["Link"]:
            st.markdown(f"**[{row['Headline']}]({row['Link']})**  _(Source: {row['Source']})_")
        else:
            st.markdown(f"**{row['Headline']}**  _(Source: {row['Source']})_")
        col1, col2 = st.columns(2)
        with col1:
            # Transform sentiment (-1 to 1) to a scale of 0 to 100.
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
        with col2:
            impact_value = int(row["Impact"])
            st.progress(impact_value / 100)
            st.write(f"Impact Score: {impact_value}/100")
        st.markdown("---")
    
    # Cluster headlines for overlapping stories.
    st.header("Overlapping Stories & Unbiased Summaries")
    combined_headlines = []
    for headlines in all_headlines.values():
        for item in headlines:
            combined_headlines.append(item["title"])
    clusters = cluster_headlines([{"title": h} for h in combined_headlines])
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
