import streamlit as st
import requests
from bs4 import BeautifulSoup
import openai
from textblob import TextBlob
import pandas as pd
import plotly.graph_objects as go
import datetime
import re

# ----------------------------------------------------------------------
# 1. Configuration and Helper Functions
# ----------------------------------------------------------------------

# Set up OpenAI API key from Streamlit secrets.
openai.api_key = st.secrets["OPENAI"]["API_KEY"]

# Define a headers dictionary to mimic a real browser.
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/103.0.0.0 Safari/537.36"
    )
}

@st.cache_data(show_spinner=False)
def get_headlines(url: str, source: str) -> list:
    """
    Fetches up to 10 headlines for a given source.
    
    - For CNN, we scrape the main page (https://www.cnn.com) with a browser-like User-Agent.
      We use a union CSS selector that targets:
         • <span data-editable="headline"> elements,
         • <span class="container__headline-text"> elements,
         • <h2 class="container__title_url-text"> elements.
      Then we filter out any headline with fewer than 5 words.
    
    - For Fox News, MSNBC, and Breitbart, existing methods (using RSS for Fox News and simple scraping for MSNBC/Breitbart) are used.
    """
    headlines = []
    try:
        if source == "CNN":
            cnn_url = "https://www.cnn.com"
            response = requests.get(cnn_url, headers=HEADERS, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            # Use a union selector to capture possible headline elements.
            elements = soup.select(
                "span[data-editable='headline'], span.container__headline-text, h2.container__title_url-text"
            )
            cnn_headlines = []
            for el in elements:
                headline_text = el.get_text(strip=True)
                # Filter out headlines with fewer than 5 words.
                if len(headline_text.split()) < 5:
                    continue
                # Attempt to get the parent anchor to retrieve the link.
                parent_a = el.find_parent("a")
                if not parent_a:
                    continue
                link = parent_a.get("href")
                if not link:
                    continue
                if not link.startswith("http"):
                    link = "https://www.cnn.com" + link
                cnn_headlines.append({"title": headline_text, "link": link})
                if len(cnn_headlines) >= 10:
                    break
            headlines = cnn_headlines
        elif source == "Fox News":
            rss_url = "https://feeds.foxnews.com/foxnews/latest"
            response = requests.get(rss_url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, "lxml-xml")
            items = soup.find_all("item")[:10]
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
    Fetches the article content from the given URL by extracting text from all <p> tags.
    """
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
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

def basic_impact(text: str) -> int:
    """
    Estimates an impact score (0 to 100) based on direct keyword matches.
    Returns 0 if no keywords are found.
    """
    keywords = {
        "global": 100,
        "world": 90,
        "nationwide": 85,
        "government": 80,
        "policy": 60,
        "state": 50,
        "local": 30,
        "individual": 10,
        "family": 20,
        "community": 40,
        "outbreak": 89,
        "crisis": 94,
        "pandemic": 98,
    }
    score = 0
    text_lower = text.lower()
    for key, value in keywords.items():
        if key in text_lower:
            score = max(score, value)
    return score

def measure_impact(text: str, link: str = None) -> int:
    """
    Uses OpenAI's Chat API to provide a human-like judgment of impact.
    If a link is available and the headline is brief, a snippet of the full article content is appended.
    The prompt instructs the model to rate the potential impact on a scale from 0 to 100,
    considering both direct effects and broader implications.
    """
    content = text
    if link:
        article_text = get_article_content(link)
        if article_text and len(article_text) > 200:
            content += "\n\n" + article_text[:500]
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
        result = response.choices[0].message.content.strip()
        numbers = re.findall(r'\d+', result)
        if numbers:
            impact_score = int(numbers[0])
        else:
            impact_score = 20  # fallback baseline
    except Exception as e:
        st.error(f"Error calculating impact score: {e}")
        impact_score = 20
    return impact_score

def get_unbiased_summary_for_story(headline: str, link: str = None) -> str:
    """
    Generates an unbiased summary for a single news story.
    If an article link is available, fetches the full article content (truncated if necessary)
    and includes it in the prompt so the full story is analyzed.
    """
    content = headline
    if link:
        article_text = get_article_content(link)
        if article_text:
            if len(article_text) > 2000:
                article_text = article_text[:2000] + " ... [truncated]"
            content += "\n\n" + article_text
    prompt = (
        "You are an expert news summarizer. Summarize the following full news article in a neutral and unbiased manner, "
        "focusing on the key facts and broader implications. Use concise language.\n\n"
        "News Article:\n" + content + "\n\nSummary:"
    )
    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150,
            temperature=0.7,
            n=1
        )
        summary = response.choices[0].message.content.strip()
    except Exception as e:
        summary = "Error generating summary: " + str(e)
    return summary

# ----------------------------------------------------------------------
# 2. Main App Function
# ----------------------------------------------------------------------
def main():
    st.title("UnSpun")
    st.write(
        "This app fetches the latest headlines from CNN, Fox News, MSNBC, and Breitbart. "
        "It provides unbiased summaries of high-impact stories and gauges the sentiment of how the stories were depicted by their original sources."
    )

    # Define the news sources and their URLs.
    news_sources = {
        "CNN": "https://www.cnn.com",  # Scrape CNN homepage directly.
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

    # --- Top 10 High Impact Story Summaries Section ---
    st.header("Top 10 High Impact Story Summaries")
    top_stories = df.sort_values(by="Impact", ascending=False).head(10)
    for idx, row in top_stories.iterrows():
        if row["Link"]:
            st.markdown(f"**[{row['Headline']}]({row['Link']})**  _(Source: {row['Source']})_")
        else:
            st.markdown(f"**{row['Headline']}**  _(Source: {row['Source']})_")
        summary = get_unbiased_summary_for_story(row["Headline"], row["Link"])
        st.write(summary)
        st.markdown("---")
    
    # --- Headlines with Metrics Section ---
    st.header("Headlines with Metrics")
    for idx, row in df.iterrows():
        if row["Link"]:
            st.markdown(f"**[{row['Headline']}]({row['Link']})**  _(Source: {row['Source']})_")
        else:
            st.markdown(f"**{row['Headline']}**  _(Source: {row['Source']})_")
        col1, col2 = st.columns(2)
        with col1:
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

if __name__ == "__main__":
    main()
