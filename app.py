from flask import Flask, render_template, request, jsonify
import pandas as pd
from textblob import TextBlob
from collections import Counter
import re

app = Flask(__name__)

# -----------------------------
# Load dataset
# -----------------------------

DATA_FILE = "data/reviews.csv"

df = pd.read_csv(DATA_FILE)


# -----------------------------
# Text preprocessing
# -----------------------------

def clean_text(text):
    text = str(text).lower()

    text = re.sub(
        r"http\S+|www\S+",
        "",
        text
    )

    text = re.sub(
        r"[^a-zA-Z\s]",
        "",
        text
    )

    stop_words = {
        "the", "is", "a", "an",
        "and", "to", "of", "was",
        "very", "with", "this",
        "that", "but", "i",
        "am", "it", "for",
        "in", "on", "my"
    }

    words = text.split()

    words = [
        word
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)


# -----------------------------
# Sentiment analysis
# -----------------------------

def analyze_sentiment(text):

    result = TextBlob(
        str(text)
    )

    polarity = result.sentiment.polarity

    subjectivity = result.sentiment.subjectivity

    if polarity > 0.05:
        sentiment = "Positive"

    elif polarity < -0.05:
        sentiment = "Negative"

    else:
        sentiment = "Neutral"

    return sentiment, polarity, subjectivity


# -----------------------------
# Analyze dataset
# -----------------------------

df["cleaned_text"] = (
    df["feedback"]
    .apply(clean_text)
)

analysis = (
    df["feedback"]
    .apply(analyze_sentiment)
)

df["sentiment"] = analysis.apply(
    lambda x: x[0]
)

df["polarity"] = analysis.apply(
    lambda x: x[1]
)

df["subjectivity"] = analysis.apply(
    lambda x: x[2]
)


# -----------------------------
# Home page
# -----------------------------

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# -----------------------------
# Analyze new review
# -----------------------------

@app.route(
    "/analyze",
    methods=["POST"]
)
def analyze():

    data = request.get_json()

    text = data.get(
        "text",
        ""
    ).strip()

    if not text:

        return jsonify({
            "error":
            "Please enter a review."
        })

    cleaned = clean_text(
        text
    )

    sentiment, polarity, subjectivity = (
        analyze_sentiment(text)
    )

    return jsonify({

        "original_text": text,

        "cleaned_text": cleaned,

        "sentiment": sentiment,

        "polarity": round(
            polarity,
            3
        ),

        "subjectivity": round(
            subjectivity,
            3
        )
    })


# -----------------------------
# Dashboard data
# -----------------------------

@app.route("/dashboard")
def dashboard():

    total = len(df)

    positive = int(
        (df["sentiment"] == "Positive")
        .sum()
    )

    negative = int(
        (df["sentiment"] == "Negative")
        .sum()
    )

    neutral = int(
        (df["sentiment"] == "Neutral")
        .sum()
    )


    # -------------------------
    # Frequent words
    # -------------------------

    all_words = []

    for text in df["cleaned_text"]:

        all_words.extend(
            text.split()
        )

    word_counts = Counter(
        all_words
    )

    top_words = (
        word_counts
        .most_common(10)
    )


    # -------------------------
    # Category analysis
    # -------------------------

    category_table = pd.crosstab(
        df["category"],
        df["sentiment"]
    )

    categories = []

    for category in category_table.index:

        categories.append({

            "category": category,

            "positive": int(
                category_table
                .loc[category]
                .get("Positive", 0)
            ),

            "negative": int(
                category_table
                .loc[category]
                .get("Negative", 0)
            ),

            "neutral": int(
                category_table
                .loc[category]
                .get("Neutral", 0)
            )
        })


    # -------------------------
    # Trend analysis
    # -------------------------

    df["date"] = pd.to_datetime(
        df["date"]
    )

    trend = (
        df.groupby("date")["polarity"]
        .mean()
        .reset_index()
    )


    return jsonify({

        "total": total,

        "positive": positive,

        "negative": negative,

        "neutral": neutral,

        "positive_percentage":
            round(
                positive / total * 100,
                1
            ),

        "negative_percentage":
            round(
                negative / total * 100,
                1
            ),

        "neutral_percentage":
            round(
                neutral / total * 100,
                1
            ),

        "top_words": [
            {
                "word": word,
                "count": count
            }

            for word, count
            in top_words
        ],

        "categories": categories,

        "trend_dates":
            trend["date"]
            .dt.strftime(
                "%Y-%m-%d"
            )
            .tolist(),

        "trend_values":
            trend["polarity"]
            .round(3)
            .tolist()
    })


# -----------------------------
# Run application
# -----------------------------

if __name__ == "__main__":

    app.run(
        debug=True
    )