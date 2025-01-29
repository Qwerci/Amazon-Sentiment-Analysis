from textblob import TextBlob


def analyze_sentiment(text):
    """
    Analyze the sentiment of the input text using TextBlob.
    
    Parameters:
    text (str): Cleaned text.
    
    Returns:
    str: Sentiment label ('Positive', 'Negative', or 'Neutral').
    """
    analysis = TextBlob(text)
    if analysis.sentiment.polarity > 0:
        return 'Positive'
    elif analysis.sentiment.polarity < 0:
        return 'Negative'
    else:
        return 'Neutral'

def apply_sentiment_analysis(df):
    """
    Apply sentiment analysis to the dataset.
    
    Parameters:
    df (pd.DataFrame): Preprocessed dataset.
    
    Returns:
    pd.DataFrame: Dataset with an additional 'Sentiment' column.
    """
    df['Sentiment'] = df['Cleaned_Review'].apply(analyze_sentiment)
    return df

def calculate_sentiment_distribution(df):
    """
    Calculate and display the percentage distribution of sentiments.
    """
    sentiment_counts = df['Sentiment'].value_counts(normalize=True) * 100
    print("\nSentiment Distribution:")
    print(sentiment_counts)
    return sentiment_counts
