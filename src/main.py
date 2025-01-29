import nltk
from analysis import analyze_sentiment, apply_sentiment_analysis, calculate_sentiment_distribution
from cleaning import preprocess_data, load_data
from visualize import extract_key_themes, generate_word_cloud, visualize_sentiment_distribution


nltk.download('stopwords')

def save_report(sentiment_counts, key_themes, output_file='sentiment_report.txt'):
    """
    Save the sentiment analysis report to a text file.
    """
    with open(output_file, 'w') as f:
        f.write("Sentiment Analysis Report\n")
        f.write("=========================\n\n")
        
        # Write sentiment distribution
        f.write("Sentiment Distribution (%):\n")
        f.write(sentiment_counts.to_string())
        f.write("\n\n")
        
        # Write key themes
        f.write("Key Themes (Most Frequent Words):\n")
        f.write(key_themes.to_string())
        f.write("\n\n")
    print(f"Report saved as {output_file}")


def main():
    # Load the dataset
    filepath = '..\data\Reviews.csv'  
    df = load_data(filepath)
    
    # Preprocess the data
    df = preprocess_data(df)

    # Perform sentiment analysis
    df['Sentiment'] = df['Cleaned_Review'].apply(analyze_sentiment)

    # Calculate sentiment distribution
    sentiment_counts = calculate_sentiment_distribution(df)

    # Extract key themes
    key_themes = extract_key_themes(df)

    # Visualize sentiment distribution
    visualize_sentiment_distribution(df)

    # Generate word clouds
    generate_word_cloud(df)  # Word cloud for all reviews
    generate_word_cloud(df, sentiment='Positive')  # Word cloud for positive reviews
    generate_word_cloud(df, sentiment='Negative')  # Word cloud for negative reviews

    # Save report
    save_report(sentiment_counts, key_themes)

if __name__ == "__main__":
    main()