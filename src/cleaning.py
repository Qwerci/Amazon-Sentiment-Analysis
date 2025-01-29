import pandas as pd
import re
from nltk.corpus import stopwords



def load_data(file_path):
    """
    Load the dataset from the specified file path.
    """
    try:
        data = pd.read_csv(file_path)
        print("Dataset loaded successfully!")
        # data.columns = ["Id", "Product Id", "User Id", "Profile Name", "Helpful Numerator", "Helpful Denominator", "Score", "Date", "Summary", "Text"]
        return data
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None
    

def preprocess_data(df):
    """
    Preprocess the dataset by selecting relevant columns, handling missing values,
    and cleaning the review text.
    
    Parameters:
    df (pd.DataFrame): Original dataset.
    
    Returns:
    pd.DataFrame: Preprocessed dataset.
    """
    # Select relevant columns
    df = df[['UserId', 'ProductId', 'Score', 'Time', 'Text']]
    df.columns = ['User', 'Product', 'Rating', 'Time', 'Review']
    
    # Handle missing values
    df.dropna(subset=['Review'], inplace=True)
    
    # Clean the review text
    df['Cleaned_Review'] = df['Review'].apply(clean_text)
    
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    df['Cleaned_Review'] = df['Cleaned_Review'].apply(
        lambda x: ' '.join([word for word in x.split() if word not in stop_words])
    )
    
    return df

def clean_text(text):
    """
    Clean the input text by converting to lowercase, removing numbers, punctuation,
    and extra whitespace.
    
    Parameters:
    text (str): Original text.
    
    Returns:
    str: Cleaned text.
    """
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\s+', ' ', text)  # Remove extra whitespace
    return text
