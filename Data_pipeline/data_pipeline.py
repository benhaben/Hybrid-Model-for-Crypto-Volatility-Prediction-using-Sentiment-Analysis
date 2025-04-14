import pandas as pd
import requests
import time
import schedule
import praw
from datetime import datetime, timedelta
import re
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch
import logging
from dotenv import load_dotenv
import os

# Configure logging and output directory
logging.basicConfig(level=logging.INFO)
data_directory = os.path.join(os.path.dirname(__file__), '..', 'Data_set')
os.makedirs(data_directory, exist_ok=True)

# Load environment variables
load_dotenv()

# Reddit API credentials
REDDIT_API_ID = os.getenv("REDDIT_CLIENT_ID")
REDDIT_API_KEY = os.getenv("REDDIT_CLIENT_SECRET")
REDDIT_APP_NAME = os.getenv("REDDIT_USER_AGENT")

# Initialize NLP components
text_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased-finetuned-sst-2-english')
sentiment_model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased-finetuned-sst-2-english')

def collect_social_data():
    crypto_communities = [
        "Bitcoin",
        "BTC",
        "Crypto",
        "CryptoNews",
        "Bitcoin",
        "Cryptocurrency",
        "BitcoinPrice",
        "CryptoAnalysis",
    ]
    collected_posts = []
    for community in crypto_communities:
        community_data = get_reddit_posts(community)
        if not community_data.empty:
            collected_posts.append(community_data)
    
    if not collected_posts:
        logging.error("No Reddit posts gathered")
        return None

    combined_posts = pd.concat(collected_posts, ignore_index=True)
    combined_posts = combined_posts[combined_posts['body'].notnull() & (combined_posts['body'].str.strip() != '')]
    combined_posts = combined_posts.rename(columns={'body':'Description','comms_num':'No_Comments'})
    combined_posts['title'] = combined_posts['title'].apply(clean_text)
    combined_posts['Description'] = combined_posts['Description'].apply(clean_text)
    
    return combined_posts

def evaluate_sentiment(text_input):
    try:
        tokenized_input = text_tokenizer(text_input, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            model_output = sentiment_model(**tokenized_input)
        prediction_scores = model_output.logits
        confidence_values = torch.nn.functional.softmax(prediction_scores, dim=1)
        sentiment_values = 2 * confidence_values[:, 1].numpy() - 1
        return sentiment_values
    except Exception as e:
        logging.error(f"Sentiment evaluation failed: {str(e)}")
        return None
    
def get_reddit_posts(subreddit_name, post_limit=100):
    try:
        reddit_api = praw.Reddit(
            client_id=REDDIT_API_ID,
            client_secret=REDDIT_API_KEY,
            user_agent=REDDIT_APP_NAME
        )
        target_subreddit = reddit_api.subreddit(subreddit_name)
        post_data = []

        for submission in target_subreddit.hot(limit=post_limit):
            post_data.append({
                'title': submission.title,
                'score': submission.score,
                'id': submission.id,
                'url': submission.url,
                'comms_num': submission.num_comments,
                'created': datetime.fromtimestamp(submission.created),
                'body': submission.selftext
            })

        return pd.DataFrame(post_data)
    except Exception as e:
        logging.error(f"Failed to get posts from r/{subreddit_name}: {str(e)}")
        return pd.DataFrame()
    
def clean_text(text_content):
    try:
        cleaned_text = re.sub(r'[^a-zA-Z\s]', '', str(text_content))
        cleaned_text = cleaned_text.lower()
        return cleaned_text
    except Exception as e:
        logging.error(f"Text cleaning error: {text_content[:50]}... Error: {str(e)}")
        return ""

def fetch_crypto_info(currency_id, time_period=60):
    api_endpoint = f"https://api.coingecko.com/api/v3/coins/{currency_id}/market_chart"
    query_params = {
        'vs_currency': 'usd',
        'days': time_period,
        'interval': 'daily'
    }
    
    try:
        api_response = requests.get(api_endpoint, params=query_params)
        api_response.raise_for_status()
        response_data = api_response.json()
        
        price_df = pd.DataFrame(response_data['prices'], columns=['timestamp', f'{currency_id}_price'])
        price_df['timestamp'] = pd.to_datetime(price_df['timestamp'], unit='ms').dt.date
        price_df.set_index('timestamp', inplace=True)
        
        price_df[f'{currency_id}_volume'] = [vol[1] for vol in response_data['total_volumes']]
        price_df[f'{currency_id}_market_cap'] = [cap[1] for cap in response_data['market_caps']]
        
        return price_df
    except requests.exceptions.RequestException as e:
        logging.error(f"API request failed for {currency_id}: {str(e)}")
        return pd.DataFrame()

def compute_price_variation(df, currency_id, period=30):
    df[f'{currency_id}_returns'] = df[f'{currency_id}_price'].pct_change()
    df[f'{currency_id}_volatility'] = df[f'{currency_id}_returns'].rolling(window=period).std() * (365**0.5)
    return df

def process_dataset():
    social_data = collect_social_data()
    if social_data is None:
        return None

    social_data['title_sentiment'] = evaluate_sentiment(social_data['title'].tolist())
    social_data['description_sentiment'] = evaluate_sentiment(social_data['Description'].tolist())

    social_data['created'] = pd.to_datetime(social_data['created'])
    daily_social = social_data.groupby(social_data['created'].dt.date).agg({
        'score': 'mean',
        'No_Comments': 'sum',
        'title_sentiment': 'mean',
        'description_sentiment': 'mean'
    }).reset_index()
    daily_social['created'] = pd.to_datetime(daily_social['created']).dt.date
    daily_social.set_index('created', inplace=True)

    bitcoin_data = fetch_crypto_info('bitcoin')
    if bitcoin_data.empty:
        logging.error("Bitcoin data retrieval unsuccessful")
        return None

    bitcoin_data = compute_price_variation(bitcoin_data, 'bitcoin')    
    bitcoin_data['bitcoin_returns'] = bitcoin_data['bitcoin_returns'].fillna(bitcoin_data['bitcoin_returns'].mean())
    bitcoin_data['bitcoin_volatility'] = bitcoin_data['bitcoin_volatility'].fillna(bitcoin_data['bitcoin_volatility'].mean())
    output_file = os.path.join(data_directory, 'crypto_market_dataset.csv')
    bitcoin_data.to_csv(output_file, index=False)
    print("Dataset saved as 'Crypto_market_dataset.csv'")
    
    merged_dataset = pd.merge(bitcoin_data, daily_social, left_index=True, right_index=True, how='left')
    logging.info('Dataset combination successful')

    merged_dataset['score'] = merged_dataset['score'].fillna(merged_dataset['score'].mean())
    merged_dataset['No_Comments'] = merged_dataset['No_Comments'].fillna(merged_dataset['No_Comments'].median())
    merged_dataset['title_sentiment'] = merged_dataset['title_sentiment'].fillna(merged_dataset['title_sentiment'].mean())
    merged_dataset['description_sentiment'] = merged_dataset['description_sentiment'].fillna(merged_dataset['description_sentiment'].mean()).infer_objects(copy=False)

    for column in ['score', 'No_Comments', 'title_sentiment', 'description_sentiment']:
        for day in range(1, 8):
            merged_dataset[f'{column}_lag{day}'] = merged_dataset[column].shift(day)

    for lag_column in ['score_lag', 'No_Comments_lag', 'title_sentiment_lag', 'description_sentiment_lag']:
        for day in range(1, 8):
            merged_dataset[f"{lag_column}{day}"] = merged_dataset[f"{lag_column}{day}"].fillna(merged_dataset[f"{lag_column}{day}"].mean())

    merged_dataset['sentiment'] = (merged_dataset['title_sentiment'] + merged_dataset['description_sentiment']) / 2
    for day in range(1, 8):
        merged_dataset[f"sentiment{day}"] = (merged_dataset[f'title_sentiment_lag{day}'] + merged_dataset[f'description_sentiment_lag{day}']) / 2

    sentiment_metrics = ['sentiment'] + [f'sentiment{day}' for day in range(1, 8)]
    merged_dataset['total_sentiment'] = merged_dataset[sentiment_metrics].mean(axis=1)

    merged_dataset.index.name = 'timestamp'
    merged_dataset = merged_dataset.reset_index()
    logging.info("Dataset processing finished.")
    return merged_dataset

if __name__ == "__main__":
    raw_social_data = collect_social_data()
    if raw_social_data is not None:
        output_location = os.path.join(data_directory, 'reddit_data_raw.csv')
        raw_social_data.to_csv(output_location, index=False)
        print("Social data saved as 'reddit_data_raw.csv'")
    else:
        print("Data collection unsuccessful")
    
    final_dataset = process_dataset()
    if final_dataset is not None:
        result_path = os.path.join(data_directory, 'final_data.csv')
        final_dataset.to_csv(result_path, index=False)
        print("Processed data saved as 'final_data.csv'")
    else:
        print("Data processing unsuccessful")