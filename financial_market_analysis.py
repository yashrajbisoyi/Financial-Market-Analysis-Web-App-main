#!/usr/bin/env python3
# Financial Market Pattern Analysis using Web Scraping

# Basic libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import random
import warnings

# Web scraping libraries
import requests
from bs4 import BeautifulSoup

# Data processing libraries
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Sentiment analysis
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob

# Visualization
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from wordcloud import WordCloud

# Suppress warnings
warnings.filterwarnings('ignore')

# Download NLTK resources if not already downloaded
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')
    nltk.download('punkt')

# WebScraper class for data collection
class WebScraper:
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept-Language': 'en-US,en;q=0.9',
        }
        # Rate limiting parameters
        self.min_delay = 2
        self.max_delay = 5
    
    def _get_random_delay(self):
        """Generate a random delay between requests to avoid getting blocked"""
        return random.uniform(self.min_delay, self.max_delay)
    
    def _make_request(self, url):
        """Make an HTTP request with appropriate headers and delays"""
        try:
            # Random delay before request
            time.sleep(self._get_random_delay())
            
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()  # Raise an exception for 4XX/5XX responses
            
            return response
        except requests.exceptions.RequestException as e:
            print(f"Error making request to {url}: {e}")
            return None
    
    def scrape_stock_price(self, stock_symbol):
        """Scrape stock price data from MoneyControl"""
        # MoneyControl URL pattern for stock quotes
        url = f"https://www.moneycontrol.com/india/stockpricequote/{stock_symbol}"
        print(f"Scraping stock price data for {stock_symbol} from {url}...")
        
        response = self._make_request(url)
        if not response:
            return None
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extract current price (this is a simplified example)
        try:
            price_div = soup.find('div', {'class': 'inprice1 nsecp'})
            if price_div:
                current_price = price_div.text.strip()
                print(f"Current price for {stock_symbol}: {current_price}")
                return {
                    'symbol': stock_symbol,
                    'current_price': current_price,
                    'timestamp': pd.Timestamp.now()
                }
            else:
                print(f"Could not find price for {stock_symbol}")
                return None
        except Exception as e:
            print(f"Error parsing price data for {stock_symbol}: {e}")
            return None
    
    def scrape_historical_data(self, stock_symbol, days=30):
        """Scrape historical stock data from NSE website"""
        # For educational purposes only, in reality, you might need to use other sources
        url = f"https://www.nseindia.com/get-quotes/equity?symbol={stock_symbol}"
        print(f"Scraping historical data for {stock_symbol} from {url}...")
        
        # This is a placeholder - actual implementation would differ based on website structure
        # In a real scenario, you would use the NSE API or parse HTML tables
        # For demonstration, we'll create synthetic data
        
        # Generate synthetic historical data
        dates = pd.date_range(end=pd.Timestamp.now().date(), periods=days)
        base_price = random.uniform(100, 5000)  # Random starting price
        prices = [base_price]
        
        # Generate random price movements
        for i in range(1, days):
            change = random.uniform(-0.03, 0.03)  # Daily change between -3% and 3%
            new_price = prices[-1] * (1 + change)
            prices.append(new_price)
        
        # Create DataFrame with historical data
        df = pd.DataFrame({
            'Date': dates,
            'Symbol': stock_symbol,
            'Open': prices,
            'High': [p * random.uniform(1.001, 1.02) for p in prices],  # Random high price
            'Low': [p * random.uniform(0.98, 0.999) for p in prices],   # Random low price
            'Close': [p * random.uniform(0.99, 1.01) for p in prices],  # Random close price
            'Volume': [int(random.uniform(100000, 1000000)) for _ in range(days)]  # Random volume
        })
        
        return df
    
    def scrape_company_metrics(self, stock_symbol):
        """Scrape company performance metrics from Screener.in"""
        url = f"https://www.screener.in/company/{stock_symbol}/"
        print(f"Scraping company metrics for {stock_symbol} from {url}...")
        
        response = self._make_request(url)
        if not response:
            # Generate synthetic metrics for demonstration
            print(f"Using synthetic data for {stock_symbol} metrics")
            return {
                'symbol': stock_symbol,
                'market_cap': round(random.uniform(10000, 200000), 2),  # In crores
                'pe_ratio': round(random.uniform(10, 40), 2),
                'eps': round(random.uniform(10, 100), 2),
                'dividend_yield': round(random.uniform(0.5, 3.5), 2),
                'book_value': round(random.uniform(100, 1000), 2),
                'roe': round(random.uniform(5, 25), 2),  # Return on Equity
                'debt_to_equity': round(random.uniform(0.1, 2.0), 2)
            }
        
        # Real parsing would happen here with BeautifulSoup
        # For demonstration, we'll still use synthetic data
        return {
            'symbol': stock_symbol,
            'market_cap': round(random.uniform(10000, 200000), 2),  # In crores
            'pe_ratio': round(random.uniform(10, 40), 2),
            'eps': round(random.uniform(10, 100), 2),
            'dividend_yield': round(random.uniform(0.5, 3.5), 2),
            'book_value': round(random.uniform(100, 1000), 2),
            'roe': round(random.uniform(5, 25), 2),  # Return on Equity
            'debt_to_equity': round(random.uniform(0.1, 2.0), 2)
        }
    
    def scrape_index_values(self):
        """Scrape index values (NIFTY, SENSEX) from NSE/BSE websites"""
        indices = ['NIFTY 50', 'SENSEX']
        index_data = {}
        
        # NSE for NIFTY
        nse_url = "https://www.nseindia.com/"
        print(f"Scraping NIFTY 50 from {nse_url}...")
        
        # BSE for SENSEX
        bse_url = "https://www.bseindia.com/"
        print(f"Scraping SENSEX from {bse_url}...")
        
        # For demonstration, we'll use synthetic data
        print("Using synthetic data for index values")
        for index in indices:
            base_value = 20000 if index == 'SENSEX' else 18000
            index_data[index] = {
                'current_value': round(base_value * random.uniform(0.98, 1.02), 2),
                'change': round(random.uniform(-200, 200), 2),
                'percent_change': round(random.uniform(-1.5, 1.5), 2),
                'timestamp': pd.Timestamp.now()
            }
        
        return index_data
    
    def scrape_financial_news(self, num_articles=10):
        """Scrape financial news articles from Economic Times and MoneyControl"""
        news_data = []
        
        # Economic Times
        et_url = "https://economictimes.indiatimes.com/markets/stocks/news"
        print(f"Scraping financial news from {et_url}...")
        
        # Generate synthetic news data for demonstration
        print("Using synthetic news data")
        
        synthetic_headlines = [
            "Sensex climbs to all-time high on strong GDP growth",
            "IT sector faces pressure amid global tech slowdown",
            "RBI keeps repo rate unchanged, maintains accommodative stance",
            "Banking stocks rally on improved NPA scenario",
            "Auto sector shows recovery signs amid festive demand",
            "FMCG companies report mixed quarterly results",
            "Pharmaceutical stocks continue upward trend",
            "Metal stocks surge on global commodity rally",
            "Investors cautious ahead of upcoming budget",
            "FIIs increase stake in Indian equities"
        ]
        
        sources = ['Economic Times', 'MoneyControl', 'LiveMint', 'Financial Express', 'Business Standard']
        
        # Generate synthetic dates within the last week
        today = pd.Timestamp.now()
        past_dates = [today - pd.Timedelta(days=random.randint(0, 7)) for _ in range(num_articles)]
        
        # Generate synthetic news data
        for i in range(num_articles):
            idx = i % len(synthetic_headlines)
            date_idx = i % len(past_dates)
            source_idx = i % len(sources)
            
            news_data.append({
                'headline': synthetic_headlines[idx],
                'date': past_dates[date_idx].strftime('%d %b %Y'),
                'source': sources[source_idx],
                'url': 'https://example.com/synthetic-article'
            })
        
        # Convert to DataFrame
        df_news = pd.DataFrame(news_data)
        return df_news

# DataProcessor class for data cleaning and processing
class DataProcessor:
    def __init__(self):
        self.imputer = SimpleImputer(strategy='mean')
        self.scaler = StandardScaler()
    
    def convert_to_dataframe(self, raw_data, data_type='stock'):
        """Convert raw scraped data to pandas DataFrame"""
        if data_type == 'stock':
            # Assuming raw_data is a dictionary of stock data
            df_list = []
            for stock_symbol, data in raw_data.items():
                if 'historical' in data and isinstance(data['historical'], pd.DataFrame):
                    df_list.append(data['historical'])
            
            if df_list:
                return pd.concat(df_list, ignore_index=True)
            else:
                print("No historical stock data found in raw data")
                return pd.DataFrame()
                
        elif data_type == 'metrics':
            # Assuming raw_data is a dictionary of company metrics
            metrics_list = []
            for stock_symbol, metrics in raw_data.items():
                metrics_list.append(metrics)
            
            if metrics_list:
                return pd.DataFrame(metrics_list)
            else:
                print("No company metrics found in raw data")
                return pd.DataFrame()
                
        elif data_type == 'index':
            # Convert index data to DataFrame
            index_list = []
            for index_name, data in raw_data.items():
                data['index'] = index_name
                index_list.append(data)
            
            if index_list:
                return pd.DataFrame(index_list)
            else:
                print("No index data found in raw data")
                return pd.DataFrame()
                
        elif data_type == 'news':
            # Assuming raw_data is already a DataFrame for news
            if isinstance(raw_data, pd.DataFrame) and not raw_data.empty:
                return raw_data
            else:
                print("No news data found or not in DataFrame format")
                return pd.DataFrame()
        else:
            print(f"Unsupported data type: {data_type}")
            return pd.DataFrame()
    
    def handle_missing_values(self, df, method='impute'):
        """Handle missing values in DataFrame"""
        if df.empty:
            print("Empty DataFrame, nothing to process")
            return df
            
        # Check for missing values
        missing_count = df.isnull().sum().sum()
        print(f"Found {missing_count} missing values in the DataFrame")
        
        if missing_count == 0:
            print("No missing values found, returning original DataFrame")
            return df
            
        if method == 'impute':
            # Get numeric columns only for imputation
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            if numeric_cols:
                # Fit imputer and transform numeric columns
                df_numeric = df[numeric_cols]
                df_imputed = pd.DataFrame(self.imputer.fit_transform(df_numeric), columns=numeric_cols)
                
                # Replace numeric columns in original DataFrame
                for col in numeric_cols:
                    df[col] = df_imputed[col].values
                    
                print(f"Imputed missing values in {len(numeric_cols)} numeric columns")
                
            # For non-numeric columns, fill with column mode
            for col in df.select_dtypes(exclude=['number']).columns:
                if df[col].isnull().any():
                    mode_value = df[col].mode()[0]
                    df[col].fillna(mode_value, inplace=True)
                    print(f"Filled missing values in column '{col}' with mode: {mode_value}")
                    
        elif method == 'drop':
            # Drop rows with any missing values
            original_rows = len(df)
            df = df.dropna()
            dropped_rows = original_rows - len(df)
            print(f"Dropped {dropped_rows} rows with missing values")
            
        elif method == 'forward_fill':
            # Forward fill (often useful for time series data)
            df = df.ffill()
            remaining_missing = df.isnull().sum().sum()
            print(f"Performed forward fill, {remaining_missing} missing values remain")
            
            # If still missing values at beginning of series, backward fill
            if remaining_missing > 0:
                df = df.bfill()
                final_missing = df.isnull().sum().sum()
                print(f"Performed backward fill to handle remaining values, {final_missing} missing values remain")
        
        return df
    
    def remove_duplicates(self, df):
        """Remove duplicate rows from DataFrame"""
        if df.empty:
            print("Empty DataFrame, nothing to process")
            return df
            
        original_rows = len(df)
        df = df.drop_duplicates().reset_index(drop=True)
        removed_rows = original_rows - len(df)
        
        print(f"Removed {removed_rows} duplicate rows")
        return df
    
    def standardize_values(self, df, columns=None):
        """Standardize numerical values in specified columns"""
        if df.empty:
            print("Empty DataFrame, nothing to process")
            return df
            
        # If no columns specified, use all numeric columns
        if columns is None:
            columns = df.select_dtypes(include=['number']).columns.tolist()
            
        if not columns:
            print("No numeric columns found for standardization")
            return df
            
        # Make a copy to avoid modifying the original
        df_standardized = df.copy()
        
        # Apply standardization only to specified columns
        df_standardized[columns] = self.scaler.fit_transform(df[columns])
        
        print(f"Standardized {len(columns)} numeric columns: {', '.join(columns)}")
        return df_standardized
    
    def format_dates(self, df, date_column='Date'):
        """Ensure date columns are in datetime format"""
        if df.empty or date_column not in df.columns:
            print(f"DataFrame is empty or doesn't contain column '{date_column}'")
            return df
            
        # Try to convert date column to datetime
        try:
            df[date_column] = pd.to_datetime(df[date_column])
            print(f"Converted column '{date_column}' to datetime format")
        except Exception as e:
            print(f"Error converting '{date_column}' to datetime: {e}")
            
        return df
    
    def process_pipeline(self, raw_data, data_type='stock'):
        """Run full data processing pipeline"""
        print(f"\n--- Processing {data_type} data ---")
        
        # Step 1: Convert to DataFrame
        print("Step 1: Converting to DataFrame...")
        df = self.convert_to_dataframe(raw_data, data_type)
        if df.empty:
            print("Empty DataFrame after conversion, stopping processing")
            return df
            
        # Step 2: Remove duplicates
        print("\nStep 2: Removing duplicates...")
        df = self.remove_duplicates(df)
        
        # Step 3: Handle missing values
        print("\nStep 3: Handling missing values...")
        if data_type == 'stock':
            # For time series data, forward fill is often appropriate
            df = self.handle_missing_values(df, method='forward_fill')
        else:
            # For other data types, use imputation
            df = self.handle_missing_values(df, method='impute')
        
        # Step 4: Format dates if applicable
        if data_type in ['stock', 'index'] and 'Date' in df.columns:
            print("\nStep 4: Formatting dates...")
            df = self.format_dates(df)
        
        print(f"\nProcessing complete. Final DataFrame shape: {df.shape}")
        return df

# PatternAnalyzer class for financial pattern analysis
class PatternAnalyzer:
    def __init__(self):
        # Initialize sentiment analyzer
        try:
            self.sentiment_analyzer = SentimentIntensityAnalyzer()
        except:
            nltk.download('vader_lexicon')
            self.sentiment_analyzer = SentimentIntensityAnalyzer()
    
    def identify_trends(self, df, stock_symbol=None, window=20):
        """Identify bullish and bearish trends using moving averages"""
        if df.empty:
            print("Empty DataFrame, cannot identify trends")
            return None
        
        # Filter for specific stock if provided
        if stock_symbol and 'Symbol' in df.columns:
            stock_df = df[df['Symbol'] == stock_symbol].copy()
            if stock_df.empty:
                print(f"No data found for symbol {stock_symbol}")
                return None
        else:
            stock_df = df.copy()
        
        # Ensure data is sorted by date
        if 'Date' in stock_df.columns:
            stock_df = stock_df.sort_values('Date')
        
        # Calculate moving averages
        stock_df['MA20'] = stock_df['Close'].rolling(window=window).mean()
        stock_df['MA50'] = stock_df['Close'].rolling(window=50).mean()
        
        # Identify bullish/bearish trends
        stock_df['Trend'] = 'Neutral'
        
        # Bullish: Price above MA, MA20 above MA50
        bullish_mask = (stock_df['Close'] > stock_df['MA20']) & (stock_df['MA20'] > stock_df['MA50'])
        stock_df.loc[bullish_mask, 'Trend'] = 'Bullish'
        
        # Bearish: Price below MA, MA20 below MA50
        bearish_mask = (stock_df['Close'] < stock_df['MA20']) & (stock_df['MA20'] < stock_df['MA50'])
        stock_df.loc[bearish_mask, 'Trend'] = 'Bearish'
        
        # Identify support and resistance levels
        # For simplicity, we'll use recent lows as support and highs as resistance
        stock_df['Rolling_Low'] = stock_df['Low'].rolling(window=window).min()
        stock_df['Rolling_High'] = stock_df['High'].rolling(window=window).max()
        
        # Find key support and resistance levels
        recent_df = stock_df.iloc[-window:] if len(stock_df) > window else stock_df
        support_level = recent_df['Rolling_Low'].mean()
        resistance_level = recent_df['Rolling_High'].mean()
        
        print(f"Identified support level: {support_level:.2f}")
        print(f"Identified resistance level: {resistance_level:.2f}")
        
        # Count trend distribution
        trend_counts = stock_df['Trend'].value_counts()
        print("\nTrend distribution:")
        for trend, count in trend_counts.items():
            print(f"{trend}: {count} days ({count/len(stock_df)*100:.1f}%)")
        
        # Get current trend
        current_trend = stock_df['Trend'].iloc[-1] if not stock_df.empty else 'Unknown'
        print(f"\nCurrent trend: {current_trend}")
        
        return {
            'data': stock_df,
            'support': support_level,
            'resistance': resistance_level,
            'current_trend': current_trend
        }
    
    def detect_volatility(self, df, stock_symbol=None, window=20):
        """Detect volatility using Bollinger Bands and other indicators"""
        if df.empty:
            print("Empty DataFrame, cannot detect volatility")
            return None
        
        # Filter for specific stock if provided
        if stock_symbol and 'Symbol' in df.columns:
            stock_df = df[df['Symbol'] == stock_symbol].copy()
            if stock_df.empty:
                print(f"No data found for symbol {stock_symbol}")
                return None
        else:
            stock_df = df.copy()
        
        # Ensure data is sorted by date
        if 'Date' in stock_df.columns:
            stock_df = stock_df.sort_values('Date')
        
        # Calculate Bollinger Bands
        stock_df['MA'] = stock_df['Close'].rolling(window=window).mean()
        stock_df['Std'] = stock_df['Close'].rolling(window=window).std()
        stock_df['Upper_Band'] = stock_df['MA'] + 2 * stock_df['Std']
        stock_df['Lower_Band'] = stock_df['MA'] - 2 * stock_df['Std']
        
        # Calculate Bollinger Band Width (BBW) - measure of volatility
        stock_df['BBW'] = (stock_df['Upper_Band'] - stock_df['Lower_Band']) / stock_df['MA']
        
        # Calculate Daily Returns
        stock_df['Daily_Return'] = stock_df['Close'].pct_change() * 100
        
        # Calculate Average True Range (ATR) - another volatility indicator
        stock_df['TR'] = 0.0
        for i in range(1, len(stock_df)):
            high_low = stock_df['High'].iloc[i] - stock_df['Low'].iloc[i]
            high_close = abs(stock_df['High'].iloc[i] - stock_df['Close'].iloc[i-1])
            low_close = abs(stock_df['Low'].iloc[i] - stock_df['Close'].iloc[i-1])
            stock_df['TR'].iloc[i] = max(high_low, high_close, low_close)
        
        stock_df['ATR'] = stock_df['TR'].rolling(window=window).mean()
        
        # Calculate MACD
        stock_df['EMA12'] = stock_df['Close'].ewm(span=12, adjust=False).mean()
        stock_df['EMA26'] = stock_df['Close'].ewm(span=26, adjust=False).mean()
        stock_df['MACD'] = stock_df['EMA12'] - stock_df['EMA26']
        stock_df['Signal'] = stock_df['MACD'].ewm(span=9, adjust=False).mean()
        
        # Calculate volatility metrics
        recent_df = stock_df.iloc[-window:] if len(stock_df) > window else stock_df
        recent_volatility = recent_df['BBW'].mean()
        historical_volatility = stock_df['BBW'].mean()
        
        volatility_state = 'High' if recent_volatility > historical_volatility else 'Low'
        
        print(f"Recent volatility (BBW): {recent_volatility:.4f}")
        print(f"Historical volatility: {historical_volatility:.4f}")
        print(f"Current volatility state: {volatility_state}")
        
        # Calculate recent average daily returns
        avg_daily_return = recent_df['Daily_Return'].mean()
        std_daily_return = recent_df['Daily_Return'].std()
        
        print(f"\nAverage daily return (last {window} days): {avg_daily_return:.2f}%")
        print(f"Standard deviation of daily returns: {std_daily_return:.2f}%")
        
        # Determine MACD signal
        if stock_df['MACD'].iloc[-1] > stock_df['Signal'].iloc[-1]:
            macd_signal = 'Bullish'
        else:
            macd_signal = 'Bearish'
        
        print(f"\nMACD Signal: {macd_signal}")
        
        return {
            'data': stock_df,
            'recent_volatility': recent_volatility,
            'historical_volatility': historical_volatility,
            'volatility_state': volatility_state,
            'avg_daily_return': avg_daily_return,
            'std_daily_return': std_daily_return,
            'macd_signal': macd_signal
        }
    
    def analyze_sentiment(self, news_df):
        """Analyze sentiment of financial news headlines"""
        if news_df.empty or 'headline' not in news_df.columns:
            print("No valid news data available for sentiment analysis")
            return None
        
        # Create a copy to avoid modifying the original
        df = news_df.copy()
        
        # Apply VADER sentiment analysis
        df['vader_sentiment'] = df['headline'].apply(lambda x: self.sentiment_analyzer.polarity_scores(x)['compound'])
        
        # Apply TextBlob sentiment analysis
        df['textblob_sentiment'] = df['headline'].apply(lambda x: TextBlob(x).sentiment.polarity)
        
        # Average the two sentiment scores
        df['sentiment'] = (df['vader_sentiment'] + df['textblob_sentiment']) / 2
        
        # Categorize sentiment
        df['sentiment_category'] = pd.cut(
            df['sentiment'], 
            bins=[-1, -0.3, 0.3, 1], 
            labels=['Negative', 'Neutral', 'Positive']
        )
        
        # Calculate sentiment statistics
        sentiment_counts = df['sentiment_category'].value_counts()
        avg_sentiment = df['sentiment'].mean()
        
        print("Sentiment analysis results:")
        print(f"Average sentiment score: {avg_sentiment:.4f}")
        print("\nSentiment distribution:")
        for category, count in sentiment_counts.items():
            print(f"{category}: {count} articles ({count/len(df)*100:.1f}%)")
        
        # Get most positive and negative headlines
        most_positive = df.loc[df['sentiment'].idxmax()]
        most_negative = df.loc[df['sentiment'].idxmin()]
        
        print(f"\nMost positive headline ({most_positive['sentiment']:.4f}): {most_positive['headline']}")
        print(f"Most negative headline ({most_negative['sentiment']:.4f}): {most_negative['headline']}")
        
        # Get overall market sentiment
        if avg_sentiment > 0.3:
            market_sentiment = "Bullish"
        elif avg_sentiment < -0.3:
            market_sentiment = "Bearish"
        else:
            market_sentiment = "Neutral"
        
        print(f"\nOverall market sentiment based on news: {market_sentiment}")
        
        return {
            'data': df,
            'avg_sentiment': avg_sentiment,
            'sentiment_counts': sentiment_counts,
            'market_sentiment': market_sentiment,
            'most_positive': most_positive,
            'most_negative': most_negative
        }
    
    def analyze_correlations(self, stock_df, events_df=None):
        """Analyze correlations between stocks and with financial events"""
        if stock_df.empty:
            print("Empty stock DataFrame, cannot analyze correlations")
            return None
        
        # Create pivot table for stock correlations (Close prices)
        if 'Symbol' in stock_df.columns and 'Close' in stock_df.columns and 'Date' in stock_df.columns:
            pivot_df = stock_df.pivot_table(index='Date', columns='Symbol', values='Close')
            
            # Calculate correlation matrix
            corr_matrix = pivot_df.corr()
            
            print("Stock price correlation matrix:")
            print(corr_matrix)
            
            # Identify highly correlated pairs
            high_corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    if abs(corr_matrix.iloc[i, j]) > 0.7:  # Threshold for high correlation
                        high_corr_pairs.append({
                            'stock1': corr_matrix.columns[i],
                            'stock2': corr_matrix.columns[j],
                            'correlation': corr_matrix.iloc[i, j]
                        })
            
            if high_corr_pairs:
                print("\nHighly correlated stock pairs:")
                for pair in high_corr_pairs:
                    print(f"{pair['stock1']} and {pair['stock2']}: {pair['correlation']:.4f}")
            else:
                print("\nNo highly correlated stock pairs found")
            
            # If events data is provided, analyze impact
            if events_df is not None and not events_df.empty:
                print("\nAnalyzing correlation with financial events:")
                # This would typically involve analyzing stock movements around event dates
                # For this example, we'll use a placeholder that would be implemented in a real system
                print("Event correlation analysis would be implemented here.")
            
            return {
                'correlation_matrix': corr_matrix,
                'high_correlation_pairs': high_corr_pairs
            }
        else:
            print("Required columns not found in stock data for correlation analysis")
            return None

# Visualizer class for financial data visualization
class Visualizer:
    def __init__(self):
        # Set plot style
        plt.style.use('seaborn-v0_8-darkgrid')  # Updated style name for newer matplotlib versions
        # Configure larger default figure size
        plt.rcParams['figure.figsize'] = (12, 8)
        # Better visualization for dark backgrounds
        plt.rcParams['axes.facecolor'] = '#f0f0f0'
        plt.rcParams['figure.facecolor'] = '#f8f8f8'
    
    def plot_stock_trends(self, stock_df, title=None):
        """Plot stock price trends with moving averages"""
        if stock_df.empty or 'Close' not in stock_df.columns:
            print("No valid stock data for trend visualization")
            return None
        
        # Create a figure
        fig = plt.figure(figsize=(14, 10))
        gs = fig.add_gridspec(2, 1, height_ratios=[3, 1])
        
        # Create main price plot with moving averages
        ax1 = fig.add_subplot(gs[0])
        
        # Plot stock price
        ax1.plot(stock_df['Date'], stock_df['Close'], label='Close Price', color='blue', linewidth=2)
        
        # Plot moving averages if available
        if 'MA20' in stock_df.columns:
            ax1.plot(stock_df['Date'], stock_df['MA20'], label='20-day MA', color='red', linewidth=1.5)
        if 'MA50' in stock_df.columns:
            ax1.plot(stock_df['Date'], stock_df['MA50'], label='50-day MA', color='green', linewidth=1.5)
        
        # Plot support and resistance if available
        if 'Rolling_Low' in stock_df.columns and 'Rolling_High' in stock_df.columns:
            # Use recent support and resistance levels
            support_level = stock_df['Rolling_Low'].iloc[-1]
            resistance_level = stock_df['Rolling_High'].iloc[-1]
            
            ax1.axhline(y=support_level, color='green', linestyle='--', alpha=0.7, label=f'Support ({support_level:.2f})')
            ax1.axhline(y=resistance_level, color='red', linestyle='--', alpha=0.7, label=f'Resistance ({resistance_level:.2f})')
        
        # Configure main plot
        if title:
            ax1.set_title(title, fontsize=16)
        else:
            stock_symbol = stock_df['Symbol'].iloc[0] if 'Symbol' in stock_df.columns else 'Stock'
            ax1.set_title(f"{stock_symbol} Price Trend Analysis", fontsize=16)
            
        ax1.set_ylabel('Price', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper left')
        
        # Create volume subplot
        ax2 = fig.add_subplot(gs[1], sharex=ax1)
        
        if 'Volume' in stock_df.columns:
            # Plot volume
            ax2.bar(stock_df['Date'], stock_df['Volume'], color='blue', alpha=0.5)
            ax2.set_ylabel('Volume', fontsize=12)
            
            # Calculate and plot volume moving average
            stock_df['Volume_MA'] = stock_df['Volume'].rolling(window=20).mean()
            ax2.plot(stock_df['Date'], stock_df['Volume_MA'], color='red', alpha=0.8)
        
        # Configure volume plot
        ax2.grid(True, alpha=0.3)
        ax2.set_xlabel('Date', fontsize=12)
        
        plt.tight_layout()
        return fig
    
    def plot_candlestick(self, stock_df, title=None):
        """Create interactive candlestick chart using Plotly"""
        if stock_df.empty or not all(col in stock_df.columns for col in ['Open', 'High', 'Low', 'Close']):
            print("No valid stock data for candlestick visualization")
            return None
        
        # Create subplot
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                           vertical_spacing=0.03, subplot_titles=('Candlestick', 'Volume'),
                           row_width=[0.2, 0.7])
        
        # Get the stock symbol
        stock_symbol = stock_df['Symbol'].iloc[0] if 'Symbol' in stock_df.columns else 'Stock'
        
        # Add candlestick trace
        fig.add_trace(go.Candlestick(x=stock_df['Date'],
                                     open=stock_df['Open'],
                                     high=stock_df['High'],
                                     low=stock_df['Low'],
                                     close=stock_df['Close'],
                                     name='Price'),
                     row=1, col=1)
        
        # Add volume trace
        if 'Volume' in stock_df.columns:
            fig.add_trace(go.Bar(x=stock_df['Date'], y=stock_df['Volume'], name='Volume', marker_color='blue', opacity=0.5),
                         row=2, col=1)
            
            # Add volume moving average
            if 'Volume_MA' not in stock_df.columns:
                stock_df['Volume_MA'] = stock_df['Volume'].rolling(window=20).mean()
                
            fig.add_trace(go.Scatter(x=stock_df['Date'], y=stock_df['Volume_MA'], name='Volume MA', line=dict(color='red', width=1.5)),
                         row=2, col=1)
        
        # Add moving averages if available
        if 'MA20' in stock_df.columns:
            fig.add_trace(go.Scatter(x=stock_df['Date'], y=stock_df['MA20'], name='20-day MA', line=dict(color='orange', width=1.5)),
                         row=1, col=1)
            
        if 'MA50' in stock_df.columns:
            fig.add_trace(go.Scatter(x=stock_df['Date'], y=stock_df['MA50'], name='50-day MA', line=dict(color='green', width=1.5)),
                         row=1, col=1)
        
        # Update layout
        if title:
            chart_title = title
        else:
            chart_title = f"{stock_symbol} Candlestick Chart"
            
        fig.update_layout(
            title=chart_title,
            yaxis_title='Price',
            xaxis_rangeslider_visible=False,
            height=800,
            width=1000,
            showlegend=True
        )
        
        # Add support and resistance lines if available
        if 'Rolling_Low' in stock_df.columns and 'Rolling_High' in stock_df.columns:
            support_level = stock_df['Rolling_Low'].iloc[-1]
            resistance_level = stock_df['Rolling_High'].iloc[-1]
            
            fig.add_shape(
                type="line", line_color="green", line_dash="dash",
                x0=stock_df['Date'].iloc[0], y0=support_level,
                x1=stock_df['Date'].iloc[-1], y1=support_level,
                opacity=0.7, row=1, col=1
            )
            
            fig.add_shape(
                type="line", line_color="red", line_dash="dash",
                x0=stock_df['Date'].iloc[0], y0=resistance_level,
                x1=stock_df['Date'].iloc[-1], y1=resistance_level,
                opacity=0.7, row=1, col=1
            )
            
            # Add annotations for support and resistance
            fig.add_annotation(
                x=stock_df['Date'].iloc[-5], y=support_level,
                text=f"Support: {support_level:.2f}",
                showarrow=False, font=dict(color="green"),
                xanchor="right", row=1, col=1
            )
            
            fig.add_annotation(
                x=stock_df['Date'].iloc[-5], y=resistance_level,
                text=f"Resistance: {resistance_level:.2f}",
                showarrow=False, font=dict(color="red"),
                xanchor="right", row=1, col=1
            )
        
        return fig
    
    def plot_correlation_heatmap(self, corr_matrix, title=None):
        """Plot correlation heatmap between stocks"""
        if corr_matrix.empty:
            print("Empty correlation matrix, cannot create heatmap")
            return None
        
        # Create a figure
        plt.figure(figsize=(12, 10))
        
        # Generate heatmap
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        heatmap = sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', 
                             vmin=-1, vmax=1, center=0, fmt='.2f', 
                             linewidths=.5, cbar_kws={'shrink': .8})
        
        # Set title
        if title:
            plt.title(title, fontsize=16)
        else:
            plt.title('Stock Price Correlation Heatmap', fontsize=16)
            
        plt.tight_layout()
        return plt.gcf()
    
    def plot_volatility_indicators(self, stock_df, title=None):
        """Plot volatility indicators including Bollinger Bands"""
        if stock_df.empty or 'Close' not in stock_df.columns:
            print("No valid stock data for volatility visualization")
            return None
        
        # Create figure
        fig = plt.figure(figsize=(14, 12))
        gs = fig.add_gridspec(3, 1, height_ratios=[3, 1, 1])
        
        # Main price plot with Bollinger Bands
        ax1 = fig.add_subplot(gs[0])
        
        # Plot price
        ax1.plot(stock_df['Date'], stock_df['Close'], label='Close Price', color='blue', linewidth=2)
        
        # Plot Bollinger Bands if available
        if all(col in stock_df.columns for col in ['MA', 'Upper_Band', 'Lower_Band']):
            ax1.plot(stock_df['Date'], stock_df['MA'], label='Moving Average', color='red', linewidth=1.5)
            ax1.plot(stock_df['Date'], stock_df['Upper_Band'], label='Upper Band', color='green', linewidth=1)
            ax1.plot(stock_df['Date'], stock_df['Lower_Band'], label='Lower Band', color='green', linewidth=1)
            ax1.fill_between(stock_df['Date'], stock_df['Upper_Band'], stock_df['Lower_Band'], alpha=0.1, color='green')
        
        # Configure main plot
        stock_symbol = stock_df['Symbol'].iloc[0] if 'Symbol' in stock_df.columns else 'Stock'
        if title:
            ax1.set_title(title, fontsize=16)
        else:
            ax1.set_title(f"{stock_symbol} Volatility Analysis", fontsize=16)
            
        ax1.set_ylabel('Price', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper left')
        
        # Bollinger Band Width subplot
        ax2 = fig.add_subplot(gs[1], sharex=ax1)
        
        if 'BBW' in stock_df.columns:
            ax2.plot(stock_df['Date'], stock_df['BBW'], color='purple', linewidth=1.5)
            ax2.set_ylabel('BB Width', fontsize=12)
            ax2.grid(True, alpha=0.3)
            ax2.axhline(y=stock_df['BBW'].mean(), color='red', linestyle='--', alpha=0.7, 
                       label=f'Avg: {stock_df["BBW"].mean():.4f}')
            ax2.legend(loc='upper left')
        
        # MACD subplot
        ax3 = fig.add_subplot(gs[2], sharex=ax1)
        
        if all(col in stock_df.columns for col in ['MACD', 'Signal']):
            ax3.plot(stock_df['Date'], stock_df['MACD'], label='MACD', color='blue', linewidth=1.5)
            ax3.plot(stock_df['Date'], stock_df['Signal'], label='Signal', color='red', linewidth=1.5)
            ax3.bar(stock_df['Date'], stock_df['MACD'] - stock_df['Signal'], color='green', alpha=0.5, label='Histogram')
            ax3.set_ylabel('MACD', fontsize=12)
            ax3.grid(True, alpha=0.3)
            ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax3.legend(loc='upper left')
        
        ax3.set_xlabel('Date', fontsize=12)
        
        plt.tight_layout()
        return fig
    
    def plot_sentiment_wordcloud(self, news_df, title=None):
        """Generate word cloud from news headlines with sentiment coloring"""
        if news_df.empty or 'headline' not in news_df.columns:
            print("No valid news data for wordcloud visualization")
            return None
        
        # Combine all headlines into a single text
        text = ' '.join(news_df['headline'].tolist())
        
        # Create and configure the WordCloud object
        wordcloud = WordCloud(
            width=800, height=400,
            background_color='white',
            max_words=100,
            colormap='viridis',
            contour_width=1,
            contour_color='black'
        ).generate(text)
        
        # Create a figure
        plt.figure(figsize=(16, 8))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        
        if title:
            plt.title(title, fontsize=16)
        else:
            plt.title('Financial News Word Cloud', fontsize=16)
        
        return plt.gcf()
    
    def plot_sentiment_distribution(self, sentiment_df, title=None):
        """Plot sentiment distribution from news analysis"""
        if sentiment_df.empty or 'sentiment_category' not in sentiment_df.columns:
            print("No valid sentiment data for visualization")
            return None
        
        # Create a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Plot sentiment distribution in a pie chart
        sentiment_counts = sentiment_df['sentiment_category'].value_counts()
        colors = {'Positive': 'green', 'Neutral': 'gray', 'Negative': 'red'}
        sentiment_counts.plot.pie(
            ax=ax1,
            autopct='%1.1f%%',
            startangle=90,
            shadow=True,
            colors=[colors.get(c, 'blue') for c in sentiment_counts.index],
            wedgeprops={'edgecolor': 'black', 'linewidth': 1}
        )
        ax1.set_ylabel('')
        
        # Plot sentiment histogram
        if 'sentiment' in sentiment_df.columns:
            ax2.hist(sentiment_df['sentiment'], bins=20, color='skyblue', edgecolor='black')
            ax2.axvline(x=0, color='gray', linestyle='--', alpha=0.7)
            ax2.axvline(x=sentiment_df['sentiment'].mean(), color='red', linestyle='-', 
                      label=f'Mean: {sentiment_df["sentiment"].mean():.4f}')
            ax2.set_xlabel('Sentiment Score')
            ax2.set_ylabel('Count')
            ax2.legend()
        
        # Set title
        if title:
            fig.suptitle(title, fontsize=16)
        else:
            fig.suptitle('Financial News Sentiment Analysis', fontsize=16)
        
        plt.tight_layout()
        return fig

class FinancialAnalysisSystem:
    def __init__(self):
        self.stock_data = {}
        self.news_data = pd.DataFrame()
        self.processed_data = {}
        self.websites = [
            'https://www.moneycontrol.com/',
            'https://economictimes.indiatimes.com/markets',
            'https://www.nseindia.com/',
            'https://www.bseindia.com/',
            'https://www.screener.in/',
        ]
        # Default stocks to analyze
        self.target_stocks = ['RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ICICIBANK']
        # Initialize web scraper
        self.scraper = WebScraper()
        # Initialize data processor
        self.processor = DataProcessor()
        # Initialize pattern analyzer
        self.analyzer = PatternAnalyzer()
        # Initialize visualizer
        self.visualizer = Visualizer()
        # Store analysis results
        self.analysis_results = {}
        self.index_data = {}
        self.company_metrics = {}

    def display_menu(self):
        print("\n========== Financial Market Analysis System ==========")
        print("1. Web Scraping (Data Collection)")
        print("2. Data Processing & Cleaning")
        print("3. Financial Pattern Analysis")
        print("4. Data Visualization")
        print("5. AI Model for Trend Prediction (Bonus)")
        print("6. Exit")
        print("======================================================")
        
        choice = input("Enter your choice (1-6): ")
        
        if choice == '1':
            self.web_scraping_menu()
        elif choice == '2':
            self.data_processing_menu()
        elif choice == '3':
            self.pattern_analysis_menu()
        elif choice == '4':
            self.visualization_menu()
        elif choice == '5':
            self.ml_prediction_menu()
        elif choice == '6':
            print("Thank you for using the Financial Market Analysis System.")
            return False
        else:
            print("Invalid choice! Please try again.")
        
        return True
    
    def web_scraping_menu(self):
        print("\n----- Web Scraping Menu -----")
        print("1. Scrape Stock Price Data")
        print("2. Scrape Index Values")
        print("3. Scrape Company Performance Metrics")
        print("4. Scrape Financial News")
        print("5. Customize Websites & Stocks")
        print("6. Return to Main Menu")
        
        choice = input("Enter your choice (1-6): ")
        
        if choice == '1':
            print("Scraping stock price data...")
            self.scrape_stock_prices()
        elif choice == '2':
            print("Scraping index values...")
            self.scrape_indices()
        elif choice == '3':
            print("Scraping company performance metrics...")
            self.scrape_metrics()
        elif choice == '4':
            print("Scraping financial news...")
            self.scrape_news()
        elif choice == '5':
            self.customize_sources()
        elif choice == '6':
            return
        else:
            print("Invalid choice! Please try again.")
        
        self.web_scraping_menu()
    
    def scrape_stock_prices(self):
        """Scrape stock price data for all target stocks"""
        print(f"Scraping stock prices for: {', '.join(self.target_stocks)}")
        
        for stock in self.target_stocks:
            # Get current price
            current_price_data = self.scraper.scrape_stock_price(stock)
            
            # Get historical data
            historical_data = self.scraper.scrape_historical_data(stock)
            
            # Store data
            self.stock_data[stock] = {
                'current': current_price_data,
                'historical': historical_data
            }
        
        print(f"Successfully scraped stock price data for {len(self.stock_data)} stocks")
        
        # Display sample of the data
        if self.stock_data:
            sample_stock = list(self.stock_data.keys())[0]
            print(f"\nSample historical data for {sample_stock}:")
            print(self.stock_data[sample_stock]['historical'].head())
    
    def scrape_indices(self):
        """Scrape index values"""
        self.index_data = self.scraper.scrape_index_values()
        
        print("\nIndex values:")
        for index, data in self.index_data.items():
            print(f"{index}: {data['current_value']} ({'+' if data['percent_change'] > 0 else ''}{data['percent_change']}%)")
    
    def scrape_metrics(self):
        """Scrape company performance metrics"""
        print(f"Scraping performance metrics for: {', '.join(self.target_stocks)}")
        
        for stock in self.target_stocks:
            metrics = self.scraper.scrape_company_metrics(stock)
            self.company_metrics[stock] = metrics
        
        print(f"Successfully scraped metrics for {len(self.company_metrics)} companies")
        
        # Display sample of the data
        if self.company_metrics:
            sample_stock = list(self.company_metrics.keys())[0]
            print(f"\nSample metrics for {sample_stock}:")
            for key, value in self.company_metrics[sample_stock].items():
                print(f"{key}: {value}")
    
    def scrape_news(self):
        """Scrape financial news from supported websites"""
        print("Scraping financial news articles...")
        self.news_data = self.scraper.scrape_financial_news(num_articles=15)
        
        print(f"Successfully scraped {len(self.news_data)} news articles")
        
        # Display sample of the news data
        if not self.news_data.empty:
            print("\nSample news headlines:")
            for idx, row in self.news_data.head(5).iterrows():
                print(f"{row['source']} ({row['date']}): {row['headline']}")
    
    def customize_sources(self):
        print("\nCurrent websites:")
        for i, website in enumerate(self.websites):
            print(f"{i+1}. {website}")
        
        print("\nCurrent stocks:")
        for i, stock in enumerate(self.target_stocks):
            print(f"{i+1}. {stock}")
        
        print("\n1. Add a website")
        print("2. Remove a website")
        print("3. Add a stock")
        print("4. Remove a stock")
        print("5. Return to Scraping Menu")
        
        choice = input("Enter your choice (1-5): ")
        
        if choice == '1':
            website = input("Enter the website URL: ")
            self.websites.append(website)
            print(f"Added {website} to the list")
        elif choice == '2':
            idx = int(input("Enter the website number to remove: ")) - 1
            if 0 <= idx < len(self.websites):
                removed = self.websites.pop(idx)
                print(f"Removed {removed} from the list")
            else:
                print("Invalid website number")
        elif choice == '3':
            stock = input("Enter the stock symbol: ")
            self.target_stocks.append(stock.upper())
            print(f"Added {stock.upper()} to the list")
        elif choice == '4':
            idx = int(input("Enter the stock number to remove: ")) - 1
            if 0 <= idx < len(self.target_stocks):
                removed = self.target_stocks.pop(idx)
                print(f"Removed {removed} from the list")
            else:
                print("Invalid stock number")
        elif choice == '5':
            return
        else:
            print("Invalid choice! Please try again.")
        
        self.customize_sources()
        
    def data_processing_menu(self):
        print("\n----- Data Processing Menu -----")
        print("1. Convert HTML to DataFrames")
        print("2. Handle Missing Values")
        print("3. Remove Duplicates")
        print("4. Standardize Numerical Values")
        print("5. Process All Data")
        print("6. Return to Main Menu")
        
        choice = input("Enter your choice (1-6): ")
        
        if choice == '1':
            print("Converting to DataFrames...")
            self.convert_to_dataframes()
        elif choice == '2':
            print("Handling missing values...")
            self.handle_missing_values()
        elif choice == '3':
            print("Removing duplicates...")
            self.remove_duplicates()
        elif choice == '4':
            print("Standardizing numerical values...")
            self.standardize_numerical_values()
        elif choice == '5':
            print("Processing all data...")
            self.process_all_data()
        elif choice == '6':
            return
        else:
            print("Invalid choice! Please try again.")
        
        self.data_processing_menu()
    
    def convert_to_dataframes(self):
        """Convert all raw data to DataFrames"""
        # Check if we have data to process
        if not self.stock_data:
            print("No stock data available. Please scrape data first.")
            return
        
        # Process stock data
        if 'stock' not in self.processed_data:
            self.processed_data['stock'] = self.processor.convert_to_dataframe(self.stock_data, 'stock')
            if not self.processed_data['stock'].empty:
                print(f"Successfully converted stock data to DataFrame with shape {self.processed_data['stock'].shape}")
                print(self.processed_data['stock'].head())
        else:
            print("Stock data already converted to DataFrame")
            
        # Process company metrics data
        if self.company_metrics and 'metrics' not in self.processed_data:
            self.processed_data['metrics'] = self.processor.convert_to_dataframe(self.company_metrics, 'metrics')
            if not self.processed_data['metrics'].empty:
                print(f"\nSuccessfully converted metrics data to DataFrame with shape {self.processed_data['metrics'].shape}")
                print(self.processed_data['metrics'].head())
        elif 'metrics' in self.processed_data:
            print("Metrics data already converted to DataFrame")
        else:
            print("No metrics data available to convert")
            
        # Process index data
        if self.index_data and 'index' not in self.processed_data:
            self.processed_data['index'] = self.processor.convert_to_dataframe(self.index_data, 'index')
            if not self.processed_data['index'].empty:
                print(f"\nSuccessfully converted index data to DataFrame with shape {self.processed_data['index'].shape}")
                print(self.processed_data['index'].head())
        elif 'index' in self.processed_data:
            print("Index data already converted to DataFrame")
        else:
            print("No index data available to convert")
            
        # Process news data
        if not self.news_data.empty and 'news' not in self.processed_data:
            self.processed_data['news'] = self.processor.convert_to_dataframe(self.news_data, 'news')
            if not self.processed_data['news'].empty:
                print(f"\nSuccessfully converted news data to DataFrame with shape {self.processed_data['news'].shape}")
                print(self.processed_data['news'].head())
        elif 'news' in self.processed_data:
            print("News data already converted to DataFrame")
        else:
            print("No news data available to convert")
    
    def handle_missing_values(self):
        """Handle missing values in all DataFrames"""
        # Check if we have processed data
        if not self.processed_data:
            print("No processed data available. Please convert data to DataFrames first.")
            return
            
        for data_type, df in self.processed_data.items():
            print(f"\nHandling missing values in {data_type} data:")
            method = 'forward_fill' if data_type == 'stock' else 'impute'
            self.processed_data[data_type] = self.processor.handle_missing_values(df, method=method)
    
    def remove_duplicates(self):
        """Remove duplicates from all DataFrames"""
        # Check if we have processed data
        if not self.processed_data:
            print("No processed data available. Please convert data to DataFrames first.")
            return
            
        for data_type, df in self.processed_data.items():
            print(f"\nRemoving duplicates from {data_type} data:")
            self.processed_data[data_type] = self.processor.remove_duplicates(df)
    
    def standardize_numerical_values(self):
        """Standardize numerical values in all DataFrames"""
        # Check if we have processed data
        if not self.processed_data:
            print("No processed data available. Please convert data to DataFrames first.")
            return
            
        data_types_to_standardize = ['stock', 'metrics']
        for data_type in data_types_to_standardize:
            if data_type in self.processed_data:
                print(f"\nStandardizing numerical values in {data_type} data:")
                
                if data_type == 'stock':
                    # For stock data, standardize only specific columns
                    columns_to_standardize = ['Open', 'High', 'Low', 'Close', 'Volume']
                    cols_present = [col for col in columns_to_standardize if col in self.processed_data[data_type].columns]
                    
                    if cols_present:
                        self.processed_data[data_type + '_standardized'] = self.processor.standardize_values(
                            self.processed_data[data_type], columns=cols_present
                        )
                        print(f"Created standardized version in '{data_type}_standardized'")
                    else:
                        print(f"No columns to standardize in {data_type} data")
                else:
                    # For other data types, auto-detect numeric columns
                    self.processed_data[data_type + '_standardized'] = self.processor.standardize_values(
                        self.processed_data[data_type]
                    )
                    print(f"Created standardized version in '{data_type}_standardized'")
    
    def process_all_data(self):
        """Run the full data processing pipeline on all data"""
        # Check if we have data to process
        if not self.stock_data and not self.company_metrics and not self.index_data and self.news_data.empty:
            print("No data available to process. Please scrape data first.")
            return
        
        # Process stock data
        if self.stock_data:
            self.processed_data['stock'] = self.processor.process_pipeline(self.stock_data, 'stock')
        
        # Process company metrics
        if self.company_metrics:
            self.processed_data['metrics'] = self.processor.process_pipeline(self.company_metrics, 'metrics')
        
        # Process index data
        if self.index_data:
            self.processed_data['index'] = self.processor.process_pipeline(self.index_data, 'index')
        
        # Process news data
        if not self.news_data.empty:
            self.processed_data['news'] = self.processor.process_pipeline(self.news_data, 'news')
        
        print("\nAll data processed successfully!")
        print(f"Available processed datasets: {', '.join(self.processed_data.keys())}")
    
    def pattern_analysis_menu(self):
        print("\n----- Pattern Analysis Menu -----")
        print("1. Trend Identification")
        print("2. Volatility Detection")
        print("3. Sentiment Analysis on News")
        print("4. Correlation with Financial Events")
        print("5. Run All Analyses")
        print("6. Return to Main Menu")
        
        choice = input("Enter your choice (1-6): ")
        
        if choice == '1':
            print("Identifying trends...")
            self.identify_trends()
        elif choice == '2':
            print("Detecting volatility...")
            self.detect_volatility()
        elif choice == '3':
            print("Analyzing sentiment from news...")
            self.analyze_news_sentiment()
        elif choice == '4':
            print("Correlating with financial events...")
            self.analyze_correlations()
        elif choice == '5':
            print("Running all analyses...")
            self.run_all_analyses()
        elif choice == '6':
            return
        else:
            print("Invalid choice! Please try again.")
        
        self.pattern_analysis_menu()
    
    def identify_trends(self):
        """Identify trends in stock data"""
        # Check if we have processed stock data
        if 'stock' not in self.processed_data or self.processed_data['stock'].empty:
            print("No processed stock data available. Please scrape and process data first.")
            return
        
        # Get available stock symbols
        available_stocks = self.processed_data['stock']['Symbol'].unique().tolist()
        
        if not available_stocks:
            print("No stock symbols found in the data")
            return
        
        print("\nAvailable stocks:")
        for i, stock in enumerate(available_stocks):
            print(f"{i+1}. {stock}")
        
        # Get user selection
        try:
            idx = int(input("\nEnter the stock number to analyze (0 for all): ")) - 1
            
            if idx == -1:
                print("Analyzing trends for all stocks...")
                stock_symbol = None
            elif 0 <= idx < len(available_stocks):
                stock_symbol = available_stocks[idx]
                print(f"\nAnalyzing trends for {stock_symbol}...")
            else:
                print("Invalid selection, analyzing all stocks")
                stock_symbol = None
        except ValueError:
            print("Invalid input, analyzing all stocks")
            stock_symbol = None
        
        # Analyze trends
        window = 20  # Default window size
        try:
            window_input = input("\nEnter window size for analysis (default: 20): ")
            if window_input.strip():
                window = int(window_input)
        except ValueError:
            print("Invalid window size, using default")
        
        trend_results = self.analyzer.identify_trends(
            self.processed_data['stock'], 
            stock_symbol=stock_symbol,
            window=window
        )
        
        if trend_results:
            # Store results for visualization
            if stock_symbol:
                self.analysis_results[f"trend_{stock_symbol}"] = trend_results
            else:
                self.analysis_results["trend_all"] = trend_results
    
    def detect_volatility(self):
        """Detect volatility in stock data"""
        # Check if we have processed stock data
        if 'stock' not in self.processed_data or self.processed_data['stock'].empty:
            print("No processed stock data available. Please scrape and process data first.")
            return
        
        # Get available stock symbols
        available_stocks = self.processed_data['stock']['Symbol'].unique().tolist()
        
        if not available_stocks:
            print("No stock symbols found in the data")
            return
        
        print("\nAvailable stocks:")
        for i, stock in enumerate(available_stocks):
            print(f"{i+1}. {stock}")
        
        # Get user selection
        try:
            idx = int(input("\nEnter the stock number to analyze (0 for all): ")) - 1
            
            if idx == -1:
                print("Analyzing volatility for all stocks...")
                stock_symbol = None
            elif 0 <= idx < len(available_stocks):
                stock_symbol = available_stocks[idx]
                print(f"\nAnalyzing volatility for {stock_symbol}...")
            else:
                print("Invalid selection, analyzing all stocks")
                stock_symbol = None
        except ValueError:
            print("Invalid input, analyzing all stocks")
            stock_symbol = None
        
        # Analyze volatility
        window = 20  # Default window size
        try:
            window_input = input("\nEnter window size for analysis (default: 20): ")
            if window_input.strip():
                window = int(window_input)
        except ValueError:
            print("Invalid window size, using default")
        
        volatility_results = self.analyzer.detect_volatility(
            self.processed_data['stock'], 
            stock_symbol=stock_symbol,
            window=window
        )
        
        if volatility_results:
            # Store results for visualization
            if stock_symbol:
                self.analysis_results[f"volatility_{stock_symbol}"] = volatility_results
            else:
                self.analysis_results["volatility_all"] = volatility_results
    
    def analyze_news_sentiment(self):
        """Analyze sentiment from financial news"""
        # Check if we have news data
        if 'news' not in self.processed_data or self.processed_data['news'].empty:
            print("No processed news data available. Please scrape and process news data first.")
            return
        
        print("Analyzing sentiment from financial news...")
        
        sentiment_results = self.analyzer.analyze_sentiment(self.processed_data['news'])
        
        if sentiment_results:
            # Store results for visualization
            self.analysis_results["sentiment"] = sentiment_results
    
    def analyze_correlations(self):
        """Analyze correlations between stocks"""
        # Check if we have processed stock data
        if 'stock' not in self.processed_data or self.processed_data['stock'].empty:
            print("No processed stock data available. Please scrape and process data first.")
            return
        
        print("Analyzing correlations between stocks...")
        
        # For now, we don't have events data, so pass None
        correlation_results = self.analyzer.analyze_correlations(
            self.processed_data['stock'], 
            events_df=None
        )
        
        if correlation_results:
            # Store results for visualization
            self.analysis_results["correlations"] = correlation_results
    
    def run_all_analyses(self):
        """Run all pattern analyses"""
        print("Running all pattern analyses...")
        
        print("\n1. Trend Identification")
        self.identify_trends()
        
        print("\n2. Volatility Detection")
        self.detect_volatility()
        
        print("\n3. Sentiment Analysis")
        self.analyze_news_sentiment()
        
        print("\n4. Correlation Analysis")
        self.analyze_correlations()
        
        print("\nAll analyses completed!")
        print(f"Available analysis results: {', '.join(self.analysis_results.keys())}")
    
    def visualization_menu(self):
        print("\n----- Visualization Menu -----")
        print("1. Stock Trend Graphs")
        print("2. Correlation Heatmaps")
        print("3. Candlestick Charts")
        print("4. Sentiment Word Clouds")
        print("5. Volatility Indicators")
        print("6. Return to Main Menu")
        
        choice = input("Enter your choice (1-6): ")
        
        if choice == '1':
            print("Generating stock trend graphs...")
            self.visualize_stock_trends()
        elif choice == '2':
            print("Generating correlation heatmaps...")
            self.visualize_correlations()
        elif choice == '3':
            print("Generating candlestick charts...")
            self.visualize_candlestick()
        elif choice == '4':
            print("Generating sentiment visualizations...")
            self.visualize_sentiment()
        elif choice == '5':
            print("Generating volatility indicators...")
            self.visualize_volatility()
        elif choice == '6':
            return
        else:
            print("Invalid choice! Please try again.")
        
        self.visualization_menu()
    
    def visualize_stock_trends(self):
        """Visualize stock price trends"""
        # Check if we have trend analysis results
        if not any(key.startswith('trend_') for key in self.analysis_results.keys()):
            print("No trend analysis results available. Please run trend analysis first.")
            return
        
        # Get available trend analyses
        trend_keys = [key for key in self.analysis_results.keys() if key.startswith('trend_')]
        
        print("\nAvailable trend analyses:")
        for i, key in enumerate(trend_keys):
            print(f"{i+1}. {key}")
        
        # Get user selection
        try:
            idx = int(input("\nEnter the trend analysis number to visualize: ")) - 1
            
            if 0 <= idx < len(trend_keys):
                selected_key = trend_keys[idx]
                trend_data = self.analysis_results[selected_key]['data']
                
                # Generate visualization
                fig = self.visualizer.plot_stock_trends(trend_data, title=f"Stock Trend Analysis: {selected_key}")
                
                # Display the figure
                plt.figure(fig.number)
                plt.show()
            else:
                print("Invalid selection")
        except ValueError:
            print("Invalid input")
            return
    
    def visualize_correlations(self):
        """Visualize stock correlations with heatmap"""
        # Check if we have correlation analysis results
        if 'correlations' not in self.analysis_results:
            print("No correlation analysis results available. Please run correlation analysis first.")
            return
        
        # Generate correlation heatmap
        corr_matrix = self.analysis_results['correlations']['correlation_matrix']
        fig = self.visualizer.plot_correlation_heatmap(corr_matrix, title="Stock Price Correlation Heatmap")
        
        # Display the figure
        plt.figure(fig.number)
        plt.show()
    
    def visualize_candlestick(self):
        """Visualize stock data with interactive candlestick chart"""
        # Check if we have volatility analysis results (which contain the needed data)
        volatility_keys = [key for key in self.analysis_results.keys() if key.startswith('volatility_')]
        trend_keys = [key for key in self.analysis_results.keys() if key.startswith('trend_')]
        
        if not volatility_keys and not trend_keys:
            print("No volatility or trend analysis results available. Please run volatility or trend analysis first.")
            return
        
        # Combine available analyses
        available_keys = volatility_keys + trend_keys
        
        print("\nAvailable analyses for candlestick chart:")
        for i, key in enumerate(available_keys):
            print(f"{i+1}. {key}")
        
        # Get user selection
        try:
            idx = int(input("\nEnter the analysis number to visualize: ")) - 1
            
            if 0 <= idx < len(available_keys):
                selected_key = available_keys[idx]
                stock_data = self.analysis_results[selected_key]['data']
                
                # Generate candlestick chart
                fig = self.visualizer.plot_candlestick(stock_data, title=f"Candlestick Chart: {selected_key}")
                
                # Display the figure
                fig.show()
            else:
                print("Invalid selection")
        except ValueError:
            print("Invalid input")
            return
    
    def visualize_sentiment(self):
        """Visualize news sentiment analysis results"""
        # Check if we have sentiment analysis results
        if 'sentiment' not in self.analysis_results:
            print("No sentiment analysis results available. Please run sentiment analysis first.")
            return
        
        sentiment_data = self.analysis_results['sentiment']['data']
        
        # Create two visualizations: word cloud and sentiment distribution
        print("Generating sentiment word cloud...")
        wc_fig = self.visualizer.plot_sentiment_wordcloud(sentiment_data, title="Financial News Word Cloud")
        plt.figure(wc_fig.number)
        plt.show()
        
        print("\nGenerating sentiment distribution charts...")
        dist_fig = self.visualizer.plot_sentiment_distribution(sentiment_data, title="Financial News Sentiment Distribution")
        plt.figure(dist_fig.number)
        plt.show()
    
    def visualize_volatility(self):
        """Visualize stock volatility indicators"""
        # Check if we have volatility analysis results
        volatility_keys = [key for key in self.analysis_results.keys() if key.startswith('volatility_')]
        
        if not volatility_keys:
            print("No volatility analysis results available. Please run volatility analysis first.")
            return
        
        print("\nAvailable volatility analyses:")
        for i, key in enumerate(volatility_keys):
            print(f"{i+1}. {key}")
        
        # Get user selection
        try:
            idx = int(input("\nEnter the volatility analysis number to visualize: ")) - 1
            
            if 0 <= idx < len(volatility_keys):
                selected_key = volatility_keys[idx]
                volatility_data = self.analysis_results[selected_key]['data']
                
                # Generate visualization
                fig = self.visualizer.plot_volatility_indicators(
                    volatility_data, 
                    title=f"Volatility Analysis: {selected_key}"
                )
                
                # Display the figure
                plt.figure(fig.number)
                plt.show()
            else:
                print("Invalid selection")
        except ValueError:
            print("Invalid input")
            return
    
    def ml_prediction_menu(self):
        print("\n----- ML Prediction Menu -----")
        print("1. Linear Regression Forecasting")
        print("2. LSTM Time-Series Analysis")
        print("3. Random Forest Prediction")
        print("4. Evaluate Model Performance")
        print("5. Return to Main Menu")
        
        choice = input("Enter your choice (1-5): ")
        
        if choice == '1':
            print("Running linear regression forecast...")
            # Will implement in next part
            print("Feature will be implemented in the next part")
        elif choice == '2':
            print("Running LSTM time-series analysis...")
            # Will implement in next part
            print("Feature will be implemented in the next part")
        elif choice == '3':
            print("Running random forest prediction...")
            # Will implement in next part
            print("Feature will be implemented in the next part")
        elif choice == '4':
            print("Evaluating model performance...")
            # Will implement in next part
            print("Feature will be implemented in the next part")
        elif choice == '5':
            return
        else:
            print("Invalid choice! Please try again.")
        
        self.ml_prediction_menu()

# Run the menu-based application
def main():
    """Main function to run the Financial Analysis System"""
    print("Starting Financial Market Pattern Analysis System...")
    
    # Create instance of the analysis system
    system = FinancialAnalysisSystem()
    
    # Main application loop
    running = True
    while running:
        running = system.display_menu()
    
    print("Exiting the system. Thank you!")

if __name__ == "__main__":
    main() 