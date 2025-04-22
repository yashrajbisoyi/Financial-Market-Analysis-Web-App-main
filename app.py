import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from wordcloud import WordCloud

# Load financial data (Placeholder for real scraped data)
st.title('Financial Market Analysis')

# Sidebar options
st.sidebar.header('Options')
selected_feature = st.sidebar.selectbox(
    'Select Feature', ['Stock Trends', 'Volatility', 'Sentiment Analysis', 'Correlation Heatmap']
)

# Load sample dataset (Replace with real scraped data)
def load_data():
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=100)
    stocks = ['RELIANCE', 'TCS', 'INFY', 'HDFCBANK', 'ICICIBANK']
    df = pd.DataFrame({
        'Date': np.tile(dates, len(stocks)),
        'Stock': np.repeat(stocks, len(dates)),
        'Close': np.random.uniform(500, 2500, len(dates) * len(stocks))
    })
    return df

data = load_data()

if selected_feature == 'Stock Trends':
    st.subheader('Stock Price Trends')
    stock_selection = st.selectbox('Select Stock', data['Stock'].unique())
    df_filtered = data[data['Stock'] == stock_selection]
    fig = px.line(df_filtered, x='Date', y='Close', title=f'{stock_selection} Price Trends')
    st.plotly_chart(fig)

elif selected_feature == 'Volatility':
    st.subheader('Stock Volatility Analysis')
    df_filtered = data.groupby('Stock')['Close'].std().reset_index()
    fig = px.bar(df_filtered, x='Stock', y='Close', title='Stock Volatility')
    st.plotly_chart(fig)

elif selected_feature == 'Sentiment Analysis':
    st.subheader('Sentiment Analysis on Financial News')
    sample_news = ['Market crashes amid economic downturn', 'Tech stocks surge on strong earnings']
    sentiment_scores = np.random.uniform(-1, 1, len(sample_news))
    df_sentiment = pd.DataFrame({'Headline': sample_news, 'Sentiment': sentiment_scores})
    st.write(df_sentiment)

    # Generate a word cloud
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(sample_news))
    fig, ax = plt.subplots()
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)

elif selected_feature == 'Correlation Heatmap':
    st.subheader('Stock Price Correlation Heatmap')
    pivot_df = data.pivot(index='Date', columns='Stock', values='Close')
    correlation_matrix = pivot_df.corr()
    fig, ax = plt.subplots()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)
