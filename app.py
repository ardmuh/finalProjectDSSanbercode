import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist


st.title("Sentiment Analysis of TV Digital")

st.header('Business Understanding')
st.image('img/news.png')
st.write('Per tanggal 2 November 2022, Pemerintahan Indonesia melalui kominfo resmi memberhentikan layanan TV Analog dan beralih ke layanan TV Digital. Tentunya ini akan menimbulkan pro dan kontra dari masyarakat.')
st.write('Project ini bertujuan untuk melihat sentimen masyarakat terhadap peralihan layanan ini. Dataset yang digunakan adalah hasil scraping dari Twitter API dan situs detik.com')

st.header('Data Understanding')
st.subheader('1. Data Collection')
df_tweets = pd.read_csv("data/tweet_tvdigital.csv")
df_news = pd.read_csv("data/news_tvdigital.csv")
option = st.selectbox(label="Data dengan sumber yang ingin ditampilkan",
        options=('None','Tweet', 'News'))
if option == 'Tweet':
    st.dataframe(df_tweets,  use_container_width=True)
elif option == 'News':
    st.dataframe(df_news,  use_container_width=True)
else: 
    pass

st.subheader('2. Data Preprocessing')
if option == 'Tweet':
    df_tweets = pd.read_csv('data/tweet_sentiment.csv')
    st.dataframe(df_tweets[{'tweets' ,'tweet_clean'}])
elif option == 'News':
    df_news = pd.read_csv('data/news_sentiment.csv')
    st.dataframe(df_news[{'judul','judul_clean'}])
else: 
    pass


st.header('Data Processing')
st.subheader('1. Text Data Analysis')
st.text('Text Analysis dilakukan pada dataset tweet dengan melihat distribusi kata dan karakter')

df_tweets = pd.read_csv('data/tweet_sentiment.csv')
col_1, col_2, col_3 = st.columns(3)
with col_1:
    bin_range = np.arange(0, 250, 10)
    g1 = df_tweets['tweet_clean'].str.len().hist(bins=bin_range)
    f1 = g1.figure
    st.pyplot(f1, clear_figure=True)
    st.write('Distribusi jumlah karakter terbanyak berada pada 50 - 70 karakter per tweet.')
    
with col_2:
    bin_range = np.arange(0, 50)
    g2 = df_tweets['tweet_clean'].str.split().map(lambda x: len(x)).hist(bins=bin_range)
    f2 = g2.figure
    st.pyplot(f2 ,clear_figure=True)
    st.write('Distribusi jumlah kata terbanyak pada 7 - 10 kata')

    
with col_3:
    g3 = df_tweets['tweet_clean'].str.split().apply(lambda x : [len(i) for i in x]).map(lambda x: np.mean(x)).hist()
    f3 = g3.figure
    st.pyplot(f3, clear_figure=True)
    st.write('Distribusi jumlah karakter rata - rata adalah 5 per kata.')

col_1, col_2 = st.columns(2)
with col_1:
    token_tweet = df_tweets['tweet_clean'].apply(lambda x: word_tokenize(str(x)))
    tweets = [word for tweet in token_tweet for word in tweet]
    fqdist = FreqDist(tweets)
    most_common_word = fqdist.most_common(30)
    g4 = fqdist.plot(30,cumulative=False)
    fig = g4.figure
    st.pyplot(fig, clear_figure=True)
with col_2:
    st.markdown('Dan untuk distribusi kata, didominasi oleh kata:')
    st.write('1. TV')
    st.write('2. Digital')
    st.write('3. Analog.')


st.subheader('2. Sentiment Analysis with Manual Polarization')
col_1, col_2 = st.columns(2)
with col_1:
    g5 =sns.countplot(data= df_tweets, x='sentiment_MP', order=df_tweets['sentiment_MP'].value_counts().index )
    fig = g5.figure
    st.pyplot(fig, clear_figure=True)
with col_2:
    st.write('Dari distribusi sentiment mengunakan metode manual polarization, sentiment masyarakat terhadap TV Digital didominasi dengan sentiment netral')

st.subheader('3. Sentiment Analysis with Navie Bayes')
col_1, col_2 = st.columns(2)
with col_1:
    g6 =sns.countplot(data= df_tweets, x='sentiment_NB', order=df_tweets['sentiment_NB'].value_counts().index)
    fig = g6.figure
    st.pyplot(fig, clear_figure=True)
with col_2:
    st.write('Dari distribusi sentiment mengunakan metode Naive Bayes, sentiment masyarakat terhadap TV Digital didominasi dengan sentiment positif')


st.header('Analisis')
st.markdown('''Dari teks analisis yang telah dilakukan dapat diketahui bahwa distribusi jumlah karakter terbanyak berada pada 50 - 70 karakter per tweet. Sedangkan distribusi jumlah kata terbanyak pada 7 - 10 kata dengan jumlah karakter rata - rata adalah 5 per kata. Dan untuk distribusi kata, didominasi oleh kata TV dan diikuti kata digital dan analog.
\n Berdasarkan analisis sentiment mengunakan  metode manual polarization dapat diketahui bahwa sentimen masyarakat terhadap peralihan ke TV digital didominasi sentiment netral
\n Sedangkan berdasarkan analisis sentiment mengunakan Naive Bayes dapat diketahui bahwa sentimen masyarakat terhadap peralihan ke TV digital didominasi sentiment positif''')

st.header('Kesimpulan')
st.markdown('''Sentiment masyarakat terhadap peralihan layanan TV Analog ke layanan TV Digital berdasarkan analisis sentimen menggunakan metode naive bayes adalah positif, dengan artian masyarakat mendukung program ini.
\n Hasil ini bukan tanpa cacat, masih banyak catatan yang harus dilakukan agar data yang dianalisis bisa lebih valid. Sample diperbanyak, pembersihan ditingkatkan, dan algoritma diperbaiki.''')

st.subheader('Referensi')
st.markdown('''
- https://github.com/riochr17/Analisis-Sentimen-ID 
- https://github.com/rizalespe/Dataset-Sentimen-Analisis-Bahasa-Indonesia
''')
