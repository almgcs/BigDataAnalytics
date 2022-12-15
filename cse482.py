import altair as alt
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import seaborn as sns
import streamlit as st
import statsmodels.api as sm
import numpy as np
import pydeck as pdk
from pandas.api.types import (
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
)
from scipy.stats import zscore
from sklearn import datasets
from wordcloud import WordCloud, ImageColorGenerator
import plotly.graph_objs as go
import datetime as dt
from PIL import Image
import urllib
import requests

st.set_page_config(layout="wide")

final_data = pd.read_csv('final_data.csv', index_col=0)
#final_data = pd.read_csv('/Users/almgacis/Documents/MSU/CSE_482/Project/manually_tagged_data/final_data.csv', index_col=0)
final_data = final_data[final_data['State'] != 'TRASH']
region_dictionary ={
# Mid West
'IL':'MW',
'IN':'MW',
'IA':'MW',
'KS':'MW',
'MI':'MW',
'MN':'MW',
'MO':'MW',
'NE':'MW',
'ND':'MW',
'OH':'MW',
'SD':'MW',
'WI':'MW',
# North East
'CT':'NE',
'DE':'NE',
'ME':'NE',
'MD':'NE',
'MA':'NE',
'NH':'NE',
'NJ':'NE',
'NY':'NE',
'PA':'NE',
'RI':'NE',
'VT':'NE',
# South East
'AL':'SE',
'AR':'SE',
'FL':'SE',
'GA':'SE',
'KY':'SE',
'LA':'SE',
'MS':'SE',
'NC':'SE',
'SC':'SE',
'TN':'SE',
# South West
'AZ':'SW',
'NM':'SW',
'OK':'SW',
'TX':'SW',
'VA':'SW',
'WV':'SW',
# West
'AK':'W',
'CA':'W',
'CO':'W',
'HI':'W',
'ID':'W',
'MT':'W',
'NV':'W',
'OR':'W',
'UT':'W',
'WA':'W',
'WY':'W'
}
# Add a new column named 'Region'
final_data['Region'] = final_data['State'].map(region_dictionary)

news = pd.read_csv('news.csv')
#news = pd.read_csv('//Users/almgacis/Documents/MSU/CSE_482/Project/news.csv')

st.markdown(""" <style> .font {
font-size:48px ; font-family: 'Cooper Black'} 
</style> """, unsafe_allow_html=True)
st.markdown('<p class="font"> A 30-Day Twitter Sentiment Analysis on "Abortion" in the US</p>', unsafe_allow_html=True)
st.markdown("*by Angelica Louise Gacis, Yena Hong*")

senti = ['All Sentiments', 'Positive Sentiments', 'Negative Sentiments', 'Neutral Sentiments']
region = ['All', 'MW', 'NE', 'SE', 'SW', 'W']
state = ['All',
 'FL',
 'WA',
 'TN',
 'TX',
 'CA',
 'IA',
 'MO',
 'GA',
 'NV',
 'IL',
 'NY',
 'AZ',
 'MT',
 'IN',
 'NJ',
 'NM',
 'NH',
 'MA',
 'KY',
 'OH',
 'LA',
 'PA',
 'DC',
 'WI',
 'VA',
 'MI',
 'AL',
 'NC',
 'MD',
 'AR',
 'MN',
 'CO',
 'NE',
 'SC',
 'OK',
 'OR',
 'DE',
 'CT',
 'KS',
 'MS',
 'WV',
 'RI',
 'UT',
 'ME',
 'VT',
 'WY',
 'AK',
 'ID',
 'HI',
 'ND',
 'SD',
 'PR']

granularity = ['All', 'by Region', 'by State']
granu = st.selectbox("Select Granularity", granularity)
if granu == 'by State':
    state_pts = st.selectbox("Filter by State", state)
    if state_pts == 'All':
        eda_data = final_data
    else:
        eda_data = final_data[(final_data['State']==state_pts)]
elif granu == 'by Region':
    reg_pts = st.selectbox("Filter by Region", region)
    if reg_pts == 'All':
        eda_data = final_data
    else:
        eda_data = final_data[(final_data['Region']==reg_pts)]
else:
    eda_data = final_data

map_pts = st.selectbox("Filter Points in Map", senti)

#-----Word Clouds--------------------------------------------------------------------------------------

# df2 = df[(df['date']>='2019-05-11') & (df['date']<='2019-05-14')]

col1, col2, col3= st.columns(3,gap='small')

# with col1:
positive = final_data[final_data['y_predict']=='positive']
all_pos_tweets = ' '.join(text for text in positive['tweet_clean'])
Mask1 = np.array(Image.open(requests.get('https://www.wpclipart.com/medical/pills/large_colors/drug_big_pill_green.png', stream=True).raw))
image_colors1 = ImageColorGenerator(Mask1)
wc1 = WordCloud(background_color='white', height=1500, width=4000,mask=Mask1).generate(all_pos_tweets)
fig1, ax = plt.subplots(figsize = (10, 20))
ax.imshow(wc1.recolor(color_func=image_colors1), interpolation = 'hamming')
plt.axis('off')
plt.title('Positive')

# with col2:
neutral = final_data[final_data['y_predict']=='neutral']
all_neu_tweets = ' '.join(map(str, (text for text in neutral['tweet_clean'])))
Mask2 = np.array(Image.open(requests.get('https://www.wpclipart.com/medical/pills/large_colors/drug_big_pill_blue.png', stream=True).raw))
image_colors2 = ImageColorGenerator(Mask2)
wc2 = WordCloud(background_color='white', height=1500, width=4000,mask=Mask2).generate(all_neu_tweets)
# def grey_color_func(word, font_size, position, orientation, random_state=None,
#                 **kwargs):
#     return "hsl(0, 0%%, %d%%)" % random.randint(60, 100)
fig2, ax = plt.subplots(figsize = (10, 20))
ax.imshow(wc2.recolor(color_func=image_colors2), interpolation = 'hamming')
plt.axis('off')
plt.title('Neutral')

# with col3:
negative = final_data[final_data['y_predict']=='negative']
all_neg_tweets = ' '.join(text for text in negative['tweet_clean'])
Mask3 = np.array(Image.open(requests.get('https://www.wpclipart.com/medical/pills/large_colors/drug_big_pill_red.png', stream=True).raw))
image_colors3 = ImageColorGenerator(Mask3)
wc3 = WordCloud(background_color='white', height=1500, width=4000,mask=Mask3).generate(all_neg_tweets)
fig3, ax = plt.subplots(figsize = (10, 20))
ax.imshow(wc3.recolor(color_func=image_colors3), interpolation = 'hamming')
plt.axis('off')
plt.title('Negative')


if map_pts == 'Positive Sentiments':
    with col1:
        st.pyplot(fig1)
elif map_pts == 'Neutral Sentiments':
    with col2:
        st.pyplot(fig2)
elif map_pts == 'Negative Sentiments':
    with col3:
        st.pyplot(fig3)
else:
    with col1:
        st.pyplot(fig1)
    with col2:
        st.pyplot(fig2)
    with col3:
        st.pyplot(fig3)

#-----Geolocation--------------------------------------------------------------------------------------

# col3, col4, col= st.columns(2,gap='small')

geo_data_0 = eda_data[eda_data['y_predict']=='positive'].iloc[:,1:3]
geo_data_1 = eda_data[eda_data['y_predict']=='negative'].iloc[:,1:3]
geo_data_2 = eda_data[eda_data['y_predict']=='neutral'].iloc[:,1:3]
if map_pts == 'Positive Sentiments':
        st.pydeck_chart(pdk.Deck(
        map_style=None,
        initial_view_state=pdk.ViewState(
            latitude=44.978718,
            longitude=-84.515887,
            zoom=1,
            pitch=0,
        ),
        layers=[
            pdk.Layer(
                'ScatterplotLayer',
                data=geo_data_0,
                get_position='[Longitude, Latitude]',
                get_color='[111, 215, 121, 160]', #green
                opacity=0.5,
                get_radius=20000,
            ),
        ],
    ))
elif map_pts == 'Negative Sentiments':
        st.pydeck_chart(pdk.Deck(
        map_style=None,
        initial_view_state=pdk.ViewState(
            latitude=44.978718,
            longitude=-84.515887,
            zoom=1,
            pitch=0,
        ),
        layers=[
            pdk.Layer(
                'ScatterplotLayer',
                data=geo_data_1,
                get_position='[Longitude, Latitude]',
                get_color='[255, 131, 131, 160]', #red
                opacity=0.5,
                get_radius=20000,
            ),
        ],
    ))
elif map_pts == 'Neutral Sentiments':
        st.pydeck_chart(pdk.Deck(
        map_style=None,
        initial_view_state=pdk.ViewState(
            latitude=44.978718,
            longitude=-84.515887,
            zoom=1,
            pitch=0,
        ),
        layers=[
            pdk.Layer(
                'ScatterplotLayer',
                data=geo_data_2,
                get_position='[Longitude, Latitude]',
                get_color='[190, 190, 190, 100]', #grey
                opacity=0.5,
                get_radius=20000,
            ),
        ],
    ))
else:
    st.pydeck_chart(pdk.Deck(
        map_style=None,
        initial_view_state=pdk.ViewState(
            latitude=44.978718,
            longitude=-84.515887,
            zoom=1,
            pitch=0,
        ),
        layers=[
            pdk.Layer(
                'ScatterplotLayer',
                data=geo_data_0,
                get_position='[Longitude, Latitude]',
                get_color='[111, 215, 121, 160]', #green
                opacity=0.5,
                get_radius=20000,
            ),
            pdk.Layer(
                'ScatterplotLayer',
                data=geo_data_1,
                get_position='[Longitude, Latitude]',
                get_color='[255, 131, 131, 160]', #red
                opacity=0.5,
                get_radius=20000,
            ),
            pdk.Layer(
                'ScatterplotLayer',
                data=geo_data_2,
                get_position='[Longitude, Latitude]',
                get_color='[190, 190, 190, 100]', #grey
                opacity=0.5,
                get_radius=20000,
            ),
        ],
    ))

#-----Time Series--------------------------------------------------------------------------------------

col3, col4= st.columns(2,gap='small')

with col3:
    neg = eda_data[eda_data['y_predict']=='negative']
    neg = neg.groupby(['date'],as_index=False).count()

    pos = eda_data[eda_data['y_predict']=='positive']
    pos = pos.groupby(['date'],as_index=False).count()

    neu = eda_data[eda_data['y_predict']=='neutral']
    neu = neu.groupby(['date'],as_index=False).count()


    pos = pos[['date','tweet_clean']]
    neg = neg[['date','tweet_clean']]
    neu = neu[['date','tweet_clean']]
    net = pd.concat([pos[['date']], pos[['tweet_clean']]-neg[['tweet_clean']]], axis=1)

    fig_each = go.Figure()
    for col in pos.columns:
        fig_each.add_trace(go.Scatter(x=pos['date'], y=pos['tweet_clean'],
                                 name = 'Positive',
                                 mode = 'lines',
                                 line=dict(shape='linear'),
                                 connectgaps=True,
                                 line_color='green'
                                 )
                     )

    for col in neg.columns:
        fig_each.add_trace(go.Scatter(x=neg['date'], y=neg['tweet_clean'],
                                 name = 'Negative',
                                 mode = 'lines',
                                 line=dict(shape='linear'),
                                 connectgaps=True,
                                 line_color='red'
                                 )
                     )
    for col in neu.columns:
        fig_each.add_trace(go.Scatter(x=neu['date'], y=neu['tweet_clean'],
                                 name = 'Neutral',
                                 mode = 'lines',
                                 line=dict(shape='linear'),
                                 connectgaps=True,
                                 line_color='grey'
                                 )
                     )

    # set the place of legend
    fig_each.update_layout(legend=dict(orientation="h",yanchor="bottom",y=-0.3))
    # remove the duplicate legend
    names = set()
    fig_each.for_each_trace(
        lambda trace:
            trace.update(showlegend=False)
            if (trace.name in names) else names.add(trace.name))
    st.write(fig_each)

with col4:
    fig_net = go.Figure()
    for col in net.columns:
        fig_net.add_trace(go.Scatter(x=net['date'], y=net['tweet_clean'],
                                 name = 'Net Sentiment',
                                 mode = 'lines',
                                 line=dict(shape='linear'),
                                 connectgaps=True,
                                 line_color='blue'
                                 )
                     )
        fig_net.add_shape(type='line',
                    x0='2022-11-10',
                    #min(net['date']),
                    y0=0,
                    x1='2022-12-10',
                    # max(net['date']),
                    y1=0,
                    line=dict(color='grey',dash='dash'),
                    xref='x',
                    yref='y'
            )

    # set the place of legend
    fig_net.update_layout(legend=dict(orientation="h",yanchor="bottom",y=-0.3))
    # remove the duplicate legend
    names = set()
    fig_net.for_each_trace(
        lambda trace:
            trace.update(showlegend=False)
            if (trace.name in names) else names.add(trace.name))
    st.write(fig_net)


#-----News Articles--------------------------------------------------------------------------------------

st.markdown("Top Google Search News Results for 10 Nov - 10 Dec 2022")
st.dataframe(news[['title','author','published_date','link']])

    
