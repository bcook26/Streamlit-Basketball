import pandas as pd
import numpy as np
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import base64
from PIL import Image


logo = Image.open('lebron-dwade.jpg')

st.image(logo, use_column_width=True)

st.title('NBA Player Stats Explorer')

st.markdown('''

This app perfroms a simple webscrape of NBA player stats data. 
* **Python libraries:** base64, numpy, pandas, streamlit, seaborn, matplotlib
* **Data source:** [basketball-reference.com](https://www.basketball-reference.com/)

''') 

st.sidebar.header('User input Features')
selected_year = st.sidebar.selectbox('Year', list(reversed(range(1950,2023))))

# Scraping player stats from the website
@st.cache

def load_data(year):
    url = "https://www.basketball-reference.com/leagues/NBA_" + str(year) + "_per_game.html"
    html = pd.read_html(url, header = 0)
    df = html[0]
    raw = df.drop(df[df.Age == 'Age'].index)
    raw = raw.fillna(0)
    player_stats = raw.drop(['Rk'], axis = 1)
    return player_stats

playerstats = load_data(selected_year)

# siderbar for team selection 

sorted_unique_teams = sorted(playerstats['Tm'].unique())
selected_team = st.sidebar.multiselect('Team', sorted_unique_teams, sorted_unique_teams)

# sidebar for position selection
position_list = ['C', 'PF', 'SF', 'SG', 'PG']
selected_pos = st.sidebar.multiselect('Position', position_list, position_list) 

# filtering data 
selected_team_df = playerstats[(playerstats.Tm.isin(selected_team)) & (playerstats.Pos.isin(selected_pos))]

st.header('Display Player Stats of Selected Teams')
st.write('Data Dimension: ' + str(selected_team_df.shape[0]) + ' rows and ' + str(selected_team_df.shape[1]) + ' columns.')
st.dataframe(selected_team_df)

# Downloading player stats
# https://discuss.streamlit.io/t/how-to-download-file-in-streamlit/1806

def file_download(dataframe):
    csv = dataframe.to_csv(index = False)
    b64 = base64.b64encode(csv.encode()).decode() 
    href = f'<a href="data:file/csv;base64,{b64}" download="playerstats.csv">Download CSV File</a>'
    return href

st.markdown(file_download(selected_team_df), unsafe_allow_html=True)

# Heatmap

if st.button('Intercorrelation Map'):
    st.header('Intercorrelation Matrix Heatmap')
    selected_team_df.to_csv('output.csv', index = False)
    df = pd.read_csv('output.csv')

    corr = df.corr()

    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True
    with sns.axes_style("whitegrid"):
        f, ax = plt.subplots(figsize = (8,6))
        ax = sns.heatmap(corr, mask = mask, vmax = 1, square = True)

    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()

# Future implementations ...

# if st.button('Top Players Ranked - Bar Chart'):
#     st.header('Bar Chart of top 10 Players')
#     selected_team_df.to_csv('output1.csv', index = False)
#     df = pd.read_csv('output1.csv')

#     barplot = sns.boxplot()
# st.write(selected_team_df.columns)