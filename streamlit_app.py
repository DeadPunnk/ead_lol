#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
#import matplotlib.pyplot as plt
#import seaborn as sns
#import altair as alt
from pandas.api.types import (
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
)

st.set_page_config(page_title="LoL Analytics", page_icon="🎮", initial_sidebar_state="expanded")



#@st.cache
def load_and_prep_players():
    dfplayers = pd.read_csv('all_players.csv')
    #dfplayers['KDA'] = str(round(((dfplayers['kills'] + dfplayers['assists'])/dfplayers['deaths']), 2))
    dfplayers = dfplayers.set_index(dfplayers['date'])

    
    return dfplayers

#def load_and_prep_teams():
#    dfteams = pd.read_csv('teamsst.csv')
#    #dfplayers['KDA'] = str(round(((dfplayers['kills'] + dfplayers['assists'])/dfplayers['deaths']), 2))
#    #dfteams = dfteams.set_index(dfplayers['date'])
#    return dfteams

dfplayers = load_and_prep_players()
#dfteams = load_and_prep_teams() 

def color_surplusvalue(val):
    if str(val) == '0':
        color = 'azure'
    elif str(val)[0] == '-':
        color = 'lightpink'
    else:
        color = 'lightgreen'
    return 'background-color: %s' % color


heading_properties = [('font-size', '16px'),('text-align', 'center'),
                      ('color', 'white'),  ('font-weight', 'bold'),
                      ('background', 'green'),('border', '1.2px solid')]

cell_properties = [('font-size', '16px'),('text-align', 'center'), ('color', 'black'), ('text-align', 'center'), ('font-weight', 'bold')]


dfstyle = [{"selector": "th", "props": heading_properties},
               {"selector": "td", "props": cell_properties}]


#tab_player, tab_team = st.tabs(["Player", "Time"])


cols = ['teamname', 'position', 'kills', 'deaths', 'assists', 'KDA', 'totalgold', 'total cs']


coluna1, coluna2 = st.columns([10, 10])


def filter_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a UI on top of a dataframe to let viewers filter columns

    Args:
        df (pd.DataFrame): Original dataframe

    Returns:
        pd.DataFrame: Filtered dataframe
    """
    #modify = st.checkbox("Add filters", key = 123)

    #if not modify:
    #    return df

    df = df.copy()



    modification_container = st.container()

    with modification_container:
        to_filter_columns = st.multiselect("Filtrar tabela", df.columns)
        for column in to_filter_columns:
            left, right = st.columns((1, 20))
            # Treat columns with < 10 unique values as categorical
            if is_categorical_dtype(df[column]) or df[column].nunique() < 10:
                user_cat_input = right.multiselect(
                    f"Valores {column}",
                    df[column].unique(),
                    default=list(df[column].unique()),
                )
                df = df[df[column].isin(user_cat_input)]
            elif is_numeric_dtype(df[column]):
                _min = float(df[column].min())
                _max = float(df[column].max())
                step = (_max - _min) / 100
                user_num_input = right.slider(
                    f"Values {column}",
                    min_value=_min,
                    max_value=_max,
                    value=(_min, _max),
                    step=step,
                )
                df = df[df[column].between(*user_num_input)]
            elif is_datetime64_any_dtype(df[column]):
                user_date_input = right.date_input(
                    f"Valores {column}",
                    value=(
                        df[column].min(),
                        df[column].max(),
                    ),
                )
                if len(user_date_input) == 2:
                    user_date_input = tuple(map(pd.to_datetime, user_date_input))
                    start_date, end_date = user_date_input
                    df = df.loc[df[column].between(start_date, end_date)]
            else:
                user_text_input = right.text_input(
                    f"Substring ou regex {column}",
                )
                if user_text_input:
                    df = df[df[column].astype(str).str.contains(user_text_input)]

    return df




with coluna1:
    #player = st.selectbox("Escolha um player:", dfplayers['playername'])
    #date = st.selectbox("Escolha uma data:", dfplayers['date'])

    #styler_player = (dfplayers[dfplayers.playername == player][cols]
    #               .style.set_properties(**{'background': 'azure', 'border': '0.5px solid'})
    #               .hide(axis='index')
    #               .set_table_styles(dfstyle))
                   #.applymap(color_surplusvalue, subset=pd.IndexSlice[:, ['playername']]))	
    st.write(f'''
         ##### <div style="text-align: center">Campeonato CBLOL 2018 a 2022<span style="color:blue">
         ''', unsafe_allow_html=True)


    player_selected = filter_dataframe(dfplayers)
    st.dataframe(player_selected)
    
    
    #color_map = ['orange','pink']
    #plt.stackplot(player_selected.kills, player_selected.deaths, labels=['Kills', 'Deaths'])
    #sns.stackplot(data=player_selected,x="kills", hue="deaths",kind="kde", height=6,multiple="fill", clip=(0, None),palette="ch:rot=-.25,hue=1,light=.75",)
    #fig = px.area(p, x = 'kills', y='deaths', template = 'seaborn', color='blue', line_group='kills')
    #g1.plotly_chart(fig, use_container_width=True)
    #st.bar_chart(x=dfplayers['deaths'], y=dfplayers['kills'])
    #st.area_chart(dfstyle)

with coluna2:
    #player = st.selectbox("Escolha um player:", dfplayers['playername'])
    #date = st.selectbox("Escolha uma data:", dfplayers['date'])

    #styler_player = (dfplayers[dfplayers.playername == player][cols]
    #               .style.set_properties(**{'background': 'azure', 'border': '0.5px solid'})
    #               .hide(axis='index')
    #               .set_table_styles(dfstyle))
                   #.applymap(color_surplusvalue, subset=pd.IndexSlice[:, ['playername']])) 


    g1 = st.container()
    #player_selected = dfplayers[dfplayers.playername == player][cols]
    chart_data = pd.DataFrame(player_selected, columns=['kills', 'assists', 'deaths'])
    st.area_chart(chart_data)
    line_chart = pd.DataFrame(player_selected, columns=['totalgold'])
    st.line_chart(line_chart)
    bar_chart = pd.DataFrame(player_selected, columns=['total cs'])
    st.bar_chart(bar_chart)


    #color_map = ['orange','pink']
    #plt.stackplot(player_selected.kills, player_selected.deaths, labels=['Kills', 'Deaths'])
    #sns.stackplot(data=player_selected,x="kills", hue="deaths",kind="kde", height=6,multiple="fill", clip=(0, None),palette="ch:rot=-.25,hue=1,light=.75",)
    #fig = px.area(p, x = 'kills', y='deaths', template = 'seaborn', color='blue', line_group='kills')
    #g1.plotly_chart(fig, use_container_width=True)
    #st.bar_chart(x=dfplayers['deaths'], y=dfplayers['kills'])
    #st.area_chart(dfstyle)



cols2 = ['teamname', 'split', 'result', 'teamkills', 'teamdeaths', 'totalgold', 'towers', 'dragons', 'barons']

#with tab_team:
#	fig = px.scatter(
#	dfteams,
#	x="totalgold",
#	y="teamkills",
#	color="teamname",
#	hover_name="teamname",
#	log_x=True,
#	size_max=60,
#	)
#	st.plotly_chart(fig, theme="streamlit", use_container_width=True)
#
##	#styler_team = (dfteams[dfteams.teamname == team][cols2]
 #   #	.style.set_properties(**{'background': 'azure', 'border': '1.2px solid'})
 #   #	.hide(axis='index')
 #   #	.set_table_styles(dfstyle))
 #   	#.applymap(color_surplusvalue, subset=pd.IndexSlice[:, ['playername']]))	
#	#st.table(styler_team)
