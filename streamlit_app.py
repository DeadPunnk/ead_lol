#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
#import matplotlib.pyplot as plt
#import seaborn as sns
#import altair as alt
from pandas.api.types import (
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
)

st.set_page_config(page_title="LoL Analytics", page_icon="🎮", initial_sidebar_state="expanded", layout='wide')

st.title("E-sports Analytics Dashboard")

#@st.cache
def load_and_prep_players():
    dfplayers = pd.read_csv('all_players.csv')
    #dfplayers['KDA'] = str(round(((dfplayers['kills'] + dfplayers['assists'])/dfplayers['deaths']), 2))
    #dfplayers = dfplayers.set_index(dfplayers['date'])

    
    return dfplayers

def load_and_prep_teams():
    dfteams = pd.read_csv('teamdatast.csv')
    #dfplayers['KDA'] = str(round(((dfplayers['kills'] + dfplayers['assists'])/dfplayers['deaths']), 2))
    #dfteams = dfteams.set_index(dfplayers['date'])
    return dfteams

dfplayers = load_and_prep_players()
dfteams = load_and_prep_teams() 

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


tab1, tab2, tab3 = st.tabs(['Player', 'Team', 'Player VS Player'])




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




with tab1:

    g1, g2, g3 = st.columns((1,1,1))

    st.write(f'''
         ##### <div style="text-align: center">Campeonato CBLOL 2018 a 2022<span style="color:blue">
         ''', unsafe_allow_html=True)




    player_selected = filter_dataframe(dfplayers)
    st.dataframe(player_selected)
    

    chart_data = pd.DataFrame(player_selected, columns=['kills', 'assists', 'deaths'])

    g1.write(f'''
         ##### <div style="text-align: center">Kills, Assists, mortes por partida<span style="color:blue">
         ''', unsafe_allow_html=True)
    g1.area_chart(chart_data)


    g2.write(f'''
         ##### <div style="text-align: center">Total de ouro por partida<span style="color:blue">
         ''', unsafe_allow_html=True)
    line_chart = pd.DataFrame(player_selected, columns=['totalgold'])

    g2.line_chart(line_chart)

    bar_chart = pd.DataFrame(player_selected, columns=['total cs'])

    g3.write(f'''
         ##### <div style="text-align: center">Total de farme por partida<span style="color:blue">
         ''', unsafe_allow_html=True)
    g3.bar_chart(bar_chart)



cols2 = ['teamname', 'split', 'result', 'teamkills', 'teamdeaths', 'totalgold', 'towers', 'dragons', 'barons']

with tab2:

    c1, c2 = st.columns((1,1))

    #pie = dfteams.query('teamname in ["INTZ", "KaBuM! e-Sports", "Flamengo Esports", "paiN Gaming", "RED Canids"]')

    #pie_chart_team = pd.DataFrame(data = [[816, 750, 702, 618, 576]], columns = ['KaBuM! e-Sports', 'Flamengo Esports', 'paiN Gaming', 'INTZ', 'RED Canids'], index = ['Vitorias']).transpose()
    #pd.DataFrame(data = [[816, 750, 702, 618, 576]], columns = ['KaBuM! e-Sports', 'Flamengo Esports', 'paiN Gaming', 'INTZ', 'RED Canids'])

    bar_chart = dfteams[['teamname', 'result']].query('teamname in ["INTZ", "KaBuM! e-Sports", "Flamengo Esports", "paiN Gaming", "RED Canids"]')

    c1.write(f'''
         ##### <div style="text-align: center">Total de vitorias por time<span style="color:blue">
         ''', unsafe_allow_html=True)
    #st.bar_chart(bar_chart_team, x = 'teamname', y = 'result')

    fig_bar = px.bar(bar_chart, x = 'teamname', y = 'result', color = 'teamname')

    #fig_1.update_traces(textposition = 'inside', textinfo = 'percent+label')

    c1.plotly_chart(fig_bar, use_container_width = True)

    c2.write(f'''
         ##### <div style="text-align: center">Times que venceram de acordo com kills e gold<span style="color:blue">
         ''', unsafe_allow_html=True)

    line_chart = dfteams[['teamname', 'teamkills', 'totalgold', 'result']].query('teamname in ["INTZ", "KaBuM! e-Sports", "Flamengo Esports", "paiN Gaming", "RED Canids"]')

    fig_line = px.scatter(line_chart, x = 'totalgold', y = 'teamkills', color = 'result', hover_data = ['teamname'])

    c2.plotly_chart(fig_line, use_container_width = True)

    st.dataframe(dfteams)
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


    #styler_team = (dfteams[dfteams.teamname == team][cols2]
   #	.style.set_properties(**{'background': 'azure', 'border': '1.2px solid'})
   #	.hide(axis='index')
   #	.set_table_styles(dfstyle))
   	#.applymap(color_surplusvalue, subset=pd.IndexSlice[:, ['playername']]))	
	#st.table(styler_team)


with tab3:

    pcol1, pcol2 = st.columns((1,1))

    df_1 = dfplayers['playername']
    player1 = pcol1.selectbox("Escolha um player:", df_1, key = 1)
    
    styler_player1 = (dfplayers[dfplayers.playername == player1][cols]
                   .style.set_properties(**{'background': 'azure', 'border': '0.5px solid'})
                   .hide(axis='index')
                   .set_table_styles(dfstyle))

    pcol1.dataframe(styler_player1)

    df_2 = dfplayers['playername']
    player2 = pcol2.selectbox("Escolha um player:", df_2, key = 2)
    
    styler_player2 = (dfplayers[dfplayers.playername == player2][cols]
                   .style.set_properties(**{'background': 'azure', 'border': '0.5px solid'})
                   .hide(axis='index')
                   .set_table_styles(dfstyle))

    pcol2.dataframe(styler_player2)

    player_selected1 = dfplayers[dfplayers.playername == player1][cols]

    chart_data = pd.DataFrame(player_selected1, columns=['kills', 'assists', 'deaths'])

    pcol1.write(f'''
         ##### <div style="text-align: center">Kills, Assists, mortes por partida<span style="color:blue">
         ''', unsafe_allow_html=True)
    pcol1.area_chart(chart_data)


    pcol1.write(f'''
         ##### <div style="text-align: center">Total de ouro por partida<span style="color:blue">
         ''', unsafe_allow_html=True)
    line_chart = pd.DataFrame(player_selected1, columns=['totalgold'])

    pcol1.line_chart(line_chart)

    bar_chart = pd.DataFrame(player_selected1, columns=['total cs'])

    pcol1.write(f'''
         ##### <div style="text-align: center">Total de farme por partida<span style="color:blue">
         ''', unsafe_allow_html=True)
    pcol1.bar_chart(bar_chart)


    ####### player 2 #############


    player_selected2 = dfplayers[dfplayers.playername == player2][cols]

    chart_data = pd.DataFrame(player_selected2, columns=['kills', 'assists', 'deaths'])

    pcol2.write(f'''
         ##### <div style="text-align: center">Kills, Assists, mortes por partida<span style="color:blue">
         ''', unsafe_allow_html=True)
    pcol2.area_chart(chart_data)


    pcol2.write(f'''
         ##### <div style="text-align: center">Total de ouro por partida<span style="color:blue">
         ''', unsafe_allow_html=True)
    line_chart = pd.DataFrame(player_selected2, columns=['totalgold'])

    pcol2.line_chart(line_chart)

    bar_chart = pd.DataFrame(player_selected2, columns=['total cs'])

    pcol2.write(f'''
         ##### <div style="text-align: center">Total de farme por partida<span style="color:blue">
         ''', unsafe_allow_html=True)
    pcol2.bar_chart(bar_chart)

