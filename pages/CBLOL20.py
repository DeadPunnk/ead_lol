#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
#import matplotlib.pyplot as plt
#import seaborn as sns
#import altair as alt


st.set_page_config(page_title="LoL Analytics", page_icon="🎮", initial_sidebar_state="expanded")



#@st.cache
def load_and_prep_players():
    dfplayers = pd.read_csv('pages/playersst_20.csv')
    #dfplayers['KDA'] = str(round(((dfplayers['kills'] + dfplayers['assists'])/dfplayers['deaths']), 2))
    #dfplayers = dfplayers.set_index(dfplayers['date'])

    
    return dfplayers

def load_and_prep_teams():
    dfteams = pd.read_csv('pages/teamsst.csv')
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


tab_player, tab_team = st.tabs(["Player", "Time"])


cols = ['teamname', 'position', 'kills', 'deaths', 'assists', 'KDA', 'totalgold', 'total cs']

with tab_player:
    player = st.selectbox("Escolha um player:", dfplayers['playername'])
    styler_player = (dfplayers[dfplayers.playername == player][cols]
                   .style.set_properties(**{'background': 'azure', 'border': '1.2px solid'})
                   .hide(axis='index')
                   .set_table_styles(dfstyle))
                   #.applymap(color_surplusvalue, subset=pd.IndexSlice[:, ['playername']]))	
    st.write(f'''
         ##### <div style="text-align: center">Campeonato CBLOL 2020<span style="color:blue">
         ''', unsafe_allow_html=True)


    
    g1 = st.container()
    player_selected = dfplayers[dfplayers.playername == player][cols]
    chart_data = pd.DataFrame(player_selected, columns=['kills', 'assists', 'deaths'])
    st.area_chart(chart_data)
    line_chart = pd.DataFrame(player_selected, columns=['totalgold'])
    st.line_chart(line_chart)
    bar_chart = pd.DataFrame(player_selected, columns=['total cs'])
    st.bar_chart(bar_chart)

    st.table(styler_player)

    #color_map = ['orange','pink']
    #plt.stackplot(player_selected.kills, player_selected.deaths, labels=['Kills', 'Deaths'])
    #sns.stackplot(data=player_selected,x="kills", hue="deaths",kind="kde", height=6,multiple="fill", clip=(0, None),palette="ch:rot=-.25,hue=1,light=.75",)
    #fig = px.area(p, x = 'kills', y='deaths', template = 'seaborn', color='blue', line_group='kills')
    #g1.plotly_chart(fig, use_container_width=True)
    #st.bar_chart(x=dfplayers['deaths'], y=dfplayers['kills'])
    #st.area_chart(dfstyle)



cols2 = ['teamname', 'split', 'result', 'teamkills', 'teamdeaths', 'totalgold', 'towers', 'dragons', 'barons']

with tab_team:
	fig = px.scatter(
	dfteams,
	x="totalgold",
	y="teamkills",
	color="teamname",
	hover_name="teamname",
	log_x=True,
	size_max=60,
	)
	st.plotly_chart(fig, theme="streamlit", use_container_width=True)

	#team = st.selectbox("Choose a Team (or click below and start typing):", dfteams['teamname'])
	#styler_team = (dfteams[dfteams.teamname == team][cols2]
    #	.style.set_properties(**{'background': 'azure', 'border': '1.2px solid'})
    #	.hide(axis='index')
    #	.set_table_styles(dfstyle))
    	#.applymap(color_surplusvalue, subset=pd.IndexSlice[:, ['playername']]))	
	#st.table(styler_team)
