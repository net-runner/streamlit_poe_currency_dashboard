import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import load_data as dataLoader
import translations

from tabs.general_info_tab import show_general_info_tab
from tabs.value_trends_tab import show_value_trends_tab
from tabs.periodic_averages_tab import show_periodic_averages_tab
from tabs.avg_value_tab import show_avg_value_tab
from tabs.value_distribution_tab import show_value_distribution_tab
from tabs.currency_profile_tab import show_currency_profile
from tabs.tainted_currencies_tab import show_tainted_currencies_tab
from tabs.league_movers_tab import show_league_movers_tab
from tabs.price_trajectories_tab import show_price_trajectories_tab
from tabs.volatility_trends_tab import show_volatility_trends_tab
from tabs.correlation_matrix_tab import show_correlation_matrix_tab


if 'language' not in st.session_state:
    st.session_state.language = 'en' 

if st.session_state.language == 'en':
    t = translations.translations_en
else:
    t = translations.translations_pl


st.set_page_config(layout="wide", page_title=t["app_title"])

st.sidebar.header(t["language_selection_header"])
selected_language_option = st.sidebar.selectbox(
    t["select_language"],
    options=['English', 'Polski'],
    index=0 if st.session_state.language == 'en' else 1
)

if selected_language_option == 'English':
    st.session_state.language = 'en'
elif selected_language_option == 'Polski':
    st.session_state.language = 'pl'


if st.session_state.language == 'en':
    t = translations.translations_en
else:
    t = translations.translations_pl

st.title(t["main_title"])
st.markdown(t["description"])

df_raw = dataLoader.load_data()
league_info_df_global = dataLoader.load_league_info()

df_raw = dataLoader.load_data()

if df_raw.empty:
    st.warning(t["warning_no_data_loaded"])
else:
    df_raw_merged_league_info = pd.merge(df_raw, league_info_df_global[['Release Date']], on='league', how='left')
    df_raw_merged_league_info['days_since_league_start'] = (df_raw_merged_league_info['date'] - df_raw_merged_league_info['Release Date']).dt.days

    st.sidebar.header(t["sidebar_header"])
    
    leagues_available = sorted(df_raw['league'].unique())
    selected_leagues = st.sidebar.multiselect(
        t["select_leagues"], 
        leagues_available, 
        default=leagues_available
    )
    
    df_filtered_by_league = df_raw_merged_league_info[df_raw_merged_league_info['league'].isin(selected_leagues)]
    
    if not df_filtered_by_league.empty:
        common_get_currencies = sorted(df_filtered_by_league['get'].unique())
        common_pay_currencies = sorted(df_filtered_by_league['pay'].unique())

        default_gets = [c for c in ["Divine Orb", "Chaos Orb", "Exalted Orb"] if c in common_get_currencies]
        if not default_gets:
            default_gets = common_get_currencies[:min(3, len(common_get_currencies))]

        selected_get_currencies = st.sidebar.multiselect(
            t["select_get_currencies"], 
            common_get_currencies, 
            default=default_gets
        )
        
        default_pay = "Chaos Orb" if "Chaos Orb" in common_pay_currencies else \
                      ("Divine Orb" if "Divine Orb" in common_pay_currencies else \
                       (common_pay_currencies[0] if common_pay_currencies else None))
        
        selected_pay_currency = st.sidebar.selectbox(
            t["select_pay_currency"], 
            common_pay_currencies, 
            index=common_pay_currencies.index(default_pay) if default_pay and default_pay in common_pay_currencies else 0
        )

        min_date_sidebar = df_filtered_by_league['date'].min()
        max_date_sidebar = df_filtered_by_league['date'].max()
        
        selected_date_range_sidebar = st.sidebar.date_input(
            t["select_date_range"],
            value=(min_date_sidebar.date() if pd.notna(min_date_sidebar) else None, 
                   max_date_sidebar.date() if pd.notna(max_date_sidebar) else None),
            min_value=min_date_sidebar.date() if pd.notna(min_date_sidebar) else None,
            max_value=max_date_sidebar.date() if pd.notna(max_date_sidebar) else None,
        )
        
        if selected_date_range_sidebar and len(selected_date_range_sidebar) == 2 and selected_date_range_sidebar[0] is not None and selected_date_range_sidebar[1] is not None:
            start_date_filter = pd.to_datetime(selected_date_range_sidebar[0])
            end_date_filter = pd.to_datetime(selected_date_range_sidebar[1])

            df_sidebar_filtered = df_filtered_by_league[
                (df_filtered_by_league['get'].isin(selected_get_currencies)) &
                (df_filtered_by_league['pay'] == selected_pay_currency) &
                (df_filtered_by_league['date'] >= start_date_filter) &
                (df_filtered_by_league['date'] <= end_date_filter)
            ]
            
            df_league_pay_date_filtered = df_filtered_by_league[
                (df_filtered_by_league['pay'] == selected_pay_currency) &
                (df_filtered_by_league['date'] >= start_date_filter) &
                (df_filtered_by_league['date'] <= end_date_filter)
            ]

            tab_titles_pl = [
                t["tab_general_info"],
                t["tab_value_trends"], 
                t["tab_periodic_averages"],
                t["tab_avg_value_comparison"], 
                t["tab_value_distribution"],
                t["tab_currency_profile_radar"],
                t["tab_tainted_currencies"],
                t["tab_league_movers"], 
                t["tab_price_trajectories"],
                t["tab_volatility_trends"],
                t["tab_correlation_matrix"]
            ]
            tabs = st.tabs(tab_titles_pl)

            with tabs[0]: # General Info
                show_general_info_tab(df_raw,league_info_df_global,t)

            with tabs[1]: # Value Trends
                show_value_trends_tab(selected_pay_currency,df_sidebar_filtered,t)

            with tabs[2]: # Periodic Averages
                show_periodic_averages_tab(selected_get_currencies,selected_pay_currency,t,df_sidebar_filtered)

            with tabs[3]: # Avg Value Comparison
                show_avg_value_tab(selected_pay_currency,common_get_currencies,df_league_pay_date_filtered,t,default_gets)
            
            with tabs[4]: # Value Distribution
                show_value_distribution_tab(df_sidebar_filtered,selected_get_currencies,t,selected_pay_currency)

            with tabs[5]: # Currency Profile (Radar)
                show_currency_profile(selected_pay_currency,t,common_get_currencies,selected_get_currencies,df_league_pay_date_filtered,selected_leagues)
            
            with tabs[6]: # Tainted Currencies
                show_tainted_currencies_tab(t,df_filtered_by_league,selected_pay_currency,start_date_filter,end_date_filter)
                
            with tabs[7]: # League Movers
                show_league_movers_tab(t,selected_pay_currency,selected_leagues,league_info_df_global,df_raw)

            with tabs[8]: # Price Trajectories
                show_price_trajectories_tab(t,df_sidebar_filtered,selected_pay_currency)
            
            with tabs[9]: # Volatility Trends
                show_volatility_trends_tab(t,selected_pay_currency,df_sidebar_filtered)

            with tabs[10]: # Correlation Matrix
                show_correlation_matrix_tab(t,selected_pay_currency,df_league_pay_date_filtered,selected_leagues,selected_get_currencies)

        else:
            st.error(t["error_date_range"])
    else:
        st.warning(t["warning_no_data_for_leagues"])

st.markdown(
    """
    <style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        text-align: center;
        padding: 10px;
        font-size: 14px;
    }
    .footer p {
        color: inherit;
    }
    .footer img {
        vertical-align: middle; /* Aligns the image with the text */
        margin-left: 5px; /* Adds a little space between text and image */
    }
    .footer a {
        margin-left: 5px;
    }
    </style>
    <div class="footer">
        <p>Made with <img src="https://i.redd.it/jelw95fth1691.png" alt="exalt" width="20" height="20"> by<a href="https://www.github.com/net-runner" target="_blank">@net-runner</a></p>
    </div>
    """,
    unsafe_allow_html=True
)