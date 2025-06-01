import streamlit as st
import pandas as pd
import plotly.express as px

def show_price_trajectories_tab(t,df_sidebar_filtered,selected_pay_currency):
    st.header(t["price_trajectories_header"])
    st.markdown(t["price_trajectories_markdown"].format(selected_pay_currency=selected_pay_currency))
    if not df_sidebar_filtered.empty and 'days_since_league_start' in df_sidebar_filtered.columns:
        trajectory_df = df_sidebar_filtered.dropna(subset=['days_since_league_start'])
        trajectory_df['legend_label'] = trajectory_df['get'] + " (" + trajectory_df['league'] + ")"
        
        fig_trajectory = px.line(
            trajectory_df.sort_values(by=['league', 'get', 'days_since_league_start']),
            x='days_since_league_start',
            y='value',
            color='legend_label',
            labels={'value': t["value_in_pay_currency"].format(selected_pay_currency=selected_pay_currency), 'days_since_league_start': t["days_since_league_start_label"], 'legend_label': t["currency_league_label"]},
            title=t["price_trajectories_vs_pay_currency_title"].format(selected_pay_currency=selected_pay_currency)
        )
        num_traj_lines = len(trajectory_df['legend_label'].unique())
        traj_height = max(400, 70 * num_traj_lines)
        fig_trajectory.update_layout(height=min(traj_height, 1200), legend_title_text=t["currency_league_label"])
        st.plotly_chart(fig_trajectory, use_container_width=True)
    else:
        st.warning(t["warning_no_data_price_trajectories"])
            