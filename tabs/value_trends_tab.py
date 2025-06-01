import streamlit as st
import plotly.express as px

def show_value_trends_tab(selected_pay_currency,df_sidebar_filtered,t):
  
    if not df_sidebar_filtered.empty:
        df_sidebar_filtered['legend_label'] = df_sidebar_filtered['get'] + " (" + df_sidebar_filtered['league'] + ")"
        fig_trend = px.line(
            df_sidebar_filtered.sort_values(by=['league', 'get', 'date']), 
            x='date', 
             y='value', 
            color='legend_label',
            labels={'value': t["value_in_pay_currency"].format(selected_pay_currency=selected_pay_currency), 'date': t["date_label"], 'legend_label': t["currency_league_label"]},
            title=t["value_trends_title"].format(selected_pay_currency=selected_pay_currency)
        )
        num_lines = len(df_sidebar_filtered['legend_label'].unique())
        dynamic_height = max(400, 70 * num_lines) 
        fig_trend.update_layout(
            height=min(dynamic_height, 1200), 
            legend_title_text=t["currency_league_label"]
        )
        st.plotly_chart(fig_trend, use_container_width=True)
    else:
        st.warning(t["warning_no_data_value_trends"])