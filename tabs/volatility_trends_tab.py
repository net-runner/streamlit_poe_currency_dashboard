import streamlit as st
import pandas as pd
import plotly.express as px

def show_volatility_trends_tab(t,selected_pay_currency,df_sidebar_filtered):
    st.header(t["volatility_trends_header"])
    st.markdown(t["volatility_trends_markdown"].format(selected_pay_currency=selected_pay_currency))
    if not df_sidebar_filtered.empty:
        volatility_dfs = []
        for (league, item), group in df_sidebar_filtered.groupby(['league', 'get']):
            group = group.sort_values('date')
            group['volatility'] = group['value'].rolling(window=7, min_periods=1).std()
            volatility_dfs.append(group)
        
        if volatility_dfs:
            volatility_plot_df = pd.concat(volatility_dfs)
            volatility_plot_df['legend_label'] = volatility_plot_df['get'] + " (" + volatility_plot_df['league'] + ")"
            fig_volatility = px.line(
                volatility_plot_df.sort_values(by=['league', 'get', 'date']),
                x='date',
                y='volatility',
                color='legend_label',
                labels={'volatility': t["rolling_std_dev_label"], 'date': t["date_label"], 'legend_label': t["currency_league_label"]},
                title=t["volatility_trends_vs_pay_currency_title"].format(selected_pay_currency=selected_pay_currency)
            )
            num_vol_lines = len(volatility_plot_df['legend_label'].unique())
            vol_height = max(400, 70 * num_vol_lines)
            fig_volatility.update_layout(height=min(vol_height, 1200), legend_title_text=t["currency_league_label"])
            st.plotly_chart(fig_volatility, use_container_width=True)
        else:
            st.warning(t["warning_no_calculate_volatility"])
    else:
        st.warning(t["warning_no_data_volatility_trends"])