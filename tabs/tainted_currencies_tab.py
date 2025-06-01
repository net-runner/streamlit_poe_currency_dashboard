import streamlit as st
import pandas as pd
import plotly.express as px

def show_tainted_currencies_tab(t,df_filtered_by_league,selected_pay_currency,start_date_filter,end_date_filter):
    st.header(t["tainted_currencies_header"])
    df_tainted_base = df_filtered_by_league[
        df_filtered_by_league['get'].str.startswith("Tainted ", na=False) &
        (df_filtered_by_league['pay'] == selected_pay_currency) &
        (df_filtered_by_league['date'] >= start_date_filter) &
        (df_filtered_by_league['date'] <= end_date_filter)
    ]

    if not df_tainted_base.empty:
        tainted_get_currencies = sorted(df_tainted_base['get'].unique())
        selected_tainted_gets = st.multiselect(
            t["select_tainted_gets_display"],
            tainted_get_currencies,
            default=tainted_get_currencies[:min(5, len(tainted_get_currencies))],
            key="tainted_select"
        )
        df_tainted_filtered = df_tainted_base[df_tainted_base['get'].isin(selected_tainted_gets)]

        if not df_tainted_filtered.empty:
            df_tainted_filtered['legend_label'] = df_tainted_filtered['get'] + " (" + df_tainted_filtered['league'] + ")"
            fig_tainted_trend = px.line(
                df_tainted_filtered.sort_values(by=['league', 'get', 'date']),
                x='date',
                y='value',
                color='legend_label',
                labels={'value': t["value_in_pay_currency"].format(selected_pay_currency=selected_pay_currency), 'date': t["date_label"], 'legend_label': t["tainted_currency_league_label"]},
                title=t["tainted_trends_title"].format(selected_pay_currency=selected_pay_currency)
            )
            num_tainted_lines = len(df_tainted_filtered['legend_label'].unique())
            tainted_height = max(400, 70 * num_tainted_lines)
            fig_tainted_trend.update_layout(height=min(tainted_height, 1000), legend_title_text=t["tainted_currency_league_label"])
            st.plotly_chart(fig_tainted_trend, use_container_width=True)
        else:
            st.info(t["info_no_data_selected_tainted"])
    else:
        st.info(t["info_no_tainted_data_found"].format(selected_pay_currency=selected_pay_currency))

