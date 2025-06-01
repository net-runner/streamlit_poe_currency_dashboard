import streamlit as st
import pandas as pd
import plotly.express as px

def show_avg_value_tab(selected_pay_currency,common_get_currencies,df_league_pay_date_filtered,t,default_gets):
    st.header(t["avg_value_comparison_header"])
    avg_val_get_select_options = common_get_currencies if common_get_currencies else []
    default_avg_get_index = 0
    if default_gets and default_gets[0] in avg_val_get_select_options:
        default_avg_get_index = avg_val_get_select_options.index(default_gets[0])
                
    avg_val_get_select = st.selectbox(
        t["select_get_for_avg_comparison"], 
        avg_val_get_select_options, 
        index=default_avg_get_index, 
        key='avg_get_select'
    )

    if avg_val_get_select and selected_pay_currency:
        avg_df_source = df_league_pay_date_filtered[
            (df_league_pay_date_filtered['get'] == avg_val_get_select)
        ]
        if not avg_df_source.empty:
            avg_value_by_league = avg_df_source.groupby('league')['value'].mean().reset_index().sort_values(by='value', ascending=False)
            fig_bar_avg = px.bar(
                avg_value_by_league,
                x='league',
                y='value',
                color='league',
                labels={'value': t["avg_value_in_pay_currency"].format(selected_pay_currency=selected_pay_currency), 'league': t["league_label"]},
                title=t["avg_value_of_one_get_title"].format(avg_val_get_select=avg_val_get_select, selected_pay_currency=selected_pay_currency)
            )
            st.plotly_chart(fig_bar_avg, use_container_width=True)
        else:
            st.warning(t["warning_no_data_to_compare"].format(avg_val_get_select=avg_val_get_select, selected_pay_currency=selected_pay_currency))
    else:
        st.info(t["info_select_get_pay_for_avg_comp"])