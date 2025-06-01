import streamlit as st
import pandas as pd
import plotly.express as px

def show_value_distribution_tab(df_sidebar_filtered,selected_get_currencies,t,selected_pay_currency):
    st.header(t["value_distribution_header"])
    if not df_sidebar_filtered.empty:
        dist_get_choice_options = selected_get_currencies if selected_get_currencies else []
        dist_get_choice = st.selectbox(
            t["select_get_for_distribution"], 
            dist_get_choice_options, 
            key='dist_get_choice',
            index = 0 if dist_get_choice_options else -1
        )
        if dist_get_choice:
            dist_df = df_sidebar_filtered[df_sidebar_filtered['get'] == dist_get_choice]
            if not dist_df.empty:
                st.subheader(t["distribution_of_get_value_subheader"].format(dist_get_choice=dist_get_choice, selected_pay_currency=selected_pay_currency))
                col1, col2 = st.columns(2)
                with col1:
                    fig_box = px.box(dist_df, x='league', y='value', color='league', title=t["box_plot_by_league_title"],
                                    labels={'value': t["value_in_pay_currency"].format(selected_pay_currency=selected_pay_currency), 'league': t["league_label"]})
                    st.plotly_chart(fig_box, use_container_width=True)
                with col2:
                    fig_hist = px.histogram(dist_df, x='value', color='league', marginal="rug", barmode='overlay', opacity=0.7, title=t["histogram_by_league_title"],
                                            labels={'value': t["value_in_pay_currency"].format(selected_pay_currency=selected_pay_currency), 'league': t["league_label"]})
                    fig_hist.update_layout(bargap=0.1)
                    st.plotly_chart(fig_hist, use_container_width=True)
            else:
                st.warning(t["warning_no_distribution_data"].format(dist_get_choice=dist_get_choice))
        elif selected_get_currencies:
            st.info(t["info_select_get_for_distribution_plot"])
        else:
            st.info(t["info_no_get_selected_for_distribution"])
    else:
        st.info(t["info_ensure_get_pay_sidebar_distribution"])