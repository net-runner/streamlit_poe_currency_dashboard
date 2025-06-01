import streamlit as st
import pandas as pd
import plotly.express as px

def show_correlation_matrix_tab(t,selected_pay_currency,df_league_pay_date_filtered,selected_leagues,selected_get_currencies):
    st.header(t["correlation_matrix_header"])
    st.markdown(t["correlation_matrix_markdown"].format(selected_pay_currency=selected_pay_currency))
    
    corr_league_options = selected_leagues
    selected_corr_league = st.selectbox(t["select_league_for_correlation"], corr_league_options, index=0 if corr_league_options else -1, key="corr_league_select")

    if selected_corr_league and selected_pay_currency:
        corr_data_source = df_league_pay_date_filtered[
            (df_league_pay_date_filtered['league'] == selected_corr_league) &
            (df_league_pay_date_filtered['get'].isin(selected_get_currencies))
        ]

        if len(corr_data_source['get'].unique()) > 1:
            try:
                pivot_df = corr_data_source.pivot_table(index='date', columns='get', values='value')
                pivot_df_interpolated = pivot_df.interpolate(method='linear', limit_direction='both', axis=0)
                corr_matrix = pivot_df_interpolated.corr()

                fig_corr = px.imshow(
                    corr_matrix, 
                    text_auto=True, 
                    aspect="auto", 
                    color_continuous_scale='RdBu_r', 
                    zmin=-1, zmax=1,
                    title=t["correlation_matrix_for_league_title"].format(selected_corr_league=selected_corr_league, selected_pay_currency=selected_pay_currency)
                )
                st.plotly_chart(fig_corr, use_container_width=True)
            except Exception as e:
                st.error(t["error_could_not_generate_correlation"].format(e=e))
        else:
            st.info(t["info_select_two_get_for_correlation"])
    else:
        st.info(t["info_select_league_pay_for_correlation"])