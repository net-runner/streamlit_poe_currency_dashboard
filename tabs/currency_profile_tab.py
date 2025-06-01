import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def show_currency_profile(selected_pay_currency,t,common_get_currencies,selected_get_currencies,df_league_pay_date_filtered,selected_leagues):
    st.header(t["currency_profile_radar_header"])
    st.markdown(t["radar_markdown"].format(selected_pay_currency=selected_pay_currency))

    radar_get_options = common_get_currencies if common_get_currencies else []
    default_radar_gets = [g for g in selected_get_currencies if g in radar_get_options][:min(5, len(selected_get_currencies))]

    radar_selected_get_currencies = st.multiselect(
        t["select_get_for_radar"], 
        radar_get_options, 
        default=default_radar_gets,
        key='radar_get_multiselect'
    )

    if radar_selected_get_currencies and selected_pay_currency:
        radar_data_source = df_league_pay_date_filtered[
            df_league_pay_date_filtered['get'].isin(radar_selected_get_currencies)
        ]

        if not radar_data_source.empty:
            radar_metrics_calc = radar_data_source.groupby(['league', 'get']).agg(
                avg_value=('value', 'mean'),
                max_value=('value', 'max'),
                min_value=('value', 'min'),
                median_value=('value','median')
            ).reset_index()
            radar_metrics_calc.fillna(0, inplace=True) 

            if not radar_metrics_calc.empty:
                metrics_for_radar_keys = ['avg_value', 'max_value', 'min_value', 'median_value']
                radar_labels_pl = [t["radar_label_avg_value"], t["radar_label_max_value"], t["radar_label_min_value"], t["radar_label_median_value"]]
                            
                st.subheader(t["avg_profile_across_leagues_subheader"])
                radar_metrics_avg_leagues = radar_metrics_calc.groupby('get')[metrics_for_radar_keys].mean().reset_index()
                            
                if not radar_metrics_avg_leagues.empty:
                    fig_radar_avg_leagues = go.Figure()
                    for _, row_data in radar_metrics_avg_leagues.iterrows():
                        currency_name = row_data['get']
                        values = [row_data[metric] for metric in metrics_for_radar_keys]
                        fig_radar_avg_leagues.add_trace(go.Scatterpolar(r=values, theta=radar_labels_pl, fill='toself', name=currency_name))
                    fig_radar_avg_leagues.update_layout(polar=dict(radialaxis=dict(visible=True, autorange=True)), showlegend=True, title=t["radar_avg_profile_title"].format(selected_pay_currency=selected_pay_currency))
                    st.plotly_chart(fig_radar_avg_leagues, use_container_width=True)
                else:
                    st.warning(t["warning_cant_aggregate_radar_avg_leagues"])

                if len(selected_leagues) == 1:
                    st.subheader(t["currency_profiles_in_league_subheader"].format(selected_league=selected_leagues[0]))
                    radar_metrics_single_league = radar_metrics_calc[radar_metrics_calc['league'] == selected_leagues[0]]
                    if not radar_metrics_single_league.empty:
                        fig_radar_sl = go.Figure()
                        for _, row_data in radar_metrics_single_league.iterrows():
                            currency_name = row_data['get']
                            values = [row_data[metric] for metric in metrics_for_radar_keys]
                            fig_radar_sl.add_trace(go.Scatterpolar(r=values, theta=radar_labels_pl, fill='toself', name=currency_name))
                        fig_radar_sl.update_layout(polar=dict(radialaxis=dict(visible=True, autorange=True)), showlegend=True, title=t["radar_profiles_in_league_title"].format(selected_league=selected_leagues[0], selected_pay_currency=selected_pay_currency))
                        st.plotly_chart(fig_radar_sl, use_container_width=True)
                else:
                    st.warning(t["warning_no_aggregated_data_radar"])
            else:
                st.warning(t["warning_no_source_data_radar"])
        else:
            st.info(t["info_select_get_pay_for_radar"])