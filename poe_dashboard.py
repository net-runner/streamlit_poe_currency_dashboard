import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import load_data as dataLoader
import translations

t = translations.translations

st.set_page_config(layout="wide", page_title=t["app_title"])
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
                st.header(t["tab_general_info"])
                st.subheader(t["general_summary_header"])
                st.write(f"{t['total_entries']}: {len(df_raw)}")
                st.write(f"{t['unique_leagues_count']}: {df_raw['league'].nunique()}")
                st.write(f"{t['data_date_range']}: {df_raw['date'].min().strftime('%Y-%m-%d')} {t['to']} {df_raw['date'].max().strftime('%Y-%m-%d')}")
                st.write(f"{t['unique_get_currencies_overall']}: {df_raw['get'].nunique()}")
                st.write(f"{t['unique_pay_currencies_overall']}: {df_raw['pay'].nunique()}")

                st.subheader(t["league_stats_header"])
                league_summary_list = []
                for league_name in league_info_df_global.index:
                    league_data = df_raw[df_raw['league'] == league_name]
                    info = league_info_df_global.loc[league_name]
                    league_summary_list.append({
                        t["col_league_name"]: league_name,
                        t["col_release_date"]: info['Release Date'].strftime('%Y-%m-%d'),
                        t["col_end_date"]: info['End Date'].strftime('%Y-%m-%d'),
                        t["col_total_weeks"]: f"{info['Total Weeks']:.2f}",
                        t["col_data_rows"]: len(league_data),
                        t["col_unique_get_items"]: league_data['get'].nunique() if not league_data.empty else 0,
                        t["col_unique_pay_items"]: league_data['pay'].nunique() if not league_data.empty else 0,
                    })
                if league_summary_list:
                    league_summary_df = pd.DataFrame(league_summary_list)
                    st.dataframe(league_summary_df, use_container_width=True)
                else:
                    st.info("Brak informacji o ligach do wyświetlenia.")


            with tabs[1]: # Value Trends
                st.header(t["value_trends_header"].format(selected_pay_currency=selected_pay_currency))
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

            with tabs[2]: # Periodic Averages
                st.header(t["periodic_averages_header"])
                st.markdown(t["periodic_averages_markdown"].format(selected_pay_currency=selected_pay_currency))

                if selected_get_currencies and selected_pay_currency:
                    for currency_get_avg in selected_get_currencies:
                        data_for_avg = df_sidebar_filtered[df_sidebar_filtered['get'] == currency_get_avg]
                        if not data_for_avg.empty:
                            st.subheader(t["averages_for_currency"].format(currency_get=currency_get_avg))
                            
                            avg_col1, avg_col2 = st.columns(2)
                            
                            with avg_col1:
                                weekly_avg_dfs = []
                                for league_name, group in data_for_avg.groupby('league'):
                                    group_copy = group.set_index('date').copy()
                                    weekly_mean = group_copy['value'].resample('W').mean().reset_index()
                                    weekly_mean['league'] = league_name
                                    weekly_avg_dfs.append(weekly_mean)
                                
                                if weekly_avg_dfs:
                                    weekly_avg_df = pd.concat(weekly_avg_dfs)
                                    fig_weekly = px.line(weekly_avg_df, x='date', y='value', color='league',
                                                         title=t["weekly_averages_title"].format(currency_get=currency_get_avg),
                                                         labels={'value': t["average_value_label"], 'date': t["week_label"], 'league': t["league_label"]})
                                    st.plotly_chart(fig_weekly, use_container_width=True)
                                else:
                                    st.caption(f"Brak danych tygodniowych dla {currency_get_avg}")

                            with avg_col2:
                                monthly_avg_dfs = []
                                for league_name, group in data_for_avg.groupby('league'):
                                    group_copy = group.set_index('date').copy()
                                    monthly_mean = group_copy['value'].resample('M').mean().reset_index() # 'M' is deprecated, use 'ME'
                                    monthly_mean['league'] = league_name
                                    monthly_avg_dfs.append(monthly_mean)

                                if monthly_avg_dfs:
                                    monthly_avg_df = pd.concat(monthly_avg_dfs)
                                    fig_monthly = px.line(monthly_avg_df, x='date', y='value', color='league',
                                                          title=t["monthly_averages_title"].format(currency_get=currency_get_avg),
                                                          labels={'value': t["average_value_label"], 'date': t["month_label"], 'league': t["league_label"]})
                                    st.plotly_chart(fig_monthly, use_container_width=True)
                                else:
                                    st.caption(f"Brak danych miesięcznych dla {currency_get_avg}")
                        else:
                            st.warning(t["warning_no_data_for_avg_calc"].format(currency_get=currency_get_avg))
                else:
                    st.info(t["info_select_currencies_for_averages"])


            with tabs[3]: # Avg Value Comparison
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
            
            with tabs[4]: # Value Distribution
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

            with tabs[5]: # Currency Profile (Radar)
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
            
            with tabs[6]: # Tainted Currencies
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

            with tabs[7]: # League Movers
                st.header(t["league_movers_header"])
                st.markdown(t["league_movers_markdown"].format(selected_pay_currency=selected_pay_currency))

                if selected_pay_currency:
                    league_start_end_changes = []
                    leagues_to_process_movers = [l for l in selected_leagues if l in league_info_df_global.index]

                    for league_name_iter in leagues_to_process_movers:
                        if league_name_iter not in league_info_df_global.index:
                            st.caption(t["skip_league_no_dates_caption"].format(league_name_iter=league_name_iter))
                            continue
                        
                        league_start_dt = league_info_df_global.loc[league_name_iter, 'Release Date']
                        league_end_dt = league_info_df_global.loc[league_name_iter, 'End Date']

                        league_data_for_movers = df_raw[
                            (df_raw['league'] == league_name_iter) &
                            (df_raw['pay'] == selected_pay_currency)
                        ]
                        if league_data_for_movers.empty: continue

                        start_day_data = league_data_for_movers[league_data_for_movers['date'] >= league_start_dt.normalize()].sort_values('date')
                        start_values = start_day_data.groupby('get', as_index=False).first()
                        start_values = start_values[['get', 'value']].rename(columns={'value': 'start_value'})
                        
                        end_day_data = league_data_for_movers[league_data_for_movers['date'] <= league_end_dt.normalize()].sort_values('date', ascending=False)
                        end_values = end_day_data.groupby('get', as_index=False).first()
                        end_values = end_values[['get', 'value']].rename(columns={'value': 'end_value'})
                        
                        if start_values.empty or end_values.empty: continue

                        change_df_league = pd.merge(start_values, end_values, on='get', how='inner')
                        if not change_df_league.empty:
                            change_df_league['league'] = league_name_iter
                            league_start_end_changes.append(change_df_league)
                    
                    if league_start_end_changes:
                        final_change_df = pd.concat(league_start_end_changes, ignore_index=True)
                        final_change_df['pct_change'] = np.where(
                            final_change_df['start_value'] != 0,
                            ((final_change_df['end_value'] - final_change_df['start_value']) / final_change_df['start_value']) * 100,
                            np.nan
                        )
                        final_change_df['pct_change'] = final_change_df['pct_change'].round(2)
                        final_change_df.replace([float('inf'), -float('inf')], float('nan'), inplace=True)
                        final_change_df.dropna(subset=['pct_change', 'start_value', 'end_value'], inplace=True)

                        if not final_change_df.empty:
                            df_display_movers = final_change_df[['league', 'get', 'start_value', 'end_value', 'pct_change']].rename(columns={
                                'league': t["league_label"],
                                'get': t["select_get_currencies"].split('(')[0].strip(), # "Wybierz Waluty 'Otrzymujesz'" -> "Wybierz Waluty 'Otrzymujesz'"
                                'start_value': t["col_start_value"],
                                'end_value': t["col_end_value"],
                                'pct_change': t["col_pct_change"]
                            }).sort_values(by=t["col_pct_change"], ascending=False)
                            st.dataframe(df_display_movers, height=300, use_container_width=True)
                            
                            n_movers_league = st.slider(t["num_top_bottom_movers_slider"], 1, min(20, len(final_change_df)), 10, key='n_movers_league_slider')
                            top_movers_league = final_change_df.nlargest(n_movers_league, 'pct_change')
                            bottom_movers_league = final_change_df.nsmallest(n_movers_league, 'pct_change')
                            plot_movers_league_df = pd.concat([top_movers_league, bottom_movers_league]).drop_duplicates()
                            if not plot_movers_league_df.empty:
                                plot_movers_league_df['label'] = plot_movers_league_df['get'] + " (" + plot_movers_league_df['league'] + ")"
                                plot_movers_league_df['type'] = np.where(plot_movers_league_df['pct_change'] >= 0, t["gainer_label"], t["loser_label"])
                                fig_movers_league = px.bar(plot_movers_league_df.sort_values('pct_change', ascending=True), 
                                                           x='pct_change', y='label', color='type', 
                                                           color_discrete_map={t["gainer_label"]: 'green', t["loser_label"]: 'red'}, 
                                                           orientation='h', 
                                                           title=t["top_bottom_movers_title"].format(n_movers_league=n_movers_league),
                                                           labels={'pct_change': t["col_pct_change"], 'label': t["currency_league_label"]})
                                fig_movers_league.update_layout(yaxis_title=None, height=max(400, 50 * len(plot_movers_league_df)))
                                st.plotly_chart(fig_movers_league, use_container_width=True)
                        else:
                            st.info(t["info_no_valid_pct_change_movers"])
                    else:
                        st.warning(t["warning_no_calculate_league_changes"])
                else:
                    st.info(t["info_select_pay_for_league_movers"])

            with tabs[8]: # Price Trajectories
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
            
            with tabs[9]: # Volatility Trends
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

            with tabs[10]: # Correlation Matrix
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

        else:
            st.error(t["error_date_range"])
    else:
        st.warning(t["warning_no_data_for_leagues"])
