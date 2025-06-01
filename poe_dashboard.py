import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import load_data as dataLoader

st.set_page_config(layout="wide", page_title="PoE Currency Dashboard")
st.title("PoE League Currency Dashboard")
st.markdown("""
Explore various leagues currency trends
Data is sourced from daily snapshots performed
by the poe.ninja website.
""")

df_raw = dataLoader.load_data()
league_info_df = dataLoader.load_league_info()

if df_raw.empty:
    st.warning("No data loaded. Please ensure your `load_data.py` functions correctly and CSV files are present and formatted (no header, specified column order: League, Date, Get, Pay, Value, Confidence).")
else:
    # --- Merge league start dates for trajectory analysis ---
    # Ensure 'league' column in df_raw matches league_info_df index (e.g., "Necropolis", not "Necropolis league")
    df_raw_merged_league_info = pd.merge(df_raw, league_info_df[['Release Date']], on='league', how='left')
    df_raw_merged_league_info['days_since_league_start'] = (df_raw_merged_league_info['date'] - df_raw_merged_league_info['Release Date']).dt.days


    st.sidebar.header("⚙️ Global Filters")
    
    leagues_available = sorted(df_raw['league'].unique())
    selected_leagues = st.sidebar.multiselect(
        "Select Leagues", 
        leagues_available, 
        default=leagues_available
    )
    
    df_filtered_by_league = df_raw_merged_league_info[df_raw_merged_league_info['league'].isin(selected_leagues)] # Use merged df
    
    if not df_filtered_by_league.empty:
        common_get_currencies = sorted(df_filtered_by_league['get'].unique())
        common_pay_currencies = sorted(df_filtered_by_league['pay'].unique())

        default_gets = [c for c in ["Divine Orb", "Chaos Orb", "Exalted Orb"] if c in common_get_currencies]
        if not default_gets:
            default_gets = common_get_currencies[:min(3, len(common_get_currencies))]

        selected_get_currencies = st.sidebar.multiselect(
            "Select 'Get' Currencies (Items you buy)", 
            common_get_currencies, 
            default=default_gets
        )
        
        default_pay = "Chaos Orb" if "Chaos Orb" in common_pay_currencies else \
                      ("Divine Orb" if "Divine Orb" in common_pay_currencies else \
                       (common_pay_currencies[0] if common_pay_currencies else None))
        
        selected_pay_currency = st.sidebar.selectbox(
            "Select 'Pay' Currency (Currency you use)", 
            common_pay_currencies, 
            index=common_pay_currencies.index(default_pay) if default_pay and default_pay in common_pay_currencies else 0
        )

        min_date_sidebar = df_filtered_by_league['date'].min()
        max_date_sidebar = df_filtered_by_league['date'].max()
        
        selected_date_range_sidebar = st.sidebar.date_input(
            "Select Date Range (for general charts)",
            value=(min_date_sidebar.date() if pd.notna(min_date_sidebar) else None, 
                   max_date_sidebar.date() if pd.notna(max_date_sidebar) else None),
            min_value=min_date_sidebar.date() if pd.notna(min_date_sidebar) else None,
            max_value=max_date_sidebar.date() if pd.notna(max_date_sidebar) else None,
        )
        
        if selected_date_range_sidebar and len(selected_date_range_sidebar) == 2 and selected_date_range_sidebar[0] is not None and selected_date_range_sidebar[1] is not None:
            start_date_filter = pd.to_datetime(selected_date_range_sidebar[0])
            end_date_filter = pd.to_datetime(selected_date_range_sidebar[1])

            # This df is for general charts using the sidebar date range
            df_sidebar_filtered = df_filtered_by_league[
                (df_filtered_by_league['get'].isin(selected_get_currencies)) &
                (df_filtered_by_league['pay'] == selected_pay_currency) &
                (df_filtered_by_league['date'] >= start_date_filter) &
                (df_filtered_by_league['date'] <= end_date_filter)
            ]
            
            # This df is for charts that might need broader 'get' selection but respect league, pay, and date range
            df_league_pay_date_filtered = df_filtered_by_league[
                (df_filtered_by_league['pay'] == selected_pay_currency) &
                (df_filtered_by_league['date'] >= start_date_filter) &
                (df_filtered_by_league['date'] <= end_date_filter)
            ]

            tab_titles = [
                "💹 Value Trends", 
                "📊 Avg Value Comparison", 
                "⚖️ Value Distribution",
                "🕸️ Currency Profile (Radar)",
                "🌀 Tainted Currencies",
                "🚀 League Movers", 
                "📈 Price Trajectories",
                "📉 Volatility Trends",
                "🔗 Correlation Matrix"
            ]
            tabs = st.tabs(tab_titles)

            with tabs[0]: # Value Trends
                st.header(f"💹 Value of Selected 'Get' Currencies (Paid with {selected_pay_currency})")
                if not df_sidebar_filtered.empty:
                    df_sidebar_filtered['legend_label'] = df_sidebar_filtered['get'] + " (" + df_sidebar_filtered['league'] + ")"
                    fig_trend = px.line(
                        df_sidebar_filtered.sort_values(by=['league', 'get', 'date']), 
                        x='date', 
                        y='value', 
                        color='legend_label',
                        labels={'value': f'Value in {selected_pay_currency}', 'date': 'Date', 'legend_label': 'Currency (League)'},
                        title=f"Value Trends: Selected 'Get' Currencies vs. {selected_pay_currency}"
                    )
                    num_lines = len(df_sidebar_filtered['legend_label'].unique())
                    dynamic_height = max(400, 70 * num_lines) 
                    fig_trend.update_layout(
                        height=min(dynamic_height, 1200), 
                        legend_title_text='Currency (League)'
                    )
                    st.plotly_chart(fig_trend, use_container_width=True)
                else:
                    st.warning(f"No data available for 'Value Trends' with current sidebar filters.")

            with tabs[1]: # Avg Value Comparison
                st.header("📊 Average Currency Value Comparison Across Leagues")
                avg_val_get_select_options = common_get_currencies if common_get_currencies else []
                # Ensure default_gets[0] is valid if default_gets exists
                default_avg_get_index = 0
                if default_gets and default_gets[0] in avg_val_get_select_options:
                    default_avg_get_index = avg_val_get_select_options.index(default_gets[0])
                
                avg_val_get_select = st.selectbox(
                    "Select 'Get' Currency for Average Comparison", 
                    avg_val_get_select_options, 
                    index=default_avg_get_index, 
                    key='avg_get_select'
                )

                if avg_val_get_select and selected_pay_currency:
                    # Use df_league_pay_date_filtered but filter for the specific 'get' currency
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
                            labels={'value': f'Average Value (in {selected_pay_currency})', 'league': 'League'},
                            title=f"Average Value of 1 {avg_val_get_select} (Paid with {selected_pay_currency}) Across Selected Leagues"
                        )
                        st.plotly_chart(fig_bar_avg, use_container_width=True)
                    else:
                        st.warning(f"No data to compare for {avg_val_get_select} to {selected_pay_currency} in selected leagues/date range.")
                else:
                    st.info("Select 'Get' and 'Pay' currencies to see average value comparison.")
            
            with tabs[2]: # Value Distribution
                st.header("⚖️ Currency Value Distribution")
                if not df_sidebar_filtered.empty:
                    dist_get_choice_options = selected_get_currencies if selected_get_currencies else []
                    dist_get_choice = st.selectbox(
                        "Select 'Get' Currency for Distribution Plot", 
                        dist_get_choice_options, 
                        key='dist_get_choice',
                        index = 0 if dist_get_choice_options else -1
                    )
                    if dist_get_choice:
                        dist_df = df_sidebar_filtered[df_sidebar_filtered['get'] == dist_get_choice]
                        if not dist_df.empty:
                            st.subheader(f"Distribution of {dist_get_choice} Value (Paid with {selected_pay_currency})")
                            col1, col2 = st.columns(2)
                            with col1:
                                fig_box = px.box(dist_df, x='league', y='value', color='league', title="Box Plot by League")
                                st.plotly_chart(fig_box, use_container_width=True)
                            with col2:
                                fig_hist = px.histogram(dist_df, x='value', color='league', marginal="rug", barmode='overlay', opacity=0.7, title="Histogram by League")
                                fig_hist.update_layout(bargap=0.1)
                                st.plotly_chart(fig_hist, use_container_width=True)
                        else:
                             st.warning(f"No distribution data for {dist_get_choice} with current filters.")
                    elif selected_get_currencies:
                         st.info("Select a 'Get' currency from the dropdown to see distribution plots.")
                    else:
                        st.info("No 'Get' currencies selected in the main filter for distribution plots.")
                else:
                    st.info("Ensure 'Get' and 'Pay' currencies are selected in sidebar for distribution plots.")

            with tabs[3]: # Currency Profile (Radar)
                st.header("🕸️ Currency Profile Radar Chart")
                st.markdown(f"Comparing profiles for items paid with **{selected_pay_currency}**.")

                radar_get_options = common_get_currencies if common_get_currencies else []
                default_radar_gets = [g for g in selected_get_currencies if g in radar_get_options][:min(5, len(selected_get_currencies))]

                radar_selected_get_currencies = st.multiselect(
                    "Select 'Get' Currencies for Radar Chart (max 5 recommended)", 
                    radar_get_options, 
                    default=default_radar_gets,
                    key='radar_get_multiselect'
                )

                if radar_selected_get_currencies and selected_pay_currency:
                    # Use df_league_pay_date_filtered, then filter by radar_selected_get_currencies
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
                            metrics_for_radar = ['avg_value', 'max_value', 'min_value', 'median_value']
                            
                            st.subheader("Average Profile Across Selected Leagues")
                            radar_metrics_avg_leagues = radar_metrics_calc.groupby('get')[metrics_for_radar].mean().reset_index()
                            
                            if not radar_metrics_avg_leagues.empty:
                                fig_radar_avg_leagues = go.Figure()
                                for _, row_data in radar_metrics_avg_leagues.iterrows():
                                    currency_name = row_data['get']
                                    values = [row_data[metric] for metric in metrics_for_radar]
                                    fig_radar_avg_leagues.add_trace(go.Scatterpolar(r=values, theta=metrics_for_radar, fill='toself', name=currency_name))
                                fig_radar_avg_leagues.update_layout(polar=dict(radialaxis=dict(visible=True, autorange=True)), showlegend=True, title=f"Avg. Profile (Paid with {selected_pay_currency})")
                                st.plotly_chart(fig_radar_avg_leagues, use_container_width=True)
                            else:
                                st.warning("Could not aggregate radar metrics averaged over leagues.")

                            if len(selected_leagues) == 1:
                                st.subheader(f"Currency Profiles in {selected_leagues[0]}")
                                radar_metrics_single_league = radar_metrics_calc[radar_metrics_calc['league'] == selected_leagues[0]]
                                if not radar_metrics_single_league.empty:
                                    fig_radar_sl = go.Figure()
                                    for _, row_data in radar_metrics_single_league.iterrows():
                                        currency_name = row_data['get']
                                        values = [row_data[metric] for metric in metrics_for_radar]
                                        fig_radar_sl.add_trace(go.Scatterpolar(r=values, theta=metrics_for_radar, fill='toself', name=currency_name))
                                    fig_radar_sl.update_layout(polar=dict(radialaxis=dict(visible=True, autorange=True)), showlegend=True, title=f"Profiles in {selected_leagues[0]} (Paid with {selected_pay_currency})")
                                    st.plotly_chart(fig_radar_sl, use_container_width=True)
                        else:
                            st.warning("No aggregated data for radar chart with current selections.")
                    else:
                        st.warning(f"No source data for radar chart with current filters.")
                else:
                    st.info("Select 'Get' currencies and a 'Pay' currency for the Radar Chart.")
            
            with tabs[4]: # Tainted Currencies
                st.header("🌀 Tainted Currency Trends")
                df_tainted_base = df_filtered_by_league[
                    df_filtered_by_league['get'].str.startswith("Tainted ", na=False) &
                    (df_filtered_by_league['pay'] == selected_pay_currency) & # Respect sidebar pay currency
                    (df_filtered_by_league['date'] >= start_date_filter) &    # Respect sidebar date range
                    (df_filtered_by_league['date'] <= end_date_filter)
                ]

                if not df_tainted_base.empty:
                    tainted_get_currencies = sorted(df_tainted_base['get'].unique())
                    selected_tainted_gets = st.multiselect(
                        "Select Tainted 'Get' Currencies to Display",
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
                            labels={'value': f'Value in {selected_pay_currency}', 'date': 'Date', 'legend_label': 'Tainted Currency (League)'},
                            title=f"Value Trends for Selected Tainted Currencies (Paid with {selected_pay_currency})"
                        )
                        num_tainted_lines = len(df_tainted_filtered['legend_label'].unique())
                        tainted_height = max(400, 70 * num_tainted_lines)
                        fig_tainted_trend.update_layout(height=min(tainted_height, 1000), legend_title_text='Tainted Currency (League)')
                        st.plotly_chart(fig_tainted_trend, use_container_width=True)
                    else:
                        st.info("No data for selected Tainted currencies within the filtered criteria or no Tainted currencies selected.")
                else:
                    st.info(f"No Tainted currency data found for the selected leagues, pay currency ({selected_pay_currency}), and date range.")

            with tabs[5]: # League Movers
                st.header("🚀 League Start-to-End Movers")
                st.markdown(f"Showing percentage change for items paid with **{selected_pay_currency}**, from each selected league's official start to its official end date.")

                if selected_pay_currency:
                    league_start_end_changes = []
                    leagues_to_process_movers = [l for l in selected_leagues if l in league_info_df.index]

                    for league_name_iter in leagues_to_process_movers:
                        if league_name_iter not in league_info_df.index:
                            st.caption(f"Skipping {league_name_iter}: No start/end date info found.")
                            continue
                        
                        league_start_dt = league_info_df.loc[league_name_iter, 'Release Date']
                        league_end_dt = league_info_df.loc[league_name_iter, 'End Date']

                        # Filter data for the specific league and pay currency from df_raw (not sidebar date filtered)
                        league_data_for_movers = df_raw[
                            (df_raw['league'] == league_name_iter) &
                            (df_raw['pay'] == selected_pay_currency)
                        ]
                        if league_data_for_movers.empty: continue

                        # Data closest to league start (on or after)
                        start_day_data = league_data_for_movers[league_data_for_movers['date'] >= league_start_dt.normalize()].sort_values('date')
                        start_values = start_day_data.groupby('get', as_index=False).first()
                        start_values = start_values[['get', 'value']].rename(columns={'value': 'start_value'})
                        
                        # Data closest to league end (on or before)
                        end_day_data = league_data_for_movers[league_data_for_movers['date'] <= league_end_dt.normalize()].sort_values('date', ascending=False)
                        end_values = end_day_data.groupby('get', as_index=False).first() # .first() because it's sorted descending
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
                        final_change_df.dropna(subset=['pct_change', 'start_value', 'end_value'], inplace=True) # Ensure values are not NaN

                        if not final_change_df.empty:
                            st.dataframe(final_change_df[['league', 'get', 'start_value', 'end_value', 'pct_change']].sort_values(by='pct_change', ascending=False), height=300, use_container_width=True)
                            n_movers_league = st.slider("Number of Top/Bottom League Movers to Display", 1, min(20, len(final_change_df)), 10, key='n_movers_league_slider')
                            top_movers_league = final_change_df.nlargest(n_movers_league, 'pct_change')
                            bottom_movers_league = final_change_df.nsmallest(n_movers_league, 'pct_change')
                            plot_movers_league_df = pd.concat([top_movers_league, bottom_movers_league]).drop_duplicates()
                            if not plot_movers_league_df.empty:
                                plot_movers_league_df['label'] = plot_movers_league_df['get'] + " (" + plot_movers_league_df['league'] + ")"
                                plot_movers_league_df['type'] = np.where(plot_movers_league_df['pct_change'] >= 0, 'Gainer', 'Loser')
                                fig_movers_league = px.bar(plot_movers_league_df.sort_values('pct_change', ascending=True), x='pct_change', y='label', color='type', color_discrete_map={'Gainer': 'green', 'Loser': 'red'}, orientation='h', title=f"Top/Bottom {n_movers_league} League Movers")
                                fig_movers_league.update_layout(yaxis_title=None, height=max(400, 50 * len(plot_movers_league_df)))
                                st.plotly_chart(fig_movers_league, use_container_width=True)
                        else:
                            st.info("No valid percentage changes could be calculated for selected league movers.")
                    else:
                        st.warning("Could not calculate league start-to-end changes. Ensure data exists around league start/end dates for selected leagues and pay currency.")
                else:
                    st.info("Select a 'Pay' currency to see league movers.")

            with tabs[6]: # Price Trajectories
                st.header("📈 Price Trajectories (Days Since League Start)")
                st.markdown(f"Comparing value of selected 'Get' currencies against **{selected_pay_currency}** based on days since league start.")
                # Uses df_sidebar_filtered which respects all sidebar filters including selected 'get' items
                if not df_sidebar_filtered.empty and 'days_since_league_start' in df_sidebar_filtered.columns:
                    trajectory_df = df_sidebar_filtered.dropna(subset=['days_since_league_start'])
                    trajectory_df['legend_label'] = trajectory_df['get'] + " (" + trajectory_df['league'] + ")"
                    
                    fig_trajectory = px.line(
                        trajectory_df.sort_values(by=['league', 'get', 'days_since_league_start']),
                        x='days_since_league_start',
                        y='value',
                        color='legend_label',
                        labels={'value': f'Value in {selected_pay_currency}', 'days_since_league_start': 'Days Since League Start', 'legend_label': 'Currency (League)'},
                        title=f"Price Trajectories vs. {selected_pay_currency}"
                    )
                    num_traj_lines = len(trajectory_df['legend_label'].unique())
                    traj_height = max(400, 70 * num_traj_lines)
                    fig_trajectory.update_layout(height=min(traj_height, 1200), legend_title_text='Currency (League)')
                    st.plotly_chart(fig_trajectory, use_container_width=True)
                else:
                    st.warning("No data for price trajectories. Ensure 'Release Date' is available for selected leagues and data exists for the filters.")
            
            with tabs[7]: # Volatility Trends
                st.header("📉 Volatility Trends (7-Day Rolling Std. Dev.)")
                st.markdown(f"Showing 7-day rolling standard deviation for selected 'Get' currencies (Paid with **{selected_pay_currency}**).")
                # Uses df_sidebar_filtered
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
                            labels={'volatility': '7-Day Rolling Std. Dev.', 'date': 'Date', 'legend_label': 'Currency (League)'},
                            title=f"Volatility Trends vs. {selected_pay_currency}"
                        )
                        num_vol_lines = len(volatility_plot_df['legend_label'].unique())
                        vol_height = max(400, 70 * num_vol_lines)
                        fig_volatility.update_layout(height=min(vol_height, 1200), legend_title_text='Currency (League)')
                        st.plotly_chart(fig_volatility, use_container_width=True)
                    else:
                        st.warning("Could not calculate volatility for the selected items.")
                else:
                    st.warning("No data for volatility trends with current filters.")

            with tabs[8]: # Correlation Matrix
                st.header("🔗 Correlation Matrix")
                st.markdown(f"Shows the correlation of daily values between selected 'Get' currencies, within a single chosen league, paid with **{selected_pay_currency}**.")
                
                corr_league_options = selected_leagues
                selected_corr_league = st.selectbox("Select League for Correlation Matrix", corr_league_options, index=0 if corr_league_options else -1, key="corr_league_select")

                if selected_corr_league and selected_pay_currency:
                    # Use df_league_pay_date_filtered, then filter by selected_corr_league and selected_get_currencies from main sidebar
                    corr_data_source = df_league_pay_date_filtered[
                        (df_league_pay_date_filtered['league'] == selected_corr_league) &
                        (df_league_pay_date_filtered['get'].isin(selected_get_currencies)) # Use main selected 'get' items
                    ]

                    if len(corr_data_source['get'].unique()) > 1:
                        try:
                            pivot_df = corr_data_source.pivot_table(index='date', columns='get', values='value')
                            # Interpolate to handle missing values for a more stable correlation matrix
                            pivot_df_interpolated = pivot_df.interpolate(method='linear', limit_direction='both', axis=0)
                            corr_matrix = pivot_df_interpolated.corr()

                            fig_corr = px.imshow(
                                corr_matrix, 
                                text_auto=True, 
                                aspect="auto", 
                                color_continuous_scale='RdBu_r', # Red-Blue diverging scale
                                zmin=-1, zmax=1, # Fix range for correlation
                                title=f"Correlation Matrix for {selected_corr_league} (Paid with {selected_pay_currency})"
                            )
                            st.plotly_chart(fig_corr, use_container_width=True)
                        except Exception as e:
                            st.error(f"Could not generate correlation matrix: {e}. Ensure sufficient overlapping data for selected items.")
                    else:
                        st.info("Please select at least two 'Get' currencies in the main sidebar filter to compute a correlation matrix.")
                else:
                    st.info("Select a league and ensure 'Pay' currency is chosen for the correlation matrix.")

        else:
            st.error("Please select a valid date range in the sidebar. Both start and end dates must be set.")
    else:
        st.warning("No data available for the selected leagues. Please select at least one league with data in the sidebar.")

st.sidebar.markdown("---")
st.sidebar.info("Streamlit dashboard to observe currency trends in markets from Crucible to Necropolis PoE challenge leagues")
