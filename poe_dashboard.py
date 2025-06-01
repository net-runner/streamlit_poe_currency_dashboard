import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import load_data as dataLoader # Import the external data loader

st.set_page_config(layout="wide", page_title="PoE Currency Dashboard")
st.title("📈 Path of Exile Currency Dashboard 📉")
st.markdown("""
Welcome to the Path of Exile Currency Dashboard! 
Use the sidebar filters to explore currency trends across different leagues.
Data is sourced from daily snapshots.
""")

df_raw = dataLoader.load_data()

if df_raw.empty:
    st.warning("No data loaded. Please ensure your `load_data.py` functions correctly and CSV files are present and formatted (no header, specified column order: League, Date, Get, Pay, Value, Confidence).")
else:
    confidence_mapping = {"Low": 1, "Medium": 2, "High": 3}
    df_raw['confidence_score'] = df_raw['confidence'].map(confidence_mapping)
    # Rows where 'confidence' was not in Low/Medium/High will have NaN in 'confidence_score'
    # These NaNs are typically handled (ignored) by pandas/plotly aggregations.
    # If strict removal is needed: df_raw.dropna(subset=['confidence_score'], inplace=True)

    st.sidebar.header("⚙️ Global Filters")
    
    leagues_available = sorted(df_raw['league'].unique())
    selected_leagues = st.sidebar.multiselect(
        "Select Leagues", 
        leagues_available, 
        default=leagues_available
    )
    
    df_filtered_by_league = df_raw[df_raw['league'].isin(selected_leagues)]
    
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

        min_date_val = df_filtered_by_league['date'].min()
        max_date_val = df_filtered_by_league['date'].max()
        
        selected_date_range = st.sidebar.date_input(
            "Select Date Range",
            value=(min_date_val.date() if pd.notna(min_date_val) else None, 
                   max_date_val.date() if pd.notna(max_date_val) else None),
            min_value=min_date_val.date() if pd.notna(min_date_val) else None,
            max_value=max_date_val.date() if pd.notna(max_date_val) else None,
        )
        
        if selected_date_range and len(selected_date_range) == 2 and selected_date_range[0] is not None and selected_date_range[1] is not None:
            start_date = pd.to_datetime(selected_date_range[0])
            end_date = pd.to_datetime(selected_date_range[1])

            df_final_filters = df_filtered_by_league[
                (df_filtered_by_league['get'].isin(selected_get_currencies)) &
                (df_filtered_by_league['pay'] == selected_pay_currency) &
                (df_filtered_by_league['date'] >= start_date) &
                (df_filtered_by_league['date'] <= end_date)
            ]
            
            df_league_date_pay_filtered = df_filtered_by_league[
                (df_filtered_by_league['pay'] == selected_pay_currency) &
                (df_filtered_by_league['date'] >= start_date) &
                (df_filtered_by_league['date'] <= end_date)
            ]

            tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
                "💹 Value Trends", 
                "📊 Avg Value Comparison", 
                "✨ Value vs. Confidence", 
                "🕸️ Currency Profile (Radar)", 
                "⚖️ Value Distribution", 
                "🚀 Movers"
            ])

            with tab1:
                st.header(f"💹 Value of Selected 'Get' Currencies (Paid with {selected_pay_currency})")
                if not df_final_filters.empty:
                    df_final_filters['legend_label'] = df_final_filters['get'] + " (" + df_final_filters['league'] + ")"
                    fig_trend = px.line(
                        df_final_filters.sort_values(by=['league', 'get', 'date']), 
                        x='date', 
                        y='value', 
                        color='legend_label',
                        labels={'value': f'Value in {selected_pay_currency}', 'date': 'Date', 'legend_label': 'Currency (League)'},
                        title=f"Value Trends: Selected 'Get' Currencies vs. {selected_pay_currency}"
                    )
                    num_lines = len(df_final_filters['legend_label'].unique())
                    dynamic_height = max(400, 70 * num_lines) 
                    fig_trend.update_layout(
                        height=min(dynamic_height, 1200), 
                        legend_title_text='Currency (League)'
                    )
                    st.plotly_chart(fig_trend, use_container_width=True)
                else:
                    st.warning(f"No data available for the selected 'Get' currencies, 'Pay' currency ({selected_pay_currency}), leagues, and date range.")

            with tab2:
                st.header("📊 Average Currency Value Comparison Across Leagues")
                avg_val_get_select_options = common_get_currencies if common_get_currencies else []
                avg_val_get_select = st.selectbox(
                    "Select 'Get' Currency for Average Comparison", 
                    avg_val_get_select_options, 
                    index=avg_val_get_select_options.index(default_gets[0]) if default_gets and default_gets[0] in avg_val_get_select_options else 0, 
                    key='avg_get_select'
                )

                if avg_val_get_select and selected_pay_currency:
                    avg_df_source = df_filtered_by_league[
                        (df_filtered_by_league['get'] == avg_val_get_select) & 
                        (df_filtered_by_league['pay'] == selected_pay_currency) &
                        (df_filtered_by_league['date'] >= start_date) &
                        (df_filtered_by_league['date'] <= end_date)
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
            
            with tab3:
                st.header("✨ Value vs. Confidence")
                if not df_final_filters.empty:
                    scatter_get_choice_options = selected_get_currencies if selected_get_currencies else []
                    scatter_get_choice = st.selectbox(
                        "Focus on 'Get' Currency for Scatter Plot:", 
                        scatter_get_choice_options, 
                        index = 0 if scatter_get_choice_options else -1,
                        key='scatter_get_choice'
                    )
                    if scatter_get_choice:
                        scatter_df = df_final_filters[df_final_filters['get'] == scatter_get_choice]
                        if not scatter_df.empty:
                            fig_scatter = px.scatter(
                                scatter_df,
                                x='value',
                                y='confidence_score',
                                color='league',
                                size='value', 
                                hover_data=['date', 'get', 'pay', 'confidence'],
                                labels={'value': f'Value (in {selected_pay_currency})', 
                                        'confidence_score': 'Confidence Score (1:Low, 2:Medium, 3:High)',
                                        'confidence': 'Confidence Level'},
                                title=f"Value vs. Confidence Score for {scatter_get_choice} (Paid with {selected_pay_currency})"
                            )
                            st.plotly_chart(fig_scatter, use_container_width=True)
                        else:
                             st.warning(f"No data for {scatter_get_choice} with current filters for scatter plot.")
                    elif selected_get_currencies : 
                         st.info("Select a 'Get' currency from the dropdown to view the scatter plot.")
                    else: 
                        st.info("No 'Get' currencies selected in the main filter to choose from for the scatter plot.")
                else:
                    st.info("Select 'Get' currencies, 'Pay' currency, leagues and date range from the sidebar to see Value vs. Confidence plot.")

            with tab4:
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
                    radar_data_source = df_filtered_by_league[
                        (df_filtered_by_league['pay'] == selected_pay_currency) &
                        (df_filtered_by_league['get'].isin(radar_selected_get_currencies)) &
                        (df_filtered_by_league['date'] >= start_date) &
                        (df_filtered_by_league['date'] <= end_date)
                    ]

                    if not radar_data_source.empty:
                        radar_metrics_calc = radar_data_source.groupby(['league', 'get']).agg(
                            avg_value=('value', 'mean'),
                            max_value=('value', 'max'),
                            min_value=('value', 'min'),
                            avg_confidence=('confidence_score', 'mean'), # Use confidence_score for calculation
                            median_value=('value','median')
                        ).reset_index()
                        
                        radar_metrics_calc.fillna(0, inplace=True) 

                        if not radar_metrics_calc.empty:
                            metrics_for_radar = ['avg_value', 'max_value', 'min_value', 'median_value', 'avg_confidence']
                            
                            st.subheader("Average Profile Across Selected Leagues")
                            radar_metrics_avg_leagues = radar_metrics_calc.groupby('get')[metrics_for_radar].mean().reset_index()
                            
                            if not radar_metrics_avg_leagues.empty:
                                fig_radar_avg_leagues = go.Figure()
                                for _, row in radar_metrics_avg_leagues.iterrows():
                                    currency_name = row['get']
                                    values = [row[metric] for metric in metrics_for_radar]
                                    fig_radar_avg_leagues.add_trace(go.Scatterpolar(
                                        r=values,
                                        theta=metrics_for_radar,
                                        fill='toself',
                                        name=currency_name
                                    ))
                                fig_radar_avg_leagues.update_layout(
                                    polar=dict(radialaxis=dict(visible=True, autorange=True)),
                                    showlegend=True,
                                    title=f"Avg. Profile for 'Get' Currencies (Paid with {selected_pay_currency})"
                                )
                                st.plotly_chart(fig_radar_avg_leagues, use_container_width=True)
                            else:
                                st.warning("Could not aggregate radar metrics averaged over leagues.")

                            if len(selected_leagues) == 1:
                                st.subheader(f"Currency Profiles in {selected_leagues[0]}")
                                radar_metrics_single_league = radar_metrics_calc[radar_metrics_calc['league'] == selected_leagues[0]]
                                if not radar_metrics_single_league.empty:
                                    fig_radar_sl = go.Figure()
                                    for _, row in radar_metrics_single_league.iterrows():
                                        currency_name = row['get']
                                        values = [row[metric] for metric in metrics_for_radar]
                                        fig_radar_sl.add_trace(go.Scatterpolar(
                                            r=values,
                                            theta=metrics_for_radar,
                                            fill='toself',
                                            name=currency_name
                                        ))
                                    fig_radar_sl.update_layout(
                                        polar=dict(radialaxis=dict(visible=True, autorange=True)),
                                        showlegend=True,
                                        title=f"Currency Profiles in {selected_leagues[0]} (Paid with {selected_pay_currency})"
                                    )
                                    st.plotly_chart(fig_radar_sl, use_container_width=True)
                                else:
                                    st.warning(f"No radar data for currencies in league {selected_leagues[0]}.")
                        else:
                            st.warning("No aggregated data for radar chart with current selections.")
                    else:
                        st.warning(f"No source data for radar chart with Pay currency {selected_pay_currency}, selected 'Get' currencies, leagues, and date range.")
                elif not selected_pay_currency:
                    st.info("Select a 'Pay' currency from the sidebar for the Radar Chart.")
                else: 
                    st.info("Select 'Get' currencies for the Radar Chart.")


            with tab5:
                st.header("⚖️ Currency Value Distribution")
                if not df_final_filters.empty:
                    dist_get_choice_options = selected_get_currencies if selected_get_currencies else []
                    dist_get_choice = st.selectbox(
                        "Select 'Get' Currency for Distribution Plot", 
                        dist_get_choice_options, 
                        key='dist_get_choice',
                        index = 0 if dist_get_choice_options else -1
                    )
                    if dist_get_choice:
                        dist_df = df_final_filters[df_final_filters['get'] == dist_get_choice]
                        if not dist_df.empty:
                            st.subheader(f"Distribution of {dist_get_choice} Value (Paid with {selected_pay_currency})")
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                fig_box = px.box(
                                    dist_df, x='league', y='value', color='league',
                                    title="Box Plot by League"
                                )
                                st.plotly_chart(fig_box, use_container_width=True)
                            with col2:
                                fig_hist = px.histogram(
                                    dist_df, x='value', color='league', marginal="rug", 
                                    barmode='overlay',
                                    opacity=0.7,
                                    title="Histogram by League"
                                )
                                fig_hist.update_layout(bargap=0.1)
                                st.plotly_chart(fig_hist, use_container_width=True)
                        else:
                             st.warning(f"No distribution data for {dist_get_choice} with current filters.")
                    elif selected_get_currencies:
                         st.info("Select a 'Get' currency from the dropdown to see distribution plots.")
                    else:
                        st.info("No 'Get' currencies selected in the main filter to choose from for distribution plots.")
                else:
                    st.info("Ensure 'Get' and 'Pay' currencies are selected in the sidebar, along with leagues and date range, for distribution plots.")

            with tab6:
                st.header("🚀 Currency Value Movers")
                st.markdown(f"Showing percentage change for items paid with **{selected_pay_currency}** between **{start_date.date()}** and **{end_date.date()}**.")
                
                movers_source_df = df_league_date_pay_filtered.copy()

                if not movers_source_df.empty and 'date' in movers_source_df.columns and not movers_source_df['date'].isnull().all():
                    movers_source_df = movers_source_df.sort_values(by=['league', 'get', 'date'])
                    
                    first_values = movers_source_df.groupby(['league', 'get'])['value'].first().reset_index()
                    first_values.rename(columns={'value': 'start_value'}, inplace=True)
                    
                    last_values = movers_source_df.groupby(['league', 'get'])['value'].last().reset_index()
                    last_values.rename(columns={'value': 'end_value'}, inplace=True)

                    change_df = pd.merge(first_values, last_values, on=['league', 'get'], how='inner')
                    
                    change_df['pct_change'] = np.where(
                        change_df['start_value'] != 0, 
                        ((change_df['end_value'] - change_df['start_value']) / change_df['start_value']) * 100,
                        np.nan 
                    )
                    change_df['pct_change'] = change_df['pct_change'].round(2)
                    
                    change_df.replace([float('inf'), -float('inf')], float('nan'), inplace=True)
                    change_df.dropna(subset=['pct_change'], inplace=True)

                    if not change_df.empty:
                        st.subheader(f"Value Change from {start_date.date()} to {end_date.date()} (Paid with {selected_pay_currency})")
                        
                        show_all_movers = st.checkbox("Show movers for ALL 'Get' currencies (ignores main 'Get' filter for this table/chart)", value=True)
                        if not show_all_movers:
                            change_df_display = change_df[change_df['get'].isin(selected_get_currencies)]
                        else:
                            change_df_display = change_df
                        
                        st.dataframe(
                            change_df_display[['league', 'get', 'start_value', 'end_value', 'pct_change']].sort_values(by='pct_change', ascending=False), 
                            use_container_width=True,
                            height=300
                        )

                        n_movers = st.slider("Number of Top/Bottom Movers to Display in Chart", 1, min(20, len(change_df_display) if not change_df_display.empty else 1), 10, key='n_movers_slider')
                        
                        if not change_df_display.empty:
                            top_movers = change_df_display.nlargest(n_movers, 'pct_change')
                            bottom_movers = change_df_display.nsmallest(n_movers, 'pct_change')
                            
                            plot_movers_df = pd.concat([top_movers, bottom_movers]).drop_duplicates()
                            if not plot_movers_df.empty:
                                plot_movers_df['label'] = plot_movers_df['get'] + " (" + plot_movers_df['league'] + ")"
                                plot_movers_df['type'] = np.where(plot_movers_df['pct_change'] >= 0, 'Gainer', 'Loser')


                                fig_movers = px.bar(
                                    plot_movers_df.sort_values('pct_change', ascending=True), 
                                    x='pct_change', 
                                    y='label', 
                                    color='type',
                                    color_discrete_map={'Gainer': 'green', 'Loser': 'red'},
                                    orientation='h',
                                    labels={'pct_change': '% Change', 'label': 'Currency (League)'},
                                    title=f"Top/Bottom {n_movers} Value Movers"
                                )
                                fig_movers.update_layout(yaxis_title=None, height=max(400, 50 * len(plot_movers_df)))
                                st.plotly_chart(fig_movers, use_container_width=True)
                            else:
                                st.info("Not enough distinct data to display movers chart based on selection.")
                        else:
                             st.info("No data for movers chart after filtering. Try 'Show all movers' or adjust main filters.")
                    else:
                        st.warning("Could not calculate value changes. Ensure the selected date range has at least two data points for selected currencies, or that start values are not zero.")
                else:
                    st.warning("Not enough data or date information missing for calculating movers with the current 'Pay' currency and date range.")
        else:
            st.error("Please select a valid date range in the sidebar. Both start and end dates must be set.")
    else:
        st.warning("No data available for the selected leagues. Please select at least one league with data in the sidebar.")

st.sidebar.markdown("---")
st.sidebar.info("Path of Exile Streamlit currency dashboard from Crucible to Necropolis leagues. \nEconomy data dumps from poe.ninja/data")