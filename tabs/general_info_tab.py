import streamlit as st
import pandas as pd
import plotly.express as px


def show_general_info_tab(df_raw, league_info_df_global, t):

    st.header(t["tab_general_info"])
    st.write(
        f"{t['data_date_range']}: {df_raw['date'].min().strftime('%Y-%m-%d')} {t['to']} {df_raw['date'].max().strftime('%Y-%m-%d')}"
    )
    league_summary_list = []
    
    
    col_genMetric1, col_genMetric2 = st.columns(2)
    
    with col_genMetric1:
        
        st.metric(
            label=t['total_entries'],
            value=len(df_raw),
        )
        
        st.metric(
            label=t['unique_get_currencies_overall'],
            value=df_raw['get'].nunique(),
        )

    with col_genMetric2:
        
        st.metric(
            label=t['unique_leagues_count'],
            value=df_raw['league'].nunique(),
        )
        
        st.metric(
            label=t['unique_pay_currencies_overall'],
            value=df_raw['pay'].nunique(),
        )
    
    
    # Track the newest league's info to use for delta calculations
    newest_league_name = None
    newest_league_end_date = pd.Timestamp.min # Initialize with the earliest possible timestamp
    newest_league_total_weeks = 0.0
    newest_league_unique_get_items = 0
    
    for league_name in league_info_df_global.index:

        league_data = df_raw[df_raw["league"] == league_name]
        info = league_info_df_global.loc[league_name]
        
        # Ensure 'Release Date' and 'End Date' are datetime objects for comparison
        release_date = pd.to_datetime(info['Release Date'])
        end_date = pd.to_datetime(info['End Date'])

        unique_get_items = league_data["get"].nunique() if not league_data.empty else 0
        unique_pay_items = league_data["pay"].nunique() if not league_data.empty else 0
        
        try:
            total_weeks_numeric = float(info['Total Weeks'])
        except ValueError:
            # Handle cases where conversion might fail (e.g., if it's 'N/A' or empty)
            st.warning(f"Could not convert 'Total Weeks' for league '{league_name}' to a number. Setting to 0. Please check your data.")
            total_weeks_numeric = 0.0

        league_summary_list.append(
            {
                t["col_league_name"]: league_name,
                t["col_release_date"]: info["Release Date"].strftime("%Y-%m-%d"),
                t["col_end_date"]: info["End Date"].strftime("%Y-%m-%d"),
                t["col_total_weeks"]: f"{info['Total Weeks']:.2f}",
                t["col_data_rows"]: len(league_data),
                t["col_unique_get_items"]: unique_get_items,
                t["col_unique_pay_items"]: unique_pay_items,
            }
        )
        
        # Identify the newest league based on 'End Date'
        if end_date > newest_league_end_date:
            newest_league_end_date = end_date
            newest_league_name = league_name
            newest_league_total_weeks = info['Total Weeks']
            newest_league_unique_get_items = unique_get_items

    if league_summary_list:
        
        league_summary_df = pd.DataFrame(league_summary_list)
        league_summary_df[t["col_total_weeks"]] = pd.to_numeric(
            league_summary_df[t["col_total_weeks"]], errors='coerce'
        ).fillna(0)
        
        if newest_league_name:
            st.write(t['last_league'].format(newest_league_name=newest_league_name,newest_league_end_date=newest_league_end_date.strftime('%Y-%m-%d')))

            # Calculate overall averages
            avg_league_length_overall = league_summary_df[t["col_total_weeks"]].mean()
            avg_unique_get_items_overall = league_summary_df[t["col_unique_get_items"]].mean()

            # Calculate deltas against the newest league's values
            delta_league_length = newest_league_total_weeks - avg_league_length_overall 
            delta_unique_get_items = newest_league_unique_get_items- avg_unique_get_items_overall 

            col_metric1, col_metric2 = st.columns(2)

            with col_metric1:
                st.metric(
                    label=t['league_length'],
                    value=f"{newest_league_total_weeks:.2f}",
                    delta=f"{delta_league_length:.2f}"
                )

            with col_metric2:
                st.metric(
                    label=t['unique_currencies'],
                    value=f"{newest_league_unique_get_items:.2f}",
                    delta=f"{delta_unique_get_items:.2f}"
                )
        else:
            st.info("Could not determine the newest league for delta calculations.")

        st.markdown("---") # Another separator

        # --- Plots ---
        st.subheader("Visualizations")

        # Plot 1: Unique Items per League
        st.write("#### Unique Get/Pay Items per League")
        fig_items = px.bar(
            league_summary_df,
            x=t["col_league_name"],
            y=[t["col_unique_get_items"], t["col_unique_pay_items"]],
            title="Unique Items (Get & Pay) by League",
            labels={
                t["col_league_name"]: t["col_league_name"], # Use translated labels for plots too
                "value": t["col_num_unique_items"],
                "variable": t["col_item_type"]
            },
            barmode='group'
        )
        st.plotly_chart(fig_items, use_container_width=True)

        # Plot 2: League Length Distribution
        st.write("#### League Length Distribution")
        fig_length = px.histogram(
            league_summary_df,
            x=t["col_total_weeks"],
            nbins=10,
            title="Distribution of League Lengths",
            labels={t["col_total_weeks"]: t["col_total_weeks"]}, # Use translated label
            marginal="box"
        )
        st.plotly_chart(fig_length, use_container_width=True)
        
        # League list
        st.dataframe(league_summary_df, use_container_width=True)

    else:
        st.info(t["no_league_info"])
