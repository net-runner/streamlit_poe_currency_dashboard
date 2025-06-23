import streamlit as st
import pandas as pd
import plotly.express as px

def show_price_trajectories_tab(t, df_sidebar_filtered, selected_pay_currency):
    st.header(t["price_trajectories_header"])
    st.markdown(t["price_trajectories_markdown"].format(selected_pay_currency=selected_pay_currency))

    if not df_sidebar_filtered.empty and 'days_since_league_start' in df_sidebar_filtered.columns:
        # --- NEW: Filter out data where 'days_since_league_start' is -1 ---
        # Create a working copy to avoid modifying the original df_sidebar_filtered if it's used elsewhere
        processed_df = df_sidebar_filtered[df_sidebar_filtered['days_since_league_start'] != -1].copy()

        if processed_df.empty:
            st.warning(t["warning_no_data_after_negative_days_filter"])
            return # Exit if no data remains after this filter

        # Get min and max days since league start for the slider from the processed_df
        min_days = int(processed_df['days_since_league_start'].min())
        max_days = int(processed_df['days_since_league_start'].max())

        # Add a slider for days since league start
        selected_days_range = st.slider(
            t["days_since_league_start_slider_label"],
            min_value=min_days,
            max_value=max_days,
            value=(min_days, max_days) # Default to full range
        )

        # Filter the DataFrame based on the slider selection
        trajectory_df = processed_df[
            (processed_df['days_since_league_start'] >= selected_days_range[0]) &
            (processed_df['days_since_league_start'] <= selected_days_range[1])
        ].dropna(subset=['days_since_league_start'])

        if not trajectory_df.empty:
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
            st.warning(t["warning_no_data_for_selected_days_range"])
    else:
        st.warning(t["warning_no_data_price_trajectories"])