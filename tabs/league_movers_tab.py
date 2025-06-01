import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

def show_league_movers_tab(t,selected_pay_currency,selected_leagues,league_info_df_global,df_raw):
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
