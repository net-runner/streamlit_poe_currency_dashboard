import streamlit as st
import pandas as pd
import plotly.express as px

def show_periodic_averages_tab(selected_get_currencies,selected_pay_currency,t, df_sidebar_filtered):
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
                