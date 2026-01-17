import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
from transformers import pipeline
import numpy as np
from datetime import datetime

# Page Configuration
st.set_page_config(page_title="JPMC SDT Analyzer", layout="wide", page_icon="📈")

# --- DATA EXTRACTION (Sourced from PDF Page 10) ---
# Hardcoding the data for stability and speed
data = {
    'Year': [2021, 2022, 2023, 2024],
    'Annual_Volume_B': [284, 196, 193, 226],
    'Cumulative_B': [284, 480, 673, 900]
}
df_progress = pd.DataFrame(data)
df_progress['Date'] = pd.to_datetime(df_progress['Year'].astype(str) + '-12-31')

# Category breakdown (Page 10)
categories_2024 = {
    'Category': ['Green', 'Development Finance', 'Community Development'],
    'Cumulative_Amount_B': [309, 434, 156] # Total 900 (rounded)
}
df_cat = pd.DataFrame(categories_2024)

# --- HEADER ---
st.title("JPMorgan Chase $2.5T SDT Progress Analyzer")
st.markdown("""
**Based on the 2024 Sustainability Report**
This tool analyzes progress toward the **$2.5 Trillion Sustainable Development Target (SDT)** by 2030. 
It utilizes time-series forecasting to predict target completion and models the impact of macroeconomic scenarios, such as the AI-driven energy boom.
""")

st.info(f"**Current Status (End of 2024):** ${df_progress['Cumulative_B'].iloc[-1]} Billion achieved of $2,500 Billion Target (36%)")

# --- TABS ---
tab1, tab2, tab3, tab4 = st.tabs(["📊 Current Progress", "🔮 2030 Forecast", "⚡ AI Scenario Simulator", "🧠 Semantic Analysis"])

# --- TAB 1: CURRENT PROGRESS ---
with tab1:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Annual Financing Volume")
        fig_bar = px.bar(df_progress, x='Year', y='Annual_Volume_B', 
                         title="Annual Financing & Facilitation ($ Billions)",
                         color_discrete_sequence=['#413a30']) # JPMC-ish brown/slate
        fig_bar.add_hline(y=250, line_dash="dot", annotation_text="Avg Required to Hit Target", annotation_position="top left")
        st.plotly_chart(fig_bar, use_container_width=True)
        st.caption("Data Source: 2024 Sustainability Report, Page 10")

    with col2:
        st.subheader("Cumulative Composition by Objective")
        fig_pie = px.pie(df_cat, values='Cumulative_Amount_B', names='Category', 
                         title="Cumulative Progress (2021-2024)",
                         color_discrete_sequence=px.colors.sequential.RdBu)
        st.plotly_chart(fig_pie, use_container_width=True)

# --- TAB 2: FORECASTING (PROPHET) ---
with tab2:
    st.subheader("Time Series Forecast: Path to $2.5T")
    st.markdown("Using **Facebook Prophet** (Logistic Growth) to project progress through 2030 based on 2021-2024 data.")

    # Prepare data for Prophet
    df_prophet = df_progress[['Date', 'Cumulative_B']].rename(columns={'Date': 'ds', 'Cumulative_B': 'y'})
    
    # Cap for logistic growth
    df_prophet['cap'] = 2500 # The Target
    df_prophet['floor'] = 0

    try:
        m = Prophet(growth='logistic')
        m.fit(df_prophet)
        
        # Make future dataframe
        future = m.make_future_dataframe(periods=7, freq='Y')
        future['cap'] = 2500
        future['floor'] = 0
        
        forecast = m.predict(future)
        
        # Visualization
        fig_forecast = go.Figure()
        
        # Historical Data
        fig_forecast.add_trace(go.Scatter(x=df_prophet['ds'], y=df_prophet['y'], mode='lines+markers', name='Actual', line=dict(color='black', width=3)))
        
        # Forecast
        fig_forecast.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Projected', line=dict(color='blue', dash='dash')))
        
        # Target Line
        fig_forecast.add_hline(y=2500, line_color="green", annotation_text="2030 Target ($2.5T)")
        
        fig_forecast.update_layout(title="Cumulative Progress Forecast vs. Target", yaxis_title="$ Billions", template="plotly_white")
        st.plotly_chart(fig_forecast, use_container_width=True)
        
        # Insight
        final_proj = forecast['yhat'].iloc[-1]
        st.write(f"**Model Prediction:** At current trend, the model projects reaching **${final_proj:.0f} Billion** by end of 2030.")
        if final_proj < 2500:
            st.warning("⚠️ The current trend suggests falling short of the target. Acceleration in Green or Development Finance is required.")
        else:
            st.success("✅ The current trend suggests meeting the target early.")
            
    except Exception as e:
        st.error(f"Error running Prophet (check C++ compiler requirements): {e}")

# --- TAB 3: SCENARIO SIMULATOR ---
with tab3:
    st.subheader("Scenario: The Impact of AI & Data Center Energy Demand")
    st.markdown("""
    *Context from Report (Page 3 & 23):* Rising energy demand from data centers driven by AI is expected to influence the ability to meet Carbon Intensity targets. 
    However, for the **SDT (Financing Target)**, this represents a commercial opportunity to finance grid upgrades and new power generation.
    """)
    
    col_sim1, col_sim2 = st.columns([1, 2])
    
    with col_sim1:
        st.markdown("### Adjust Assumptions")
        base_growth = st.slider("Base Annual Growth (%)", min_value=0, max_value=20, value=5)
        ai_boost = st.slider("AI Infrastructure Boost (%)", min_value=0, max_value=15, value=0, help="Additional financing volume driven by data center needs.")
        
    with col_sim2:
        # Simple simulation logic
        current_cum = 900
        years = range(2025, 2031)
        sim_data = []
        
        # Calculate remaining average needed
        remaining = 2500 - current_cum
        avg_needed = remaining / 6
        
        cumulative = current_cum
        # Base line for 2024
        last_annual = 226 
        
        for year in years:
            total_growth = (base_growth + ai_boost) / 100
            annual_vol = last_annual * (1 + total_growth)
            cumulative += annual_vol
            sim_data.append({'Year': year, 'Cumulative': cumulative, 'Annual': annual_vol})
            last_annual = annual_vol
            
        df_sim = pd.DataFrame(sim_data)
        
        fig_sim = go.Figure()
        fig_sim.add_trace(go.Bar(x=df_sim['Year'], y=df_sim['Cumulative'], name="Simulated Cumulative"))
        fig_sim.add_hline(y=2500, line_color="red", line_dash="dash", annotation_text="Goal")
        
        st.plotly_chart(fig_sim, use_container_width=True)
        
        if df_sim['Cumulative'].iloc[-1] >= 2500:
            st.success(f"Result: Target Met! (${df_sim['Cumulative'].iloc[-1]:,.0f}B)")
        else:
            st.error(f"Result: Target Missed. (${df_sim['Cumulative'].iloc[-1]:,.0f}B)")

# --- TAB 4: NLP ANALYSIS ---
with tab4:
    st.subheader("NLP Classification of Report Excerpts")
    st.markdown("Using a **Zero-Shot Classification** model to categorize key statements from the report regarding risks and strategy.")
    
    # Excerpts extracted from PDF text
    excerpts = {
        "AI Demand": "Rising energy demand from data centers driven by artificial intelligence is expected to influence our ability to meet our Electric Power target.",
        "Macro Economics": "Geopolitical tensions, diverging global economic growth and heightened interest rate volatility have slowed the pace of financing.",
        "Methodology": "We plan to continue to evaluate our SDT... and adjust it as we deem necessary, in light of latest climate science.",
        "Nuclear Energy": "In 2024, we hosted two private dialogues about financing nuclear energy with participants from technology and government."
    }
    
    selected_excerpt = st.selectbox("Select an Excerpt:", list(excerpts.keys()))
    text_to_analyze = excerpts[selected_excerpt]
    
    st.blockquote(text_to_analyze)
    
    if st.button("Analyze Text"):
        with st.spinner("Running Transformer Model..."):
            classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
            candidate_labels = ["Risk", "Commercial Opportunity", "Governance/Policy", "Strategic Pivot"]
            
            result = classifier(text_to_analyze, candidate_labels)
            
            # Display results
            df_nlp = pd.DataFrame({
                'Label': result['labels'],
                'Score': result['scores']
            })
            
            fig_nlp = px.bar(df_nlp, x='Score', y='Label', orientation='h', title="Classification Confidence", color='Score')
            st.plotly_chart(fig_nlp, use_container_width=True)

# --- FOOTER ---
st.markdown("---")
st.caption("Developed for Portfolio Demonstration. Data extracted from JPMorgan Chase 2024 Sustainability Report PDF.")
