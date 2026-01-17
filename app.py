import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from transformers import pipeline
import numpy as np
from datetime import datetime
from sklearn.linear_model import LinearRegression

# --- CONFIGURATION & SETUP ---
st.set_page_config(page_title="JPMC Strategic Analytics", layout="wide", page_icon="🏦")

# --- CUSTOM CSS (Advanced Dark Theme) ---
st.markdown("""
<style>
    /* Global Gradient */
    .stApp {
        background: radial-gradient(circle at top left, #1b2838, #0e1117);
        color: #e6e6e6;
    }
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #0e1117;
        border-right: 1px solid #2d333b;
    }
    /* Metric Cards */
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, #21262d 0%, #0d1117 100%);
        border: 1px solid #30363d;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        border-radius: 8px;
        padding: 10px;
    }
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #161b22;
        border-radius: 4px 4px 0px 0px;
        color: #8b949e;
    }
    .stTabs [aria-selected="true"] {
        background-color: #238636; /* JPMC Green-ish */
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# --- HELPER FUNCTIONS ---
def run_linear_projection(df, years_ahead):
    """Simple Linear Regression for comparison"""
    X = df['Year'].values.reshape(-1, 1)
    y = df['Cumulative_B'].values
    model = LinearRegression()
    model.fit(X, y)
    
    last_year = df['Year'].max()
    future_years = np.arange(last_year + 1, last_year + years_ahead + 1).reshape(-1, 1)
    predictions = model.predict(future_years)
    
    return future_years.flatten(), predictions

# --- DATA LOADING ---
progress_data = {
    'Year': [2021, 2022, 2023, 2024],
    'Annual_Volume_B': [284, 196, 193, 226],
    'Cumulative_B': [284, 480, 673, 900]
}
df_progress = pd.DataFrame(progress_data)
df_progress['Date'] = pd.to_datetime(df_progress['Year'].astype(str) + '-12-31')

# Detailed Category Data
df_cat = pd.DataFrame({
    'Category': ['Green', 'Development Finance', 'Community Development'],
    'Value': [309, 434, 156],
    'Parent': ['Achieved', 'Achieved', 'Achieved'],
    'Color': ['#2ea043', '#d29922', '#58a6ff'] # Green, Yellow, Blue
})

# --- SIDEBAR CONTROLS ---
with st.sidebar:
    st.title("🎛️ Risk Controls")
    
    st.subheader("Scenario Parameters")
    base_growth = st.slider("Base Market CAGR (%)", 0.0, 15.0, 5.0, 0.5)
    ai_impact = st.slider("AI Infrastructure Alpha (%)", 0.0, 10.0, 2.0, 0.5, help="Additional growth driven by data center financing.")
    
    st.subheader("Model Settings")
    show_confidence = st.checkbox("Show Confidence Intervals", value=True)
    
    st.divider()
    
    # CSV Download Button
    csv = df_progress.to_csv(index=False).encode('utf-8')
    st.download_button(
        "📥 Download Historical Data",
        csv,
        "jpmc_sdt_data.csv",
        "text/csv",
        key='download-csv'
    )

# --- MAIN HEADER ---
col_logo, col_title = st.columns([1, 5])
with col_title:
    st.title("JPMorgan Chase: Strategic Capital Analyzer")
    st.markdown("### Tracking the $2.5 Trillion Sustainable Development Target (SDT)")

# Metrics Row
current_val = df_progress['Cumulative_B'].iloc[-1]
target_val = 2500
gap = target_val - current_val
run_rate = df_progress['Annual_Volume_B'].mean()

m1, m2, m3, m4 = st.columns(4)
m1.metric("Cumulative Progress", f"${current_val}B", f"{current_val/target_val:.1%}")
m2.metric("Remaining Gap", f"${gap:,.0f}B", "To be deployed by 2030")
m3.metric("2024 Annual Velocity", f"${df_progress['Annual_Volume_B'].iloc[-1]}B", "+17% YoY")
m4.metric("Avg Run Rate (4-yr)", f"${run_rate:.0f}B", "Historical Average")

# --- TABS LAYOUT ---
tabs = st.tabs(["🌊 Capital Flow (Sankey)", "📈 Predictive Models", "🔥 Risk Heatmap", "📊 Composition", "🧠 NLP Strategy"])

# === TAB 1: SANKEY DIAGRAM (New) ===
with tabs[0]:
    st.subheader("Capital Allocation Flow & Target Gap")
    st.markdown("Visualizing the flow from the **$2.5T Goal** down to specific allocated categories and the remaining unallocated capital.")
    
    # Sankey Logic
    # Nodes: 0=Target, 1=Achieved, 2=Gap, 3=Green, 4=Dev Fin, 5=Comm Dev
    
    fig_sankey = go.Figure(data=[go.Sankey(
        node = dict(
            pad = 15,
            thickness = 20,
            line = dict(color = "black", width = 0.5),
            label = ["$2.5T Target", "Achieved Capital ($900B)", "Remaining Gap ($1.6T)", "Green ($309B)", "Dev. Finance ($434B)", "Comm. Dev ($156B)"],
            color = ["#ffffff", "#2ea043", "#da3633", "#238636", "#d29922", "#58a6ff"]
        ),
        link = dict(
            source = [0, 0, 1, 1, 1], # Indices correspond to labels above
            target = [1, 2, 3, 4, 5],
            value =  [900, 1600, 309, 434, 156],
            color =  ["rgba(46, 160, 67, 0.4)", "rgba(218, 54, 51, 0.2)", "rgba(35, 134, 54, 0.4)", "rgba(210, 153, 34, 0.4)", "rgba(88, 166, 255, 0.4)"]
        )
    )])
    
    fig_sankey.update_layout(title_text="", font_size=14, height=500, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="white")
    st.plotly_chart(fig_sankey, use_container_width=True)

# === TAB 2: PREDICTIVE MODELING ===
with tabs[1]:
    st.subheader("Comparative Forecasting: Linear Trend vs. ML Logistic Growth")
    
    col_chart, col_details = st.columns([3, 1])
    
    with col_chart:
        # 1. Linear Model
        future_years_lin, pred_lin = run_linear_projection(df_progress, 6)
        
        # 2. Prophet Model (If available)
        prophet_status = "Available"
        try:
            from prophet import Prophet
            m = Prophet(growth='logistic')
            df_prophet = df_progress[['Date', 'Cumulative_B']].rename(columns={'Date': 'ds', 'Cumulative_B': 'y'})
            df_prophet['cap'] = 2500
            df_prophet['floor'] = 0
            m.fit(df_prophet)
            future = m.make_future_dataframe(periods=6, freq='Y')
            future['cap'] = 2500
            future['floor'] = 0
            forecast = m.predict(future)
            prophet_y = forecast['yhat'].values[-7:] # Get last 7 points including overlap
            prophet_ds = forecast['ds'].dt.year.values[-7:]
        except:
            prophet_status = "Unavailable (C++ missing)"
            prophet_y = []

        # Plotting
        fig_models = go.Figure()
        
        # Historical
        fig_models.add_trace(go.Scatter(x=df_progress['Year'], y=df_progress['Cumulative_B'], 
                                        mode='lines+markers', name='Historical', 
                                        line=dict(color='white', width=4)))
        
        # Linear
        fig_models.add_trace(go.Scatter(x=future_years_lin, y=pred_lin, 
                                        mode='lines', name='Linear Projection',
                                        line=dict(color='#ff7b72', dash='dot')))
        
        # Prophet
        if prophet_status == "Available":
             fig_models.add_trace(go.Scatter(x=prophet_ds, y=prophet_y, 
                                        mode='lines', name='Prophet (ML) Projection',
                                        line=dict(color='#2ea043', width=3)))
             
        # Target Line
        fig_models.add_hline(y=2500, line_dash="dash", line_color="#58a6ff", annotation_text="Target $2.5T")
        
        fig_models.update_layout(title="Forecasting Models to 2030", yaxis_title="$ Billions", template="plotly_dark", height=450)
        st.plotly_chart(fig_models, use_container_width=True)

    with col_details:
        st.info(f"**Linear Model**: Assumes growth continues at the exact same straight-line pace as 2021-2024.")
        st.success(f"**Prophet Model**: Uses logistic growth curves to account for saturation (capping at $2.5T).")
        st.metric("2030 Estimate (Linear)", f"${pred_lin[-1]:,.0f}B")
        if prophet_status == "Available":
            st.metric("2030 Estimate (ML)", f"${prophet_y[-1]:,.0f}B")

# === TAB 3: RISK HEATMAP (Corrected) ===
with tabs[2]:
    st.subheader("Sensitivity Analysis: 2030 Outcome Matrix")
    st.markdown("How sensitive is the 2030 total to changes in **Base Market Growth** vs. **AI-Driven Demand**?")
    
    # Generate Heatmap Data
    base_rates = np.arange(0, 11, 1) # 0% to 10%
    ai_rates = np.arange(0, 6, 1)    # 0% to 5%
    
    z_values = []
    
    for ai in ai_rates:
        row = []
        for base in base_rates:
            # Simulate to 2030
            cum = 900
            last_annual = 226
            for _ in range(6):
                growth = (base + ai) / 100
                last_annual = last_annual * (1 + growth)
                cum += last_annual
            row.append(cum)
        z_values.append(row)
        
    fig_heat = go.Figure(data=go.Heatmap(
        z=z_values,
        x=base_rates,
        y=ai_rates,
        colorscale='RdBu',
        zmid=2500, # Corrected parameter
        colorbar=dict(title="2030 Total ($B)")
    ))
    
    fig_heat.update_layout(
        title="2030 Forecast Matrix (Target = $2,500B)",
        xaxis_title="Base Market Growth Rate (%)",
        yaxis_title="AI Infrastructure Boost (%)",
        template="plotly_dark",
        height=500
    )
    
    st.plotly_chart(fig_heat, use_container_width=True)
    st.caption("Blue areas indicate exceeding the target. Red areas indicate missing the target.")

# === TAB 4: COMPOSITION (Treemap) ===
with tabs[3]:
    st.subheader("Portfolio Composition")
    
    c1, c2 = st.columns(2)
    with c1:
        # Treemap
        fig_tree = px.treemap(df_cat, path=[px.Constant("Achieved $900B"), 'Category'], values='Value',
                              color='Category', color_discrete_map={
                                  'Green': '#2ea043', 
                                  'Development Finance': '#d29922', 
                                  'Community Development': '#58a6ff'
                              })
        fig_tree.update_layout(template="plotly_dark", margin=dict(t=30, l=10, r=10, b=10))
        st.plotly_chart(fig_tree, use_container_width=True)
        
    with c2:
        # Donut
        fig_donut = px.pie(df_cat, values='Value', names='Category', hole=0.6, 
                           title="Allocation Split",
                           color_discrete_sequence=['#2ea043', '#d29922', '#58a6ff'])
        fig_donut.update_layout(template="plotly_dark")
        st.plotly_chart(fig_donut, use_container_width=True)

# === TAB 5: NLP ===
with tabs[4]:
    st.subheader("Strategic Text Classifier")
    
    col_nlp_input, col_nlp_viz = st.columns([1, 2])
    
    with col_nlp_input:
        excerpts = {
            "AI Energy Risk": "Rising energy demand from data centers driven by artificial intelligence is expected to influence our ability to meet our Electric Power target.",
            "Macro Headwinds": "Geopolitical tensions, diverging global economic growth and heightened interest rate volatility have slowed the pace of financing.",
            "Methodology Update": "We plan to continue to evaluate our SDT... and adjust it as we deem necessary, in light of latest climate science.",
            "Nuclear Financing": "In 2024, we hosted two private dialogues about financing nuclear energy with participants from technology and government."
        }
        selection = st.radio("Choose Statement:", list(excerpts.keys()))
        txt = excerpts[selection]
        st.markdown(f"> *{txt}*")
        
        run_nlp = st.button("Analyze Statement")

    with col_nlp_viz:
        if run_nlp:
            with st.spinner("Processing semantics..."):
                classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
                labels = ["Financial Risk", "Commercial Opportunity", "Regulatory/Governance", "Technology Shift"]
                res = classifier(txt, labels)
                
                df_res = pd.DataFrame({'Label': res['labels'], 'Confidence': res['scores']})
                
                fig_bar = px.bar(df_res, y='Label', x='Confidence', orientation='h', 
                                 text_auto='.1%', color='Confidence', color_continuous_scale='Greens')
                fig_bar.update_layout(template="plotly_dark", title="Topic Classification Probability")
                st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("Select a statement and click 'Analyze' to see NLP model results.")

# --- FOOTER ---
st.markdown("---")
st.markdown("<h5 style='text-align: center; color: #8b949e;'>Portfolio Analyzer | J.P. Morgan Chase Sustainability Data 2024 | Created by Ankit Kumar</h5>", unsafe_allow_html=True)
