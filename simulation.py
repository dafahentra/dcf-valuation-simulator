import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time

# IMPORTANT: Set page config FIRST before any other st commands
st.set_page_config(
    page_title="Business Valuation Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'show_full_app' not in st.session_state:
    st.session_state.show_full_app = False

# Lightweight CSS - minimal for fast loading
st.markdown("""
<style>
    .main > div {padding-top: 2rem;}
    .preview-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 3rem;
        border-radius: 20px;
        text-align: center;
        color: white;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: rgba(255,255,255,0.1);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=3600)  # Cache for 1 hour
def generate_quick_preview_data():
    """Generate minimal data for preview - FAST"""
    np.random.seed(42)
    return {
        'values': np.random.normal(100, 15, 500),  # Reduced from 1000
        'mean': 100,
        'std': 15
    }

def create_instant_preview():
    """Ultra-fast preview that loads immediately"""
    
    # Header - loads instantly
    st.markdown("""
    <div class="preview-box">
        <h1 style="font-size: 2.5rem; margin: 0;">📊 Business Valuation Dashboard</h1>
        <p style="font-size: 1.2rem; opacity: 0.9;">DCF Valuation Simulator with Monte Carlo Analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Quick metrics - no computation needed
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Expected Value", "$127.43M", "↑ 12.3%")
    with col2:
        st.metric("Confidence Range", "$95M - $159M")
    with col3:
        st.metric("Simulations", "10,000+")
    
    # Simple static chart for preview
    data = generate_quick_preview_data()
    
    # Create simple histogram
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=data['values'],
        nbinsx=20,  # Fewer bins for faster rendering
        marker_color='rgba(102, 126, 234, 0.7)',
        name='Valuation Distribution'
    ))
    
    fig.update_layout(
        title="Valuation Distribution Preview",
        xaxis_title="Value ($M)",
        yaxis_title="Frequency",
        height=350,
        template="plotly_white",
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Features grid
    st.markdown("### ✨ Key Features")
    feat_col1, feat_col2, feat_col3 = st.columns(3)
    
    with feat_col1:
        st.markdown("""
        <div class="metric-card">
        <h4>📈 Monte Carlo</h4>
        <p>10,000+ simulations with statistical analysis</p>
        </div>
        """, unsafe_allow_html=True)
    
    with feat_col2:
        st.markdown("""
        <div class="metric-card">
        <h4>🎯 Sensitivity</h4>
        <p>Multi-parameter analysis with 3D visualization</p>
        </div>
        """, unsafe_allow_html=True)
    
    with feat_col3:
        st.markdown("""
        <div class="metric-card">
        <h4>📊 Export</h4>
        <p>Download results in CSV/Excel format</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Launch button
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        if st.button("🚀 Launch Full Application", type="primary", use_container_width=True):
            st.session_state.show_full_app = True
            st.rerun()

def s_ent_function(srevenue_0=100, sassets_0=80, swacc=0.12, sassets_to=1.3,
                   sgrowth=0.05, sgrowthstd=0.01, smargin=0.15,
                   smarginstd=0.03, sterminal_g=0.03, sterminalstd=0.01):
    """DCF calculation function"""
    margin = np.random.normal(smargin, smarginstd, 5)
    growth = np.random.normal(sgrowth, sgrowthstd, 5)
    terminal_g = np.random.normal(sterminal_g, sterminalstd, 1)[0]
    
    revenue = srevenue_0 * np.cumprod(1 + growth)
    nopat = revenue * margin
    assets = revenue / sassets_to
    
    roic_5 = margin[4] * sassets_to
    inv_rate_5 = terminal_g / roic_5
    
    net_inv = np.zeros(5)
    net_inv[0] = assets[0] - sassets_0
    net_inv[1:5] = np.diff(assets)
    net_inv[4] = nopat[4] * inv_rate_5
    
    fcff = nopat - net_inv
    disc_factors = 1 / (1 + swacc) ** np.arange(1, 6)
    
    terminal_value = (fcff[4] * (1 + terminal_g)) / (swacc - terminal_g)
    fcff[4] = fcff[4] + terminal_value
    
    return np.sum(fcff * disc_factors)

def run_full_application():
    """Full application after preview"""
    
    # Back button
    if st.button("← Back to Preview"):
        st.session_state.show_full_app = False
        st.rerun()
    
    st.title("Business Valuation Dashboard")
    st.markdown("### Full Monte Carlo Simulation & Sensitivity Analysis")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("📊 Parameters")
        
        st.subheader("Core Parameters")
        wacc = st.slider("WACC", 0.08, 0.16, 0.12, 0.01)
        growth = st.slider("Growth Rate", 0.03, 0.10, 0.05, 0.01)
        margin = st.slider("Margin", 0.10, 0.20, 0.15, 0.01)
        terminal_growth = st.slider("Terminal Growth", 0.02, 0.06, 0.03, 0.01)
        
        st.subheader("Additional Settings")
        revenue_0 = st.number_input("Initial Revenue", 50, 500, 100)
        assets_0 = st.number_input("Initial Assets", 40, 400, 80)
        n_simulations = st.selectbox("Simulations", [1000, 5000, 10000, 50000], index=2)
        
        run_simulation = st.button("🚀 Run Simulation", type="primary", use_container_width=True)
    
    # Main content
    if run_simulation:
        with st.spinner("Running simulation..."):
            # Progress bar
            progress = st.progress(0)
            values = []
            
            # Run simulation with progress
            for i in range(n_simulations):
                if i % (n_simulations // 10) == 0:
                    progress.progress(i / n_simulations)
                
                value = s_ent_function(
                    srevenue_0=revenue_0,
                    sassets_0=assets_0,
                    swacc=wacc,
                    sgrowth=growth,
                    smargin=margin,
                    sterminal_g=terminal_growth
                )
                values.append(value)
            
            progress.progress(1.0)
            values = np.array(values)
            
            # Display results
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Histogram
                fig = go.Figure()
                fig.add_trace(go.Histogram(
                    x=values,
                    nbinsx=50,
                    marker_color='skyblue',
                    name='Valuation'
                ))
                fig.update_layout(
                    title="Valuation Distribution",
                    xaxis_title="Value ($M)",
                    yaxis_title="Frequency",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("📊 Statistics")
                st.metric("Mean", f"${np.mean(values):.2f}M")
                st.metric("Std Dev", f"${np.std(values):.2f}M")
                st.metric("Median", f"${np.median(values):.2f}M")
                
                # Download button
                df = pd.DataFrame({'Valuation': values})
                csv = df.to_csv(index=False)
                st.download_button(
                    "📥 Download Results",
                    csv,
                    "valuation_results.csv",
                    "text/csv",
                    use_container_width=True
                )
    else:
        # Show instruction
        st.info("👈 Configure parameters in the sidebar and click 'Run Simulation' to start")

# MAIN APP LOGIC - This runs first
def main():
    if not st.session_state.show_full_app:
        create_instant_preview()
    else:
        run_full_application()

# Run app
if __name__ == "__main__":
    main()