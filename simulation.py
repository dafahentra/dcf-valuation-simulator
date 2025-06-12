import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set page configuration
st.set_page_config(page_title="Business Valuation", layout="wide")

# Function to calculate enterprise value
def s_ent_function(srevenue_0=100, sassets_0=80, swacc=0.12, sassets_to=1.3,
                   sgrowth=0.05, sgrowthstd=0.01, smargin=0.15,
                   smarginstd=0.03, sterminal_g=0.03, sterminalstd=0.01):
    
    # Random generation of parameters
    margin = np.random.normal(smargin, smarginstd, 5)
    growth = np.random.normal(sgrowth, sgrowthstd, 5)
    terminal_g = np.random.normal(sterminal_g, sterminalstd, 1)[0]
    
    # Revenue calculation
    revenue = srevenue_0 * np.cumprod(1 + growth)
    
    # NOPAT calculation
    nopat = revenue * margin
    
    # Assets calculation
    assets = revenue / sassets_to
    
    # ROIC and investment rate for terminal value
    roic_5 = margin[4] * sassets_to
    inv_rate_5 = terminal_g / roic_5
    
    # Net investment calculation
    net_inv = np.zeros(5)
    net_inv[0] = assets[0] - sassets_0
    net_inv[1:5] = np.diff(assets)
    net_inv[4] = nopat[4] * inv_rate_5
    
    # Free cash flow to firm
    fcff = nopat - net_inv
    
    # Discount factors
    disc_factors = 1 / (1 + swacc) ** np.arange(1, 6)
    
    # Terminal value
    terminal_value = (fcff[4] * (1 + terminal_g)) / (swacc - terminal_g)
    fcff[4] = fcff[4] + terminal_value
    
    # Enterprise value
    ent_value = np.sum(fcff * disc_factors)
    
    return ent_value

# Streamlit app
st.title("Business Valuation Dashboard")
st.markdown("### Monte Carlo Simulation & Sensitivity Analysis")

# Sidebar for inputs
st.sidebar.header("Parameters")

wacc = st.sidebar.slider("WACC:", min_value=0.08, max_value=0.16, value=0.12, step=0.01)
growth = st.sidebar.slider("Growth Rate:", min_value=0.03, max_value=0.10, value=0.05, step=0.01)
margin = st.sidebar.slider("Margin:", min_value=0.10, max_value=0.20, value=0.15, step=0.01)
terminal_growth = st.sidebar.slider("Terminal Growth Rate:", min_value=0.02, max_value=0.06, value=0.03, step=0.01)

# Additional parameters
st.sidebar.markdown("### Additional Parameters")
revenue_0 = st.sidebar.number_input("Initial Revenue:", value=100, min_value=10)
assets_0 = st.sidebar.number_input("Initial Assets:", value=80, min_value=10)
assets_to = st.sidebar.number_input("Assets Turnover:", value=1.3, min_value=0.5, max_value=3.0, step=0.1)

# Volatility parameters
st.sidebar.markdown("### Volatility Parameters")
growth_std = st.sidebar.slider("Growth Std Dev:", min_value=0.005, max_value=0.02, value=0.01, step=0.001)
margin_std = st.sidebar.slider("Margin Std Dev:", min_value=0.01, max_value=0.05, value=0.03, step=0.01)
terminal_std = st.sidebar.slider("Terminal Growth Std Dev:", min_value=0.005, max_value=0.02, value=0.01, step=0.001)

# Number of simulations
n_simulations = st.sidebar.number_input("Number of Simulations:", value=10000, min_value=1000, max_value=100000, step=1000)

# Run simulation button
if st.sidebar.button("Run Simulation", type="primary"):
    
    # Progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Run Monte Carlo simulation
    np.random.seed(42)  # For reproducibility
    svalues = []
    
    for i in range(n_simulations):
        if i % 1000 == 0:
            progress_bar.progress(i / n_simulations)
            status_text.text(f'Running simulation {i}/{n_simulations}...')
        
        value = s_ent_function(
            srevenue_0=revenue_0,
            sassets_0=assets_0,
            swacc=wacc,
            sassets_to=assets_to,
            sgrowth=growth,
            sgrowthstd=growth_std,
            smargin=margin,
            smarginstd=margin_std,
            sterminal_g=terminal_growth,
            sterminalstd=terminal_std
        )
        svalues.append(value)
    
    progress_bar.progress(1.0)
    status_text.text('Simulation complete!')
    
    svalues = np.array(svalues)
    
    # Create two columns for layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Create density plot using Plotly
        st.subheader("Valuation Distribution")
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=svalues,
            histnorm='probability density',
            name='Valuation',
            nbinsx=50,
            marker_color='skyblue'
        ))
        fig.update_layout(
            xaxis_title="Valuation Value",
            yaxis_title="Density",
            showlegend=False,
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Summary statistics
        st.subheader("Summary Statistics")
        summary_df = pd.DataFrame({
            'Statistic': ['Count', 'Mean', 'Std Dev', 'Min', '25%', 'Median', '75%', 'Max'],
            'Value': [
                f"{len(svalues):,}",
                f"{np.mean(svalues):.2f}",
                f"{np.std(svalues):.2f}",
                f"{np.min(svalues):.2f}",
                f"{np.percentile(svalues, 25):.2f}",
                f"{np.median(svalues):.2f}",
                f"{np.percentile(svalues, 75):.2f}",
                f"{np.max(svalues):.2f}"
            ]
        })
        st.dataframe(summary_df, hide_index=True)
        
        # Additional metrics
        st.metric("Expected Value", f"{np.mean(svalues):.2f}")
        st.metric("95% Confidence Interval", 
                 f"[{np.percentile(svalues, 2.5):.2f}, {np.percentile(svalues, 97.5):.2f}]")
    
    # Box plot
    st.subheader("Valuation Box Plot")
    fig_box = go.Figure()
    fig_box.add_trace(go.Box(
        y=svalues,
        name='Valuation',
        boxpoints='outliers',
        marker_color='lightblue'
    ))
    fig_box.update_layout(
        yaxis_title="Valuation Value",
        showlegend=False,
        height=300
    )
    st.plotly_chart(fig_box, use_container_width=True)
    
    # Store results in session state
    st.session_state['simulation_results'] = svalues

# Sensitivity Analysis Section
st.markdown("---")
st.header("Sensitivity Analysis")

col3, col4 = st.columns(2)

with col3:
    st.subheader("WACC Range")
    wacc_min = st.slider("Min WACC:", min_value=0.06, max_value=0.12, value=0.08, step=0.01)
    wacc_max = st.slider("Max WACC:", min_value=0.12, max_value=0.20, value=0.16, step=0.01)
    wacc_steps = st.slider("Number of Steps:", min_value=5, max_value=20, value=9)

with col4:
    st.subheader("Growth Rate Range")
    growth_min = st.slider("Min Growth:", min_value=0.01, max_value=0.05, value=0.03, step=0.01)
    growth_max = st.slider("Max Growth:", min_value=0.05, max_value=0.10, value=0.07, step=0.01)
    growth_steps = st.slider("Number of Steps:", min_value=5, max_value=20, value=5)

if st.button("Run Sensitivity Analysis", type="secondary"):
    
    # Create parameter grid
    wacc_range = np.linspace(wacc_min, wacc_max, wacc_steps)
    growth_range = np.linspace(growth_min, growth_max, growth_steps)
    
    # Matrix to store results
    sensitivity_results = np.zeros((len(growth_range), len(wacc_range)))
    
    # Progress indicator
    progress = st.progress(0)
    total_iterations = len(wacc_range) * len(growth_range)
    current_iteration = 0
    
    # Calculate valuations for each combination
    for i, g in enumerate(growth_range):
        for j, w in enumerate(wacc_range):
            # Run a smaller simulation for each combination
            temp_values = []
            for _ in range(100):  # Reduced number for sensitivity analysis
                temp_values.append(s_ent_function(
                    srevenue_0=revenue_0,
                    sassets_0=assets_0,
                    swacc=w,
                    sassets_to=assets_to,
                    sgrowth=g,
                    sgrowthstd=growth_std,
                    smargin=margin,
                    smarginstd=margin_std,
                    sterminal_g=terminal_growth,
                    sterminalstd=terminal_std
                ))
            sensitivity_results[i, j] = np.mean(temp_values)
            
            current_iteration += 1
            progress.progress(current_iteration / total_iterations)
    
    # Create heatmap
    st.subheader("Sensitivity Heatmap: Valuation vs WACC and Growth Rate")
    
    fig_heatmap = go.Figure(data=go.Heatmap(
        z=sensitivity_results,
        x=[f"{w:.2%}" for w in wacc_range],
        y=[f"{g:.2%}" for g in growth_range],
        colorscale='Viridis',
        text=[[f"{val:.0f}" for val in row] for row in sensitivity_results],
        texttemplate="%{text}",
        textfont={"size": 10},
        hoverongaps=False
    ))
    
    fig_heatmap.update_layout(
        xaxis_title="WACC",
        yaxis_title="Growth Rate",
        height=500
    )
    
    st.plotly_chart(fig_heatmap, use_container_width=True)
    
    # Create 3D surface plot
    st.subheader("3D Surface Plot")
    
    X, Y = np.meshgrid(wacc_range, growth_range)
    
    fig_3d = go.Figure(data=[go.Surface(
        x=X * 100,  # Convert to percentage
        y=Y * 100,  # Convert to percentage
        z=sensitivity_results,
        colorscale='Viridis'
    )])
    
    fig_3d.update_layout(
        scene=dict(
            xaxis_title='WACC (%)',
            yaxis_title='Growth Rate (%)',
            zaxis_title='Valuation'
        ),
        height=600
    )
    
    st.plotly_chart(fig_3d, use_container_width=True)

# Add export functionality
if 'simulation_results' in st.session_state:
    st.markdown("---")
    st.subheader("Export Results")
    
    # Create DataFrame for export
    export_df = pd.DataFrame({
        'Simulation_Number': range(1, len(st.session_state['simulation_results']) + 1),
        'Valuation': st.session_state['simulation_results']
    })
    
    # Convert to CSV
    csv = export_df.to_csv(index=False)
    
    st.download_button(
        label="Download Simulation Results as CSV",
        data=csv,
        file_name='valuation_simulation_results.csv',
        mime='text/csv'
    )