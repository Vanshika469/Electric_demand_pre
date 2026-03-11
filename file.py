import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import json
import requests
from streamlit_lottie import st_lottie
import base64
from io import BytesIO

# Page Configuration
st.set_page_config(
    page_title="Helios AI - Solar Energy Prediction",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Modern UI
def load_css():
    st.markdown("""
    <style>
    /* Global Styles */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Glassmorphism Cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
    }
    
    /* Metric Cards */
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, #FF6B6B 0%, #FFE66D 100%);
        padding: 15px;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        border: 1px solid rgba(255, 255, 255, 0.3);
    }
    
    div[data-testid="metric-container"] > label {
        color: white !important;
        font-weight: 600 !important;
    }
    
    div[data-testid="metric-container"] > div {
        color: white !important;
        font-size: 2rem !important;
        font-weight: bold !important;
    }
    
    /* Headers */
    .main-header {
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #FFA500 0%, #FF6347 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
        margin-bottom: 2rem;
    }
    
    .sub-header {
        font-size: 1.8rem;
        color: #2C3E50;
        font-weight: 600;
        margin: 1.5rem 0;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 30px;
        font-weight: 600;
        transition: transform 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background: rgba(255, 255, 255, 0.1);
        padding: 10px;
        border-radius: 15px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background: rgba(255, 255, 255, 0.2);
        border-radius: 10px;
        color: white;
        font-weight: 600;
        padding: 0 20px;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #FFA500 0%, #FF6347 100%);
    }
    
    /* Sliders */
    .stSlider > div > div {
        background: linear-gradient(90deg, #FFA500 0%, #FF6347 100%);
    }
    
    /* Info boxes */
    .stInfo {
        background: rgba(100, 126, 234, 0.1);
        border: 2px solid #667eea;
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# Load Lottie Animation
def load_lottie_url(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Cache model training
@st.cache_resource
def train_model():
    """Train the Random Forest model with caching"""
    try:
        # Load dataset
        df = pd.read_csv('ElectricData.csv')
        
        # Data cleaning
        df.columns = df.columns.str.strip()
        
        numeric_cols = ['Season', 'Day_of_the_week', 'DHI', 'DNI', 'GHI', 
                       'Wind_speed', 'Humidity', 'Temperature', 'PV_production', 
                       'Wind_production', 'Electric_demand']
        
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df = df.dropna(subset=numeric_cols)
        
        # Features and target
        features = ['Season', 'Day_of_the_week', 'DHI', 'DNI', 'GHI', 
                   'Wind_speed', 'Humidity', 'Temperature']
        target = 'Electric_demand'
        
        X = df[features]
        y = df[target]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train model
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf_model.fit(X_train, y_train)
        
        # Predictions and metrics
        y_pred = rf_model.predict(X_test)
        
        metrics = {
            'mae': mean_absolute_error(y_test, y_pred),
            'mse': mean_squared_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'r2': r2_score(y_test, y_pred),
            'mape': np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        }
        
        return rf_model, X_test, y_test, y_pred, metrics, features, df
    
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None, None, None, None, None

# Create gauge chart
def create_gauge_chart(value, title, max_value=5000):
    """Create an interactive gauge chart"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = value,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title, 'font': {'size': 24}},
        delta = {'reference': max_value * 0.8},
        gauge = {
            'axis': {'range': [None, max_value], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, max_value*0.25], 'color': '#90EE90'},
                {'range': [max_value*0.25, max_value*0.5], 'color': '#FFD700'},
                {'range': [max_value*0.5, max_value*0.75], 'color': '#FFA500'},
                {'range': [max_value*0.75, max_value], 'color': '#FF6347'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': max_value * 0.9
            }
        }
    ))
    fig.update_layout(
        height=400,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': "darkblue", 'family': "Arial"}
    )
    return fig

# Create feature importance radar chart
def create_radar_chart(feature_importance, features):
    """Create an interactive radar chart for feature importance"""
    fig = go.Figure(data=go.Scatterpolar(
        r=feature_importance,
        theta=features,
        fill='toself',
        marker=dict(color='#FFA500'),
        line=dict(color='#FF6347', width=2)
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max(feature_importance)]
            )
        ),
        showlegend=False,
        title="Feature Importance Radar Chart",
        height=500,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig

# Main application
def main():
    load_css()
    
    # Header
    st.markdown('<h1 class="main-header">☀️ Helios AI: Solar Energy Prediction System</h1>', 
                unsafe_allow_html=True)
    
    # Load model
    model, X_test, y_test, y_pred, metrics, features, df = train_model()
    
    if model is None:
        st.error("⚠️ Please ensure ElectricData.csv is in the same directory")
        st.stop()
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🏠 Home", "📊 Model Analytics", 
                                       "🔮 Live Predictor", "📂 Batch Processing"])
    
    # Tab 1: Home
    with tab1:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # Lottie animation
            lottie_url = "https://assets5.lottiefiles.com/packages/lf20_qxgbijbw.json"
            lottie_json = load_lottie_url(lottie_url)
            if lottie_json:
                st_lottie(lottie_json, height=300, key="solar")
        
        with col2:
            st.markdown("""
            <div class="glass-card">
                <h2 style="color: #2C3E50;">🌟 Project Mission</h2>
                <p style="font-size: 1.1rem; line-height: 1.8;">
                    Helios AI leverages advanced machine learning to predict solar energy production 
                    and electricity demand with unprecedented accuracy. Our system helps optimize 
                    energy distribution, reduce waste, and accelerate the transition to sustainable energy.
                </p>
                <br>
                <h3 style="color: #2C3E50;">🎯 Key Features</h3>
                <ul style="font-size: 1.05rem; line-height: 2;">
                    <li>🤖 Random Forest ML Model with 100+ estimators</li>
                    <li>📈 Real-time prediction with interactive controls</li>
                    <li>📊 Comprehensive analytics and visualizations</li>
                    <li>🎨 Modern, responsive UI with glassmorphism design</li>
                    <li>⚡ Batch processing for large-scale predictions</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        # Quick Stats
        st.markdown("<h2 class='sub-header'>📈 Quick Performance Stats</h2>", unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Model Accuracy", f"{metrics['r2']:.2%}", "✅ Excellent")
        with col2:
            st.metric("MAE", f"{metrics['mae']:.0f} MWh", "↓ Low Error")
        with col3:
            st.metric("Training Samples", f"{len(df)*0.8:.0f}", "📊")
        with col4:
            st.metric("Features Used", f"{len(features)}", "🔍")
    
    # Tab 2: Model Analytics
    with tab2:
        st.markdown("<h2 class='sub-header'>🎯 Model Performance Metrics</h2>", unsafe_allow_html=True)
        
        # KPI Cards
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="R² Score",
                value=f"{metrics['r2']:.4f}",
                delta=f"{(metrics['r2'] - 0.85):.2%} above baseline",
                help="Coefficient of determination - closer to 1 is better"
            )
        
        with col2:
            st.metric(
                label="MAE",
                value=f"{metrics['mae']:.2f}",
                help="Mean Absolute Error in MWh"
            )
        
        with col3:
            st.metric(
                label="RMSE",
                value=f"{metrics['rmse']:.2f}",
                help="Root Mean Squared Error"
            )
        
        with col4:
            st.metric(
                label="MAPE",
                value=f"{metrics['mape']:.2f}%",
                help="Mean Absolute Percentage Error"
            )
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            # Actual vs Predicted Scatter
            fig_scatter = px.scatter(
                x=y_test, y=y_pred,
                labels={'x': 'Actual Demand (MWh)', 'y': 'Predicted Demand (MWh)'},
                title='Actual vs Predicted Electric Demand',
                color=np.abs(y_test - y_pred),
                color_continuous_scale='Viridis'
            )
            fig_scatter.add_trace(
                go.Scatter(x=[y_test.min(), y_test.max()], 
                          y=[y_test.min(), y_test.max()],
                          mode='lines', name='Perfect Prediction',
                          line=dict(color='red', dash='dash'))
            )
            fig_scatter.update_layout(height=400)
            st.plotly_chart(fig_scatter, use_container_width=True)
        
        with col2:
            # Residual Plot
            residuals = y_test - y_pred
            fig_residual = px.scatter(
                x=y_pred, y=residuals,
                labels={'x': 'Predicted Values', 'y': 'Residuals'},
                title='Residual Plot',
                color=np.abs(residuals),
                color_continuous_scale='RdYlBu_r'
            )
            fig_residual.add_hline(y=0, line_dash="dash", line_color="red")
            fig_residual.update_layout(height=400)
            st.plotly_chart(fig_residual, use_container_width=True)
        
        # Feature Importance
        st.markdown("<h3 class='sub-header'>🔍 Feature Importance Analysis</h3>", unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Bar chart for feature importance
            feature_importance = model.feature_importances_
            importance_df = pd.DataFrame({
                'Feature': features,
                'Importance': feature_importance
            }).sort_values('Importance', ascending=True)
            
            fig_importance = px.bar(
                importance_df, 
                x='Importance', 
                y='Feature',
                orientation='h',
                title='Feature Importance Scores',
                color='Importance',
                color_continuous_scale='Sunset'
            )
            fig_importance.update_layout(height=400)
            st.plotly_chart(fig_importance, use_container_width=True)
        
        with col2:
            # Radar chart
            radar_fig = create_radar_chart(feature_importance, features)
            st.plotly_chart(radar_fig, use_container_width=True)
        
        # Demand Trends
        st.markdown("<h3 class='sub-header'>📈 Demand Trends Comparison</h3>", unsafe_allow_html=True)
        
        samples_to_show = st.slider("Number of samples to display", 10, 200, 50, 
                                    help="Adjust to see more or fewer samples")
        
        fig_trends = go.Figure()
        fig_trends.add_trace(go.Scatter(
            x=list(range(samples_to_show)),
            y=y_test.values[:samples_to_show],
            mode='lines+markers',
            name='Actual Demand',
            line=dict(color='#667eea', width=2),
            marker=dict(size=6)
        ))
        fig_trends.add_trace(go.Scatter(
            x=list(range(samples_to_show)),
            y=y_pred[:samples_to_show],
            mode='lines+markers',
            name='Predicted Demand',
            line=dict(color='#FFA500', width=2, dash='dot'),
            marker=dict(size=6)
        ))
        fig_trends.update_layout(
            title='Actual vs Predicted Demand Trends',
            xaxis_title='Sample Index',
            yaxis_title='Electric Demand (MWh)',
            hovermode='x unified',
            height=400
        )
        st.plotly_chart(fig_trends, use_container_width=True)
    
    # Tab 3: Live Predictor
    with tab3:
        st.markdown("<h2 class='sub-header'>🔮 Real-Time Energy Demand Predictor</h2>", 
                   unsafe_allow_html=True)
        
        st.info("📝 Adjust the parameters below to get instant predictions")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Input controls
            st.markdown("#### Environmental Parameters")
            
            c1, c2 = st.columns(2)
            
            with c1:
                season = st.selectbox(
                    "Season",
                    options=[1, 2, 3, 4],
                    format_func=lambda x: ['Winter', 'Spring', 'Summer', 'Fall'][x-1],
                    help="Select the current season"
                )
                
                day_of_week = st.selectbox(
                    "Day of the Week",
                    options=list(range(1, 8)),
                    format_func=lambda x: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'][x-1],
                    help="Select the day of the week"
                )
                
                temperature = st.slider(
                    "Temperature (°C)",
                    min_value=-10.0, max_value=45.0, value=20.0, step=0.5,
                    help="Ambient temperature in Celsius"
                )
                
                humidity = st.slider(
                    "Humidity (%)",
                    min_value=0.0, max_value=100.0, value=50.0, step=1.0,
                    help="Relative humidity percentage"
                )
            
            with c2:
                ghi = st.slider(
                    "Global Horizontal Irradiance (W/m²)",
                    min_value=0.0, max_value=1000.0, value=500.0, step=10.0,
                    help="Total solar radiation received"
                )
                
                dni = st.slider(
                    "Direct Normal Irradiance (W/m²)",
                    min_value=0.0, max_value=1000.0, value=400.0, step=10.0,
                    help="Direct beam solar radiation"
                )
                
                dhi = st.slider(
                    "Diffuse Horizontal Irradiance (W/m²)",
                    min_value=0.0, max_value=500.0, value=100.0, step=10.0,
                    help="Scattered solar radiation"
                )
                
                wind_speed = st.slider(
                    "Wind Speed (m/s)",
                    min_value=0.0, max_value=30.0, value=5.0, step=0.5,
                    help="Wind speed in meters per second"
                )
            
            # Prepare input for prediction
            input_data = pd.DataFrame({
                'Season': [season],
                'Day_of_the_week': [day_of_week],
                'DHI': [dhi],
                'DNI': [dni],
                'GHI': [ghi],
                'Wind_speed': [wind_speed],
                'Humidity': [humidity],
                'Temperature': [temperature]
            })
            
            # Make prediction
            prediction = model.predict(input_data)[0]
            
        with col2:
            # Display prediction with gauge
            st.markdown("#### Predicted Electric Demand")
            gauge_fig = create_gauge_chart(prediction, "Demand (MWh)", 
                                          max_value=df['Electric_demand'].max())
            st.plotly_chart(gauge_fig, use_container_width=True)
            
            # Prediction confidence
            st.markdown("#### Prediction Details")
            st.success(f"🎯 **Predicted Demand:** {prediction:.2f} MWh")
            
            # Compare with average
            avg_demand = df['Electric_demand'].mean()
            diff_from_avg = ((prediction - avg_demand) / avg_demand) * 100
            
            if diff_from_avg > 0:
                st.warning(f"📈 {diff_from_avg:.1f}% above average demand")
            else:
                st.info(f"📉 {abs(diff_from_avg):.1f}% below average demand")
        
        # Parameter Summary
        with st.expander("📊 View Current Parameters", expanded=False):
            params_df = pd.DataFrame({
                'Parameter': features,
                'Current Value': input_data.values[0],
                'Min in Data': [df[f].min() for f in features],
                'Max in Data': [df[f].max() for f in features],
                'Average': [df[f].mean() for f in features]
            })
            st.dataframe(params_df, use_container_width=True)
    
    # Tab 4: Batch Processing
    with tab4:
        st.markdown("<h2 class='sub-header'>📂 Batch Prediction Processing</h2>", 
                   unsafe_allow_html=True)
        
        st.info("📤 Upload a CSV file with the required features for batch predictions")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=['csv'],
            help="File should contain columns: " + ", ".join(features)
        )
        
        if uploaded_file is not None:
            try:
                # Read uploaded file
                batch_df = pd.read_csv(uploaded_file)
                
                st.success(f"✅ File uploaded successfully! Found {len(batch_df)} rows")
                
                # Display preview
                with st.expander("📋 Preview uploaded data", expanded=True):
                    st.dataframe(batch_df.head(10), use_container_width=True)
                
                # Check for required columns
                missing_cols = set(features) - set(batch_df.columns)
                if missing_cols:
                    st.error(f"❌ Missing required columns: {', '.join(missing_cols)}")
                else:
                    # Make predictions
                    if st.button("🚀 Generate Predictions", use_container_width=True):
                        with st.spinner("Processing predictions..."):
                            batch_X = batch_df[features]
                            batch_predictions = model.predict(batch_X)
                            
                            # Add predictions to dataframe
                            result_df = batch_df.copy()
                            result_df['Predicted_Demand'] = batch_predictions
                            
                            st.success("✅ Predictions completed!")
                            
                            # Display results
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.markdown("#### Prediction Statistics")
                                st.metric("Total Predictions", len(batch_predictions))
                                st.metric("Average Predicted Demand", 
                                         f"{batch_predictions.mean():.2f} MWh")
                                st.metric("Max Predicted Demand", 
                                         f"{batch_predictions.max():.2f} MWh")
                                st.metric("Min Predicted Demand", 
                                         f"{batch_predictions.min():.2f} MWh")
                            
                            with col2:
                                # Distribution plot
                                fig_dist = px.histogram(
                                    batch_predictions,
                                    nbins=30,
                                    title="Distribution of Predictions",
                                    labels={'value': 'Predicted Demand (MWh)', 'count': 'Frequency'},
                                    color_discrete_sequence=['#FFA500']
                                )
                                fig_dist.update_layout(height=300)
                                st.plotly_chart(fig_dist, use_container_width=True)
                            
                            # Download results
                            st.markdown("#### 📥 Download Results")
                            
                            csv = result_df.to_csv(index=False)
                            b64 = base64.b64encode(csv.encode()).decode()
                            href = f'<a href="data:file/csv;base64,{b64}" download="predictions.csv">📥 Download Predictions CSV</a>'
                            st.markdown(href, unsafe_allow_html=True)
                            
                            # Display full results
                            with st.expander("📊 View All Predictions", expanded=False):
                                st.dataframe(result_df, use_container_width=True)
                
            except Exception as e:
                st.error(f"❌ Error processing file: {str(e)}")
        
        else:
            # Sample template
            st.markdown("#### 📝 Need a template?")
            
            template_df = pd.DataFrame({
                'Season': [1, 2, 3, 4, 1],
                'Day_of_the_week': [1, 3, 5, 7, 2],
                'DHI': [100, 120, 150, 80, 110],
                'DNI': [400, 450, 500, 350, 420],
                'GHI': [500, 550, 600, 450, 520],
                'Wind_speed': [5, 7, 3, 10, 6],
                'Humidity': [45, 60, 30, 70, 50],
                'Temperature': [20, 25, 30, 15, 22]
            })
            
            csv_template = template_df.to_csv(index=False)
            b64 = base64.b64encode(csv_template.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="template.csv">📥 Download Sample Template</a>'
            st.markdown(href, unsafe_allow_html=True)
            
            st.markdown("##### Template Preview:")
            st.dataframe(template_df, use_container_width=True)

# Run the app
if __name__ == "__main__":
    main()
