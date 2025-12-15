import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import joblib
from data_processing import get_data_from_db

# Page Configuration
st.set_page_config(page_title="Blinkit Analytics", layout="wide")

# Title and Description
st.title("Blinkit Business Intelligence Dashboard")
st.markdown("Marketing ROI, Delivery Operations, and Customer Feedback Analysis")

# Fetch Data
@st.cache_data
def load_data():
    return get_data_from_db()

roas_df, orders_df = load_data()

if roas_df is None:
    st.error("Failed to load data. Please check database connection.")
else:
    # Sidebar Filters
    st.sidebar.header("Filters")
    # Date Range Filter
    min_date = roas_df['date'].min()
    max_date = roas_df['date'].max()
    
    date_range = st.sidebar.date_input(
        "Select Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    
    # Filter Data
    if len(date_range) == 2:
        start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
        filtered_roas = roas_df[(roas_df['date'] >= start_date) & (roas_df['date'] <= end_date)]
    else:
        filtered_roas = roas_df

    # Create Tabs
    tab1, tab2, tab3 = st.tabs(["Marketing Trends", "Delivery Prediction", "GenAI Insights"])

    # --- Tab 1: Marketing Trends ---
    with tab1:
        st.header("Marketing Spend vs Revenue Analysis")
        
        # KPIS
        col1, col2, col3 = st.columns(3)
        total_spend = filtered_roas['total_spend'].sum()
        total_revenue = filtered_roas['total_revenue'].sum()
        avg_roas = total_revenue / total_spend if total_spend > 0 else 0
        
        col1.metric("Total Spend", f"₹{total_spend:,.2f}")
        col2.metric("Total Revenue", f"₹{total_revenue:,.2f}")
        col3.metric("Overall ROAS", f"{avg_roas:.2f}x")
        
        # Dual Axis Chart
        fig = go.Figure()
        
        # Bar chart for Spend
        fig.add_trace(go.Bar(
            x=filtered_roas['date'],
            y=filtered_roas['total_spend'],
            name='Spend',
            marker_color='indianred',
            opacity=0.6
        ))
        
        # Line chart for Revenue
        fig.add_trace(go.Scatter(
            x=filtered_roas['date'],
            y=filtered_roas['total_revenue'],
            name='Revenue',
            yaxis='y2',
            line=dict(color='royalblue', width=3)
        ))
        
        # Layout
        fig.update_layout(
            title='Daily Spend vs Revenue',
            xaxis_title='Date',
            yaxis=dict(title='Spend (₹)'),
            yaxis2=dict(
                title='Revenue (₹)',
                overlaying='y',
                side='right'
            ),
            hovermode='x unified',
            legend=dict(x=0, y=1.2, orientation='h')
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        with st.expander("View Raw Data"):
            st.dataframe(filtered_roas)

    # --- Tab 2: Prediction ---
    with tab2:
        st.header("Delivery Delay Risk Calculator")
        
        # Load Model
        try:
            model = joblib.load('src/model.pkl')
            le_area = joblib.load('src/label_encoder_area.pkl')
            model_loaded = True
        except Exception as e:
            st.warning(f"Model not found. Please train the model first. Error: {e}")
            model_loaded = False

        if model_loaded:
            col1, col2 = st.columns(2)
            
            with col1:
                # Inputs
                areas = le_area.classes_
                selected_area = st.selectbox("Delivery Area", areas)
                
                # Dynamic slider for time
                promised_hour = st.slider("Promised Delivery Hour (0-23)", 0, 23, 10)
                promised_day = st.selectbox("Day of Week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
                
                order_total = st.number_input("Order Total (₹)", min_value=100.0, value=500.0)

            with col2:
                # Prepare Input for Prediction
                # Map Day to 0-6
                day_map = {d: i for i, d in enumerate(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])}
                
                input_data = pd.DataFrame({
                    'promised_hour': [promised_hour],
                    'promised_day': [day_map[promised_day]],
                    'area_encoded': [le_area.transform([selected_area])[0]],
                    'order_total': [order_total]
                })
                
                if st.button("Calculate Risk"):
                    # Predict
                    prob = model.predict_proba(input_data)[0][1]
                    is_late = model.predict(input_data)[0]
                    
                    st.subheader("Risk Assessment")
                    
                    # Gauge / Metric
                    st.metric("Delay Probability", f"{prob:.1%}")
                    
                    if prob > 0.7:
                        st.error("High Risk of Delay 🚨")
                    elif prob > 0.4:
                        st.warning("Moderate Risk ⚠️")
                    else:
                        st.success("On-Time Likely ✅")

    # --- Tab 3: GenAI ---
    with tab3:
        st.header("Customer Feedback Intelligence")
        
        # Initialize Chatbot (Cached)
        try:
            from genai import RAGChatbot
            
            @st.cache_resource
            def get_chatbot():
                return RAGChatbot()
            
            bot = get_chatbot()
            
            # Chat Interface
            if "messages" not in st.session_state:
                st.session_state.messages = []

            # Display chat messages from history on app rerun
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            # React to user input
            if prompt := st.chat_input("Ask a question about customer feedback (e.g., 'Why are deliveries late?')"):
                # Display user message in chat message container
                st.chat_message("user").markdown(prompt)
                # Add user message to chat history
                st.session_state.messages.append({"role": "user", "content": prompt})

                with st.chat_message("assistant"):
                    with st.spinner("Analyzing feedback..."):
                        response = bot.ask(prompt)
                        st.markdown(response)
                
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response})
                
        except Exception as e:
            st.error(f"Failed to initialize GenAI module: {e}")
            st.info("Ensure genai.py and dependencies are correctly installed.")
