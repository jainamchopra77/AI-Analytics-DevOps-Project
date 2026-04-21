import streamlit as st
import time
import pandas as pd
import plotly.express as px
from langchain_groq import ChatGroq
from langchain_experimental.agents import create_pandas_dataframe_agent

# UI Setup
st.set_page_config(page_title="AI Data Intelligence", layout="wide")
st.title("📊 Unified Business Intelligence & Agentic AI")

# 1. DATA ANALYTICS SECTION
st.header("Step 1: Data Analytics")
# Generic business data
data = {
    'Category': ['Electronics', 'Apparel', 'Home Decor', 'Beauty', 'Fitness'],
    'Stock_Units': [45, 120, 30, 85, 60],
    'Monthly_Revenue': [15000, 8000, 4500, 7200, 9500],
    'Customer_Rating': [4.5, 3.8, 4.2, 4.7, 4.0]
}
df = pd.DataFrame(data)

col1, col2 = st.columns(2)
with col1:
    fig = px.pie(df, values='Monthly_Revenue', names='Category', title="Revenue Distribution")
    st.plotly_chart(fig)
with col2:
    st.write("### Raw Business Dataset")
    st.dataframe(df)

# 2. AGENTIC AI SECTION
st.divider()
st.header("Step 2: RAG & Agentic AI")
api_key = st.sidebar.text_input("Enter Groq API Key:", type="password")
if user_query:
    with st.spinner("AI Agent is analyzing data..."):
        time.sleep(1) # Add a 1-second pause to respect rate limits
        response = agent.run(user_query)
        st.success(f"AI Response: {response}")

if api_key:
    try:
        llm = ChatGroq(temperature=0, groq_api_key=api_key, model_name="llama-3.3-70b-versatile")
        agent = create_pandas_dataframe_agent(llm, df, verbose=True, allow_dangerous_code=True)
        
        user_query = st.text_input("Ask the AI Agent about your business data:")
        if user_query:
            with st.spinner("AI Agent is analyzing data..."):
                response = agent.run(user_query)
                st.success(f"AI Response: {response}")
    except Exception as e:
        st.error(f"Error: {e}")
else:
    st.info("Please enter your API key in the sidebar to enable the AI Agent.")
