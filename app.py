import streamlit as st
import pandas as pd
import plotly.express as px
from langchain_groq import ChatGroq
from langchain_experimental.agents import create_pandas_dataframe_agent
import time

# --- 1. UI SETUP & HEADER ---
st.set_page_config(page_title="AI Data Intelligence System", layout="wide")
st.title("📊 Unified Business Intelligence & Agentic AI")
st.markdown("Academic Project: Data Analytics + RAG AI + DevOps Deployment")

# --- 2. DATA ANALYTICS SECTION (Google Data Analytics Course) ---
st.header("Step 1: Data Visualization & Analytics")

# Simple dataset for a general business use case
data = {
    'Category': ['Electronics', 'Apparel', 'Home Decor', 'Beauty', 'Fitness'],
    'Stock_Units': [45, 120, 30, 85, 60],
    'Monthly_Revenue': [15000, 8000, 4500, 7200, 9500],
    'Customer_Rating': [4.5, 3.8, 4.2, 4.7, 4.0]
}
df = pd.DataFrame(data)

# Create two columns for the dashboard
col1, col2 = st.columns(2)

with col1:
    # Visualization logic
    fig = px.pie(df, values='Monthly_Revenue', names='Category', 
                 title="Revenue Distribution by Category",
                 color_discrete_sequence=px.colors.sequential.RdBu)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    # Table logic
    st.write("### Current Inventory & Sales Data")
    st.dataframe(df, use_container_width=True)

# --- 3. RAG & AGENTIC AI SECTION (IBM AI Course) ---
st.divider()
st.header("Step 2: Interactive AI Data Agent")

# Sidebar for the API Key
st.sidebar.title("Configuration")
api_key = st.sidebar.text_input("Enter Groq API Key:", type="password", help="Get your key at console.groq.com")

if api_key:
    try:
        # Initializing the LLM (Using the current 2026 stable model)
        llm = ChatGroq(
            temperature=0, 
            groq_api_key=api_key, 
            model_name="llama-3.1-8b-instant"
        )
        
        # Creating the Agentic AI to "talk" to the dataframe
        agent = create_pandas_dataframe_agent(
            llm, 
            df, 
            verbose=True, 
            allow_dangerous_code=True # Required for the agent to process python logic internally
        )
        
        user_query = st.text_input("Ask the AI Agent a question about the data above:")
        
        if user_query:
            with st.spinner("AI Agent is analyzing and calculating..."):
                # RATE LIMIT PROTECTION: Small pause to prevent 429 errors
                time.sleep(1.5) 
                
                response = agent.run(user_query)
                st.success("AI Analysis Complete:")
                st.write(response)
                
    except Exception as e:
        if "429" in str(e):
            st.error("Rate limit reached. Please wait 10 seconds and try again. (Common in Free Tier APIs)")
        else:
            st.error(f"An error occurred: {e}")
else:
    st.info("💡 Please enter your Groq API Key in the sidebar to interact with the AI Agent.")

# --- 4. DEVOPS INFO (DevOps Mastery Course) ---
st.sidebar.divider()
st.sidebar.markdown("""
### **DevOps Technical Specs:**
- **Container:** Docker (python:3.9-slim)
- **Deployment:** Streamlit Cloud / GitHub Actions
- **CI/CD:** Automated via GitHub Repo
""")
