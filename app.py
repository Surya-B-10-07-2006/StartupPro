import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from google import genai as google_genai
from google.genai import types
import os
from dotenv import load_dotenv
import json
from datetime import datetime
import re 
import sys 

# --- 0. Setup and Initialization ---

# Page configuration - must be the first Streamlit command
st.set_page_config(
    page_title="AI Startup Success Predictor",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Gemini client
@st.cache_resource
def init_gemini_client():
    """Initialize Gemini client with error handling"""
    try:
        load_dotenv()
        api_key = os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY')
        
        if not api_key:
            st.error("❌ GEMINI_API_KEY not found. Please set it in your environment or a .env file.")
            return None
            
        client = google_genai.Client(api_key=api_key)
        
        if 'chat_session' not in st.session_state:
            config = types.GenerateContentConfig(
                system_instruction="You are 'StartupHelpBot', a friendly, concise, and expert AI assistant for startup founders. Your goal is to answer real-time doubts about the main startup analysis, not to generate a new analysis. Keep answers brief and actionable."
            )
            st.session_state.chat_session = client.chats.create(
                model="gemini-2.5-flash",
                config=config 
            )

        return client
        
    except Exception as e:
        st.error(f"❌ Failed to initialize Gemini Client: {str(e)}")
        return None

client = init_gemini_client()

# --- 1. Mock ML Prediction (Simplified) ---

def predict_success_probability(startup_description):
    """Predict success probability using only description length and base score."""
    score = 50
    desc_length = len(startup_description.split())
    if desc_length < 30:
        score += desc_length / 5
    else:
        score += min(desc_length / 10, 15)
    
    return max(15, min(85, score))

# --- 2. Advanced Trend Analysis Simulation (Market Trend) ---

def simulate_market_trend(success_prob):
    """
    Simulates a 5-year market trend based on the success probability,
    implying market resonance and growth potential.
    """
    end_date = datetime.now()
    start_date = end_date - pd.DateOffset(years=5)
    date_range = pd.date_range(start=start_date, end=end_date, freq='M')
    
    base_trend = np.linspace(50, 60, len(date_range))
    growth_factor = 0.5 + (success_prob / 100) * 1.5
    noise = np.random.normal(0, 5, len(date_range))
    trend_values = base_trend + (base_trend - 50) * growth_factor + noise
    last_val_target = 60 + success_prob / 5
    trend_values = trend_values - trend_values[0] + 40
    trend_values = pd.Series(trend_values).rolling(window=3, min_periods=1).mean().values
    
    df_trend = pd.DataFrame({
        'Date': date_range,
        'Market Interest Score': trend_values
    })
    
    return df_trend

# --- 3. Gemini Analysis and Parsing ---

def generate_analysis(client, prompt):
    """Generate analysis using Gemini API with robust JSON parsing"""
    try:
        if not client:
            return None
            
        tool_config = types.GenerateContentConfig(
            tools=[{"google_search": {}}]
        )
            
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=tool_config
        )
        
        if not response or not hasattr(response, 'text'):
            st.error("❌ Invalid or empty response from Gemini API.")
            return None

        raw_text = response.text
        text = raw_text.strip()
        
        match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
        if not match:
            match = re.search(r"```\s*(.*?)\s*```", text, re.DOTALL)

        if match:
            json_content = match.group(1).strip()
        else:
            start_index = text.find('{')
            end_index = text.rfind('}')
            
            if start_index != -1 and end_index != -1 and end_index > start_index:
                json_content = text[start_index : end_index + 1]
            else:
                st.error("❌ Could not find valid JSON delimiters in the response.")
                st.info("Raw response from AI (for debugging): \n" + raw_text)
                return None
        
        try:
            return json.loads(json_content)
        except json.JSONDecodeError as e:
            st.error(f"❌ Failed to parse JSON content: {str(e)}")
            st.info("Raw JSON Content attempting to parse:\n" + json_content)
            return None
        
    except Exception as e:
        st.error(f"❌ Error generating analysis: {str(e)}")
        return None

# --- 4. Report Generation (UPDATED for 10 Funding/Legal Points) ---

def generate_report_content(analysis_data, success_prob, startup_description):
    """Generate structured text content for the report (TXT/Markdown)."""
    try:
        report = f"# Startup Success Analysis Report\n"
        report += f"**Generated On:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        report += f"## 💡 Startup Idea\n{startup_description}\n\n"
        report += f"## 📊 Success Probability: {success_prob:.1f}%\n"
        report += "--- \n"
        
        # SWOT
        swot = analysis_data.get('swot_analysis', {})
        report += "## 🔍 SWOT Analysis\n"
        for category in ['strengths', 'weaknesses', 'opportunities', 'threats']:
            report += f"\n### {category.title()}\n"
            items = swot.get(category, [])
            for item in items:
                report += f"* {item}\n"
        
        # Roadmap
        report += "\n## 🗓️ 12-Week Roadmap\n"
        roadmap = analysis_data.get('roadmap', [])
        for item in roadmap:
            report += f"**Week {item.get('week', '?')}:** {item.get('milestone', 'No milestone')}\n"
            
        # Team Advice
        report += "\n## 👥 Team & Hiring Advice\n"
        for advice in analysis_data.get('team_advice', []):
             report += f"* {advice}\n"
            
        # Funding & Legal Advice - UPDATED
        funding_advice = analysis_data.get('funding_advice', {})
        report += "\n## 💰 Funding & Legal Advice\n"
        
        report += "\n### 💰 Funding Strategies\n"
        for strategy in funding_advice.get('funding_strategies', []):
             report += f"* {strategy}\n"
             
        report += "\n### ⚖️ Legal Considerations\n"
        for legal in funding_advice.get('legal_considerations', []):
             report += f"* {legal}\n"
        
        report += f"\n## 🔑 Key Insight\n"
        report += f"> {analysis_data.get('success_hint', 'N/A')}\n"

        return report.encode('utf-8')
    except Exception as e:
        st.error(f"❌ Error generating report content: {str(e)}")
        return None

# --- 5. Chatbot Function (Real-time Chat) ---

def startup_chatbot_ui():
    """Adds an interactive chatbot to the sidebar for real-time doubts."""
    st.sidebar.markdown("---")
    st.sidebar.header("💬 Real-Time Startup Advisor")
    st.sidebar.info("Ask follow-up questions about the analysis.")

    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
        
    for message in st.session_state.chat_messages:
        with st.sidebar.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.sidebar.chat_input("Ask a follow-up question..."):
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        with st.sidebar.chat_message("user"):
            st.markdown(prompt)

        with st.sidebar.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.chat_session.send_message(prompt)
                    full_response = response.text
                    st.markdown(full_response)
                    st.session_state.chat_messages.append({"role": "assistant", "content": full_response})
                except Exception as e:
                    st.error(f"Chatbot Error: {e}")


# --- 6. Main Streamlit Function ---

def main():
    """Main function to run the Streamlit app"""
    st.title("🚀 AI Startup Success Predictor")
    st.markdown("### Get instant, structured business insights and market trends for your startup idea.")

    if not client:
        return

    startup_chatbot_ui() 

    st.markdown("## 📝 Enter Your Startup Idea")
    
    startup_description = st.text_area(
        "Describe your startup idea in detail:",
        height=200,
        placeholder="Describe your startup idea, target market, unique value proposition, and target customers..."
    )
    
    if st.button("🔍 Analyze My Startup", type="primary", use_container_width=True):
        if not startup_description.strip():
            st.error("❌ Please enter a valid startup description.")
            return

        st.session_state.chat_messages = []
        
        system_prompt = "You are 'StartupHelpBot', a friendly, concise, and expert AI assistant for startup founders. The user just started a new analysis. All your subsequent answers must be brief and actionable, specifically relating to the content of the analysis. Do not generate a new full analysis."
        
        initial_message = f"The new startup idea being analyzed is: {startup_description}. The initial success probability is {predict_success_probability(startup_description):.1f}%."

        initial_history = [
            types.Content(
                role="user",
                parts=[types.Part(text=initial_message)] 
            )
        ]
        
        config = types.GenerateContentConfig(
            system_instruction=system_prompt
        )

        st.session_state.chat_session = client.chats.create(
            model="gemini-2.5-flash",
            history=initial_history,
            config=config
        )


        with st.spinner("🧠 Analyzing your startup idea and checking market trends..."):
            try:
                success_prob = predict_success_probability(startup_description)
                
                # --- PROMPT CONSTRUCTION (UPDATED FOR 10 ROLES AND 10 ADVICE POINTS) ---
                
                SYSTEM_INSTRUCTION = """
                You are 'StartupPro', an elite, data-driven startup success advisor. Use the Google Search tool to inform your analysis.
                YOUR OUTPUT MUST BE A SINGLE, VALID JSON OBJECT THAT STRICTLY ADHERES to the structure provided below.
                DO NOT include any introductory text, explanation, markdown formatting (like ```json), or commentary outside of the JSON structure.
                """

                # JSON TEMPLATE REFLECTING THE NEW REQUIREMENTS
                JSON_TEMPLATE = json.dumps({
                    "swot_analysis": {
                        "strengths": ["...", "...", "..."], 
                        "weaknesses": ["...", "...", "..."],
                        "opportunities": ["...", "...", "..."],
                        "threats": ["...", "..."]
                    },
                    "roadmap": [{"week": i+1, "milestone": f"Week {i+1} milestone..."} for i in range(12)],
                    "team_advice": [f"Role {i+1}: ..." for i in range(10)], # 10 roles requested
                    "funding_advice": {
                        "funding_strategies": ["Strategy 1", "Strategy 2", "Strategy 3", "Strategy 4", "Strategy 5"], # 5 funding points
                        "legal_considerations": ["Legal 1", "Legal 2", "Legal 3", "Legal 4", "Legal 5"] # 5 legal points
                    },
                    "success_hint": "..."
                }, indent=2)

                prompt = f"""{SYSTEM_INSTRUCTION}
                Analyze this startup idea and provide a comprehensive analysis. Use your search tool to find relevant market information, competitor data, and current trends to make the analysis realistic.
                
                Startup Idea: {startup_description}
                Success Probability (AI Model): {success_prob:.1f}%

                **Specific Requirements:**
                1. ROADMAP: Generate a **specific milestone for each of the 12 weeks**.
                2. TEAM ADVICE: Provide a list of exactly **10 essential and suitable roles (e.g., 'CTO', 'Lead UX Designer', 'CMO')** followed by detailed advice for each role.
                3. FUNDING/LEGAL: Generate exactly **5 distinct funding strategies** and exactly **5 critical legal considerations**. Ensure all 10 points are concise and high-value.

                Your response MUST be a valid JSON object matching this structure:
                {JSON_TEMPLATE}
                """
                
                analysis = generate_analysis(client, prompt)
                
                if not analysis:
                    return
                
                # --- Display results ---
                st.markdown("---")
                
                st.markdown("## 📊 Success Prediction & Market Analysis")
                col_metric, col_pie, col_line_chart = st.columns([1, 1, 3]) 
                
                with col_metric:
                    st.metric("Success Probability", f"{success_prob:.1f}%", help="Based on internal AI model and description detail.")

                # PIE CHART
                df_pie = pd.DataFrame({
                    'Category': ['Success Probability', 'Risk/Failure Probability'],
                    'Value': [success_prob, 100 - success_prob]
                })

                fig_pie = px.pie(
                    df_pie, 
                    values='Value', 
                    names='Category', 
                    title='Probability Breakdown',
                    hole=.5, 
                    color='Category',
                    color_discrete_map={'Success Probability':'green', 'Risk/Failure Probability':'red'}
                )
                fig_pie.update_traces(textinfo='percent', marker=dict(line=dict(color='#000000', width=1)))
                fig_pie.update_layout(showlegend=False, margin=dict(t=50, b=10, l=10, r=10)) 

                with col_pie:
                    st.plotly_chart(fig_pie, use_container_width=True)

                # LINE CHART
                with col_line_chart:
                    st.markdown("### 📈 Trajectory vs. Market Trend")
                    
                    df_trajectory = pd.DataFrame({
                        'Week': range(1, 13),
                        'Value': np.linspace(max(10, success_prob * 0.8), min(90, success_prob * 1.2), 12),
                        'Series': '12-Week Strategy Trajectory'
                    })

                    df_market = simulate_market_trend(success_prob)
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=df_trajectory['Week'], y=df_trajectory['Value'],
                        mode='lines+markers', name='Strategy Trajectory (12 Weeks)',
                        line=dict(color='green', width=3)
                    ))

                    df_market_last_12 = df_market.tail(12).reset_index(drop=True)
                    df_market_last_12['Week'] = df_market_last_12.index + 1
                    
                    fig.add_trace(go.Scatter(
                        x=df_market_last_12['Week'], y=df_market_last_12['Market Interest Score'],
                        mode='lines', name='Market Interest Trend (Last Year)',
                        line=dict(color='orange', dash='dash')
                    ))

                    fig.update_layout(
                        xaxis_title='Week of Execution', yaxis_title='Score (%)',
                        yaxis_range=[0, 100],
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                    )

                    st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("---")
                
                st.markdown("## 🔍 SWOT Analysis")
                swot = analysis.get('swot_analysis', {})
                col1, col2 = st.columns(2)
                
                with col1:
                    with st.expander("✅ Strengths", expanded=True):
                        for item in swot.get('strengths', []): st.markdown(f"- **{item}**")
                    with st.expander("❌ Weaknesses", expanded=True):
                        for item in swot.get('weaknesses', []): st.markdown(f"- **{item}**")
                with col2:
                    with st.expander("💡 Opportunities", expanded=True):
                        for item in swot.get('opportunities', []): st.markdown(f"- **{item}**")
                    with st.expander("⚠️ Threats", expanded=True):
                        for item in swot.get('threats', []): st.markdown(f"- **{item}**")
                
                st.markdown("## 🗓️ 12-Week Roadmap")
                roadmap = analysis.get('roadmap', [])
                if roadmap and isinstance(roadmap, list):
                    cleaned_roadmap = [{'Week': item.get('week', i+1), 'Milestone': item.get('milestone', 'N/A')} 
                                       for i, item in enumerate(roadmap)]
                    df_roadmap = pd.DataFrame(cleaned_roadmap)
                    df_roadmap.set_index('Week', inplace=True)
                    st.dataframe(df_roadmap, use_container_width=True)
                else:
                    st.warning("Roadmap data is missing or incomplete.")

                # UPDATED TEAM ADVICE SECTION
                st.markdown("## 👥 Team & Hiring Advice")
                for advice in analysis.get('team_advice', []): st.markdown(f"• **{advice}**")
                
                # UPDATED FUNDING & LEGAL SECTION
                st.markdown("## 💰 Funding & Legal Advice")
                funding_advice = analysis.get('funding_advice', {})
                
                st.markdown("### 💰 Funding Strategies")
                for strategy in funding_advice.get('funding_strategies', []): 
                    st.success(f"• {strategy}")
                
                st.markdown("### ⚖️ Legal Considerations")
                for legal in funding_advice.get('legal_considerations', []): 
                    st.warning(f"• {legal}")
                
                st.markdown("## 💡 Key Success Hint")
                st.info(analysis.get('success_hint', 'No specific advice available'))
                
                # 8. Download Report
                st.markdown("---")
                st.markdown("### 📥 Download Full Report")
                
                report_bytes = generate_report_content(analysis, success_prob, startup_description)
                
                if report_bytes:
                    st.download_button(
                        label="Download Full Report (TXT/Markdown)",
                        data=report_bytes,
                        file_name=f"startup_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain" 
                    )
                
            except Exception as e:
                st.error(f"❌ An unexpected error occurred during analysis: {str(e)}")

if __name__ == "__main__":
    main()