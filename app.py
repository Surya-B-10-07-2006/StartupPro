import streamlit as st
import plotly.express as px
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

# Page configuration - must be the first Streamlit command
st.set_page_config(
    page_title="Startup Success Predictor",
    page_icon="🚀",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# --- 1. Client Initialization ---

def init_gemini_client():
    """Initialize Gemini client with error handling"""
    try:
        load_dotenv()
        api_key = os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY')
        
        if not api_key:
            st.error("❌ GEMINI_API_KEY not found. Please set it in your environment or a .env file.")
            return None
            
        client = google_genai.Client(api_key=api_key)
        return client
        
    except Exception as e:
        st.error(f"❌ Failed to initialize Gemini Client: {str(e)}")
        return None

# --- 2. Mock ML Prediction (Simplified) ---

def predict_success_probability(startup_description):
    """Predict success probability using only description length and base score."""
    score = 50
    desc_length = len(startup_description.split())
    score += min(desc_length / 10, 15)
    
    return max(5, min(95, score))

# --- 3. Gemini Analysis and Robust Parsing (omitted for brevity, assume correct) ---

def generate_analysis(client, prompt):
    """Generate analysis using Gemini API with robust JSON parsing"""
    try:
        if not client:
            return None
            
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
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
                return None
        
        try:
            return json.loads(json_content)
        except json.JSONDecodeError as e:
            st.error(f"❌ Failed to parse JSON content: {str(e)}")
            return None
        
    except Exception as e:
        st.error(f"❌ Error generating analysis: {str(e)}")
        return None

# --- 4. Report Generation (omitted for brevity, assume correct) ---

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
            
        # Funding & Legal Advice
        funding_advice = analysis_data.get('funding_advice', {})
        report += "\n## 💰 Funding & Legal Advice\n"
        report += f"**Funding Strategy:** {funding_advice.get('strategy', 'N/A')}\n"
        report += f"**Legal Considerations:** {funding_advice.get('legal', 'N/A')}\n"
        
        report += f"\n## 🔑 Key Insight\n"
        report += f"> {analysis_data.get('success_hint', 'N/A')}\n"

        return report.encode('utf-8')
    except Exception as e:
        st.error(f"❌ Error generating report content: {str(e)}")
        return None

# --- 5. Main Streamlit Function ---

def main():
    """Main function to run the Streamlit app"""
    st.title("🚀 Startup Success Predictor")
    st.markdown("### Get instant business insights for your startup idea")

    client = init_gemini_client()
    if not client:
        return

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

        with st.spinner("🧠 Analyzing your startup idea..."):
            try:
                success_prob = predict_success_probability(startup_description)
                
                # --- PROMPT CONSTRUCTION (omitted for brevity, assume correct) ---
                
                SYSTEM_INSTRUCTION = """
                You are 'StartupPro', an elite, data-driven startup success advisor.
                YOUR OUTPUT MUST BE A SINGLE, VALID JSON OBJECT THAT STRICTLY ADHERES to the structure provided below.
                DO NOT include any introductory text, explanation, markdown formatting (like ```json), or commentary outside of the JSON structure.
                """

                JSON_TEMPLATE = json.dumps({
                    "swot_analysis": {
                        "strengths": ["...", "...", "..."], 
                        "weaknesses": ["...", "...", "..."],
                        "opportunities": ["...", "...", "..."],
                        "threats": ["...", "..."]
                    },
                    "roadmap": [{"week": i+1, "milestone": f"Week {i+1} milestone..."} for i in range(12)],
                    "team_advice": ["CEO: ...", "CTO: ..."],
                    "funding_advice": {"strategy": "...", "legal": "..."},
                    "success_hint": "..."
                }, indent=2)

                prompt = f"""{SYSTEM_INSTRUCTION}
                Analyze this startup idea and provide a comprehensive analysis.
                
                Startup Idea: {startup_description}
                Success Probability: {success_prob:.1f}%

                **Specific Requirements:**
                1. ROADMAP: Generate a **specific milestone for each of the 12 weeks** (1 to 12).
                2. TEAM ADVICE: For each item, clearly list a **suitable role type (e.g., 'CTO', 'Lead UX Designer', 'CMO')** followed by detailed advice.
                3. FUNDING/LEGAL: Provide concise, high-value advice on **one primary funding strategy, including a perfectly estimated funding amount (e.g., "$50,000 for 6 months runway")** and **two critical legal tasks**.

                Your response MUST be a valid JSON object matching this structure:
                {JSON_TEMPLATE}
                """
                
                analysis = generate_analysis(client, prompt)
                
                if not analysis:
                    return
                
                # --- Display results ---
                st.markdown("---")
                
                st.markdown("## 📊 Success Prediction")
                st.metric("Success Probability", f"{success_prob:.1f}%")
                
                # Success Trajectory (Real-time Line Chart)
                st.markdown("## 📈 Success Trajectory (Assuming Strategy Execution)")
                weeks = 12
                trajectory = np.linspace(
                    max(10, success_prob * 0.8), 
                    min(90, success_prob * 1.2), 
                    weeks
                )
                df = pd.DataFrame({
                    'Week': range(1, weeks + 1),
                    'Success Probability': trajectory
                })
                
                # CRITICAL FIX: Add render_mode="svg" to force rendering compatibility
                fig = px.line(df, x='Week', y='Success Probability', 
                            title='12-Week Success Trajectory Based on Strategy',
                            markers=True,
                            render_mode="svg") # <-- FIX APPLIED HERE
                fig.update_yaxes(range=[0, 100], title='Probability (%)')
                st.plotly_chart(fig, use_container_width=True)
                
                # --- Remaining display logic (SWOT, Roadmap, etc.) is the same ---
                
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
                    cleaned_roadmap = [{'week': item.get('week', i+1), 'milestone': item.get('milestone', 'N/A')} 
                                       for i, item in enumerate(roadmap)]
                    df_roadmap = pd.DataFrame(cleaned_roadmap)
                    st.dataframe(df_roadmap.set_index('week'), use_container_width=True)
                else:
                    st.warning("Roadmap data is missing or incomplete.")

                st.markdown("## 👥 Team & Hiring Advice (Focus on Roles)")
                for advice in analysis.get('team_advice', []): st.markdown(f"• **{advice}**")
                
                st.markdown("## 💰 Funding & Legal Advice")
                funding_advice = analysis.get('funding_advice', {})
                st.markdown("### Funding Strategy")
                st.success(funding_advice.get('strategy', 'Not available'))
                st.markdown("### Legal Considerations")
                st.warning(funding_advice.get('legal', 'Not available'))
                
                st.markdown("## 💡 Success Hint")
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