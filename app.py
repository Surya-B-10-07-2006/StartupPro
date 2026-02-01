import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from groq import Groq
import httpx
import os
from dotenv import load_dotenv
import json
from datetime import datetime
import re
import time

# --- 0. Setup and Initialization ---

# Page configuration - must be the first Streamlit command
st.set_page_config(
    page_title="AI Startup Success Predictor",
    page_icon="🚀",
    layout="wide"
)

# Initialize Groq client
@st.cache_resource
def init_groq_client():
    """Initialize Groq client with error handling"""
    try:
        load_dotenv()
        api_key = os.getenv('GROQ_API_KEY')
        
        if not api_key:
            st.error("❌ GROQ_API_KEY not found. Please set it in your environment or a .env file.")
            return None
            
        # Create HTTP client
        http_client = httpx.Client(
            timeout=300.0,
            headers={"User-Agent": "StartupPro/1.0"}
        )
        
        # Initialize Groq client with httpx client
        client = Groq(api_key=api_key, http_client=http_client)
        return client
        
    except Exception as e:
        st.error(f"❌ Failed to initialize Groq Client: {str(e)}")
        return None

client = init_groq_client()

# --- 1. Mock ML Prediction (Simplified) ---

def calculate_success_probability(startup_description):
    """Calculate success probability using the 5-factor formula"""
    # Analyze startup description for keywords to determine factors
    desc_lower = startup_description.lower()
    
    # Market Demand (0-100)
    market_keywords = ['demand', 'need', 'problem', 'solution', 'market', 'customers']
    market_demand = min(100, 40 + sum(10 for keyword in market_keywords if keyword in desc_lower))
    
    # Competition Level (0-100, lower is better, so we invert)
    competition_keywords = ['unique', 'innovative', 'first', 'new', 'different']
    competition_level = max(20, 80 - sum(10 for keyword in competition_keywords if keyword in desc_lower))
    
    # Startup Difficulty (0-100, lower is better, so we invert)
    difficulty_keywords = ['simple', 'easy', 'straightforward', 'basic']
    startup_difficulty = max(30, 70 - sum(8 for keyword in difficulty_keywords if keyword in desc_lower))
    
    # Profit Potential (0-100)
    profit_keywords = ['revenue', 'profit', 'monetize', 'subscription', 'sales', 'income']
    profit_potential = min(100, 30 + sum(12 for keyword in profit_keywords if keyword in desc_lower))
    
    # Regulatory Complexity (0-100, lower is better, so we invert)
    regulatory_keywords = ['fintech', 'healthcare', 'finance', 'medical', 'banking']
    regulatory_complexity = min(80, 20 + sum(15 for keyword in regulatory_keywords if keyword in desc_lower))
    
    # Apply formula
    success_score = (
        (market_demand * 0.20) +
        ((100 - competition_level) * 0.20) +  # Invert competition
        ((100 - startup_difficulty) * 0.20) +  # Invert difficulty
        (profit_potential * 0.20) +
        ((100 - regulatory_complexity) * 0.20)  # Invert regulatory
    )
    
    return max(15, min(85, success_score))

def generate_success_analysis(client, startup_description, success_prob):
    """Generate success probability analysis"""
    try:
        if not client:
            return get_mock_success_analysis(success_prob)
            
        prompt = f"""Give a success probability out of 100 for the startup idea "{startup_description}".
Also give a 1-line reason and a short 2-3 line explanation.
Output format:
score: <number>
reason: <short reason>
explanation: <short explanation>

Provide in this JSON format:
{{
  "score": {success_prob},
  "reason": "short reason",
  "explanation": "short 2-3 line explanation"
}}"""

        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=512
        )
        
        raw_text = completion.choices[0].message.content
        
        # Extract JSON
        match = re.search(r"```json\s*(.*?)\s*```", raw_text, re.DOTALL)
        if match:
            json_content = match.group(1).strip()
        else:
            start_index = raw_text.find('{')
            end_index = raw_text.rfind('}')
            if start_index != -1 and end_index != -1:
                json_content = raw_text[start_index:end_index + 1]
            else:
                return get_mock_success_analysis(success_prob)
        
        try:
            result = json.loads(json_content)
            return result
        except:
            return get_mock_success_analysis(success_prob)
            
    except Exception as e:
        return get_mock_success_analysis(success_prob)

def get_mock_success_analysis(success_prob):
    """Generate mock success analysis"""
    return {
        "score": int(success_prob),
        "reason": "Strong market potential with clear value proposition",
        "explanation": "The startup addresses a real market need with a viable solution. Success depends on execution, market timing, and competitive positioning."
    }

def generate_monthly_explanations(client, market_prediction, month_names):
    """Generate monthly trend explanations"""
    try:
        if not client:
            return get_mock_monthly_explanations(month_names)
            
        # Create trend data for prompt
        trends = []
        for i in range(1, len(market_prediction)):
            if market_prediction[i] > market_prediction[i-1]:
                trends.append(f"{month_names[i]}: increased")
            else:
                trends.append(f"{month_names[i]}: decreased")
        
        prompt = f"""For each month, give a simple, one-line explanation of why the value increased or decreased compared to the previous month.
Use simple English suitable for beginners.
Do NOT mention arrows or percentages.
Make explanations generic so they work for any type of startup.
Keep each explanation under 15 words.

Trends: {', '.join(trends[:5])}

Output format:
{month_names[1]}: <explanation>
{month_names[2]}: <explanation>
{month_names[3]}: <explanation>
{month_names[4]}: <explanation>
{month_names[5]}: <explanation>

Provide in this JSON format:
{{
  "{month_names[1]}": "explanation",
  "{month_names[2]}": "explanation",
  "{month_names[3]}": "explanation",
  "{month_names[4]}": "explanation",
  "{month_names[5]}": "explanation"
}}"""

        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=512
        )
        
        raw_text = completion.choices[0].message.content
        
        # Extract JSON
        match = re.search(r"```json\s*(.*?)\s*```", raw_text, re.DOTALL)
        if match:
            json_content = match.group(1).strip()
        else:
            start_index = raw_text.find('{')
            end_index = raw_text.rfind('}')
            if start_index != -1 and end_index != -1:
                json_content = raw_text[start_index:end_index + 1]
            else:
                return get_mock_monthly_explanations(month_names)
        
        try:
            result = json.loads(json_content)
            return result
        except:
            return get_mock_monthly_explanations(month_names)
            
    except Exception as e:
        return get_mock_monthly_explanations(month_names)

def get_mock_monthly_explanations(month_names):
    """Generate mock monthly explanations"""
    explanations = [
        "Market demand increased due to seasonal trends",
        "Customer interest grew through word of mouth",
        "Competition affected market share slightly",
        "New features attracted more users",
        "Marketing efforts showed positive results"
    ]
    
    return {month_names[i+1]: explanations[i] for i in range(min(5, len(month_names)-1))}

def get_top_competitors(startup_description):
    """Get top 5 competitors based on startup idea"""
    try:
        if not client:
            return get_mock_competitors(startup_description)
            
        prompt = f"""Analyze this startup idea and identify the top 5 real competitors in this market:

Startup Idea: {startup_description}

Provide exactly 5 competitor names in this JSON format:
{{
  "competitors": ["Competitor 1", "Competitor 2", "Competitor 3", "Competitor 4", "Competitor 5"]
}}

Focus on real, well-known companies in the same industry/market."""

        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=512
        )
        
        raw_text = completion.choices[0].message.content
        
        # Extract JSON
        match = re.search(r"```json\s*(.*?)\s*```", raw_text, re.DOTALL)
        if match:
            json_content = match.group(1).strip()
        else:
            start_index = raw_text.find('{')
            end_index = raw_text.rfind('}')
            if start_index != -1 and end_index != -1:
                json_content = raw_text[start_index:end_index + 1]
            else:
                return get_mock_competitors(startup_description)
        
        try:
            result = json.loads(json_content)
            return result.get('competitors', get_mock_competitors(startup_description))
        except:
            return get_mock_competitors(startup_description)
            
    except Exception as e:
        return get_mock_competitors(startup_description)

def generate_startup_names(client, startup_description):
    """Generate creative startup names"""
    try:
        if not client:
            return get_mock_startup_names()
            
        prompt = f"""Generate 5 creative and relevant business names for this startup idea:

Startup Idea: {startup_description}

Provide names in this JSON format:
{{
  "names": ["Name1", "Name2", "Name3", "Name4", "Name5"]
}}

Make names short, catchy, and relevant to the business."""

        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.8,
            max_tokens=512
        )
        
        raw_text = completion.choices[0].message.content
        
        # Extract JSON
        match = re.search(r"```json\s*(.*?)\s*```", raw_text, re.DOTALL)
        if match:
            json_content = match.group(1).strip()
        else:
            start_index = raw_text.find('{')
            end_index = raw_text.rfind('}')
            if start_index != -1 and end_index != -1:
                json_content = raw_text[start_index:end_index + 1]
            else:
                return get_mock_startup_names()
        
        try:
            result = json.loads(json_content)
            return result.get('names', get_mock_startup_names())
        except:
            return get_mock_startup_names()
            
    except Exception as e:
        return get_mock_startup_names()

def get_mock_startup_names():
    """Generate mock startup names"""
    return ['InnovatePro', 'StartupHub', 'VentureCore', 'LaunchPad', 'GrowthWave']

def generate_social_media_strategy(client, startup_description):
    """Generate social media content ideas"""
    try:
        if not client:
            return get_mock_social_strategy()
            
        prompt = f"""Give 5-6 very simple and attractive social media strategy tips for the startup idea "{startup_description}".
Rules:
- Use simple English.
- Keep each tip short (max 8-10 words).
- Make it practical and beginner-friendly.
- No complex marketing terms.

Provide in this JSON format:
{{
  "tips": ["tip1", "tip2", "tip3", "tip4", "tip5"]
}}"""

        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=1024
        )
        
        raw_text = completion.choices[0].message.content
        
        # Extract JSON
        match = re.search(r"```json\s*(.*?)\s*```", raw_text, re.DOTALL)
        if match:
            json_content = match.group(1).strip()
        else:
            start_index = raw_text.find('{')
            end_index = raw_text.rfind('}')
            if start_index != -1 and end_index != -1:
                json_content = raw_text[start_index:end_index + 1]
            else:
                return get_mock_social_strategy()
        
        try:
            result = json.loads(json_content)
            return result.get('tips', get_mock_social_strategy())
        except:
            return get_mock_social_strategy()
            
    except Exception as e:
        return get_mock_social_strategy()

def get_mock_social_strategy():
    """Generate mock social media strategy"""
    return [
        "Post daily updates about your product",
        "Share customer success stories",
        "Create simple how-to videos",
        "Ask questions to engage followers",
        "Show behind-the-scenes content"
    ]

def generate_competitor_tips(client, startup_description):
    """Generate competitor advantage tips"""
    try:
        if not client:
            return get_mock_competitor_tips()
            
        prompt = f"""Give 4 competitor advantage tips for the startup idea "{startup_description}".
Rules:
- Use simple English.
- Keep each tip short (max 12-15 words).
- Output as a list, not separate objects.

Provide in this JSON format:
{{
  "tips": ["tip1", "tip2", "tip3", "tip4"]
}}"""

        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=512
        )
        
        raw_text = completion.choices[0].message.content
        
        # Extract JSON
        match = re.search(r"```json\s*(.*?)\s*```", raw_text, re.DOTALL)
        if match:
            json_content = match.group(1).strip()
        else:
            start_index = raw_text.find('{')
            end_index = raw_text.rfind('}')
            if start_index != -1 and end_index != -1:
                json_content = raw_text[start_index:end_index + 1]
            else:
                return get_mock_competitor_tips()
        
        try:
            result = json.loads(json_content)
            return result.get('tips', get_mock_competitor_tips())
        except:
            return get_mock_competitor_tips()
            
    except Exception as e:
        return get_mock_competitor_tips()

def get_mock_competitor_tips():
    """Generate mock competitor tips"""
    return [
        "Focus on better customer service",
        "Offer unique features others don't have",
        "Price your product competitively",
        "Build strong brand identity"
    ]

def get_mock_competitors(startup_description):
    """Generate mock competitors based on keywords"""
    desc_lower = startup_description.lower()
    
    if any(word in desc_lower for word in ['ecommerce', 'online store', 'marketplace']):
        return ['Amazon', 'Flipkart', 'eBay', 'Shopify', 'Etsy']
    elif any(word in desc_lower for word in ['food', 'delivery', 'restaurant']):
        return ['Zomato', 'Swiggy', 'Uber Eats', 'DoorDash', 'Grubhub']
    elif any(word in desc_lower for word in ['ride', 'transport', 'taxi', 'cab']):
        return ['Uber', 'Ola', 'Lyft', 'Grab', 'DiDi']
    elif any(word in desc_lower for word in ['social', 'media', 'network']):
        return ['Facebook', 'Instagram', 'Twitter', 'LinkedIn', 'TikTok']
    elif any(word in desc_lower for word in ['fintech', 'payment', 'banking']):
        return ['PayPal', 'Stripe', 'Square', 'Razorpay', 'Paytm']
    else:
        return ['Google', 'Microsoft', 'Apple', 'Meta', 'Amazon']

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
    trend_values = trend_values - trend_values[0] + 40
    trend_values = pd.Series(trend_values).rolling(window=3, min_periods=1).mean().values
    
    df_trend = pd.DataFrame({
        'Date': date_range,
        'Market Interest Score': trend_values
    })
    
    return df_trend

def generate_legal_guidance(client, startup_description):
    """Generate industry-specific legal guidance based on startup idea"""
    try:
        if not client:
            return generate_mock_legal_guidance()
            
        prompt = f"""Give 5 simple legal considerations for this startup idea:

Startup Idea: {startup_description}

Provide exactly 5 brief guidance points in this JSON format:
{{
  "legal_guidance": [
    "Point 1: Brief guidance",
    "Point 2: Brief guidance",
    "Point 3: Brief guidance",
    "Point 4: Brief guidance",
    "Point 5: Brief guidance"
  ]
}}

Rules:
- Use very simple English
- No legal jargon
- Keep each point short and clear
- Make it suitable for beginners
- Each point under 12 words"""

        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=1024
        )
        
        raw_text = completion.choices[0].message.content
        
        # Extract JSON
        match = re.search(r"```json\s*(.*?)\s*```", raw_text, re.DOTALL)
        if match:
            json_content = match.group(1).strip()
        else:
            start_index = raw_text.find('{')
            end_index = raw_text.rfind('}')
            if start_index != -1 and end_index != -1:
                json_content = raw_text[start_index:end_index + 1]
            else:
                return generate_mock_legal_guidance()
        
        try:
            result = json.loads(json_content)
            return result.get('legal_guidance', generate_mock_legal_guidance())
        except:
            return generate_mock_legal_guidance()
            
    except Exception as e:
        return generate_mock_legal_guidance()

def generate_mock_legal_guidance():
    """Generate mock legal guidance when API fails"""
    return [
        "Register your business with the government.",
        "Get all required licenses for your startup.",
        "Check local rules and safety regulations.",
        "Create basic contracts for workers and partners.",
        "Protect your brand name or logo if needed."
    ]

# --- 3. Groq Analysis ---

def generate_analysis(client, prompt):
    """Generate analysis using Groq API with robust JSON parsing and rate limiting"""
    try:
        if not client:
            return None
            
        # Add rate limiting
        if 'last_api_call' in st.session_state:
            time_since_last = time.time() - st.session_state.last_api_call
            if time_since_last < 1:  # Wait 1 second between calls
                time.sleep(1 - time_since_last)
        
        st.session_state.last_api_call = time.time()
        
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.7,
            max_tokens=4096,
            top_p=1,
            stream=False,
            stop=None
        )
        
        raw_text = completion.choices[0].message.content
        text = raw_text.strip()
        
        # Try to extract JSON from response
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
                # If no JSON found, create a mock analysis
                return create_mock_analysis()
        
        try:
            return json.loads(json_content)
        except json.JSONDecodeError:
            # If JSON parsing fails, create mock analysis
            return create_mock_analysis()
        
    except Exception as e:
        error_msg = str(e)
        if "rate_limit" in error_msg.lower() or "429" in error_msg:
            st.error("🚫 **Rate Limit Exceeded!**")
            st.warning("**Solutions:**")
            st.info("1. Wait a few minutes and try again")
            st.info("2. Groq allows 14,400 requests per day")
        else:
            st.error(f"❌ Error generating analysis: {error_msg}")
        return create_mock_analysis()

def create_mock_analysis():
    """Create a mock analysis when API fails"""
    return {
        "swot_analysis": {
            "strengths": [
                "Innovative concept with market potential",
                "Clear value proposition identified",
                "Scalable business model structure",
                "Strong technical foundation",
                "Experienced founding team"
            ],
            "weaknesses": [
                "Limited market validation data",
                "Resource constraints for initial launch",
                "Competition from established players",
                "Lack of brand recognition",
                "Limited initial funding"
            ],
            "opportunities": [
                "Growing market demand in target sector",
                "Potential for strategic partnerships",
                "Technology advancement enabling growth",
                "Emerging market trends",
                "Government support initiatives"
            ],
            "threats": [
                "Market saturation risks",
                "Regulatory changes impact",
                "Economic downturn effects",
                "New competitor entry"
            ]
        },
        "roadmap": [
            {"week": 1, "milestone": "Market research, competitor analysis, customer interviews, MVP planning, team setup"},
            {"week": 2, "milestone": "Product development, prototype creation, user testing, feedback collection, iteration"},
            {"week": 3, "milestone": "Marketing strategy, brand development, partnership outreach, funding preparation"},
            {"week": 4, "milestone": "Beta launch, user acquisition, performance monitoring, scaling preparation, final adjustments"}
        ],
        "team_advice": [
            "CEO - Leads company vision and strategy",
            "CTO - Handles all technical development",
            "Marketing Manager - Grows customer base",
            "Sales Manager - Converts leads to customers",
            "Product Manager - Defines what to build",
            "Developer - Builds the actual product",
            "Designer - Makes product user-friendly",
            "Customer Support - Helps users with issues"
        ],
        "funding_advice": {
             "funding_strategies": [
                "Use your own savings first.",
                "Ask friends or family for small support.",
                "Take a small business loan from a bank.",
                "Apply for government schemes or support programs.",
                "Use crowdfunding to raise money from the public.",
                "Find an investor who can fund and guide your business."
            ],
            "legal_considerations": [
                "Register your business with the government.",
                "Get all required licenses for your startup.",
                "Check local rules and safety regulations.",
                "Create basic contracts for workers and partners.",
                "Protect your brand name or logo if needed."
            ]
        },
        "success_hint": "Focus on solving a real problem with a simple, scalable solution. Validate early and iterate based on user feedback."
    }

# --- 4. Enhanced Visual Components ---

def create_team_cards(team_advice):
    """Create attractive team cards with emojis"""
    role_emojis = {
        'CEO': '👑', 'CTO': '💻', 'CMO': '📈', 'Sales': '💼', 'Product': '🎯',
        'Developer': '⚡', 'Designer': '🎨', 'Data': '📊', 'Operations': '⚙️', 'Customer': '🤝'
    }
    
    st.markdown("## 👥 Team & Hiring Strategy")
    
    # Create 2 rows of 5 cards each
    for row in range(2):
        cols = st.columns(5)
        for col_idx in range(5):
            card_idx = row * 5 + col_idx
            if card_idx < len(team_advice):
                advice = team_advice[card_idx]
                role = advice.split(':')[0].strip()
                description = advice.split(':', 1)[1].strip() if ':' in advice else advice
                
                # Find matching emoji
                emoji = '👤'
                for key, value in role_emojis.items():
                    if key.lower() in role.lower():
                        emoji = value
                        break
                
                with cols[col_idx]:
                    st.markdown(f"""
                    <div style="
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        padding: 20px;
                        border-radius: 15px;
                        text-align: center;
                        color: white;
                        margin: 10px 0;
                        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
                        height: 180px;
                        display: flex;
                        flex-direction: column;
                        justify-content: center;
                    ">
                        <div style="font-size: 40px; margin-bottom: 10px;">{emoji}</div>
                        <div style="font-weight: bold; font-size: 16px; margin-bottom: 8px;">{role}</div>
                        <div style="font-size: 12px; line-height: 1.4;">{description}</div>
                    </div>
                    """, unsafe_allow_html=True)

def create_roadmap_infographic(roadmap):
    """Create attractive roadmap infographic"""
    st.markdown("## 🗓️ 12-Week Execution Roadmap")
    
    # Create timeline visualization
    weeks = [item.get('week', i+1) for i, item in enumerate(roadmap)]
    milestones = [item.get('milestone', 'No milestone') for item in roadmap]
    
    # Create Gantt-style chart
    fig = go.Figure()
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD'] * 2
    
    for i, (week, milestone) in enumerate(zip(weeks, milestones)):
        fig.add_trace(go.Scatter(
            x=[week, week+0.8],
            y=[i, i],
            mode='lines+markers',
            line=dict(color=colors[i % len(colors)], width=8),
            marker=dict(size=12, color=colors[i % len(colors)]),
            name=f"Week {week}",
            hovertemplate=f"<b>Week {week}</b><br>{milestone}<extra></extra>",
            showlegend=False
        ))
        
        # Add milestone text
        fig.add_annotation(
            x=week+1,
            y=i,
            text=f"<b>Week {week}:</b> {milestone[:40]}{'...' if len(milestone) > 40 else ''}",
            showarrow=False,
            xanchor="left",
            font=dict(size=11, color=colors[i % len(colors)])
        )
    
    fig.update_layout(
        title="📈 Startup Execution Timeline",
        xaxis_title="Weeks",
        yaxis_title="Milestones",
        height=600,
        showlegend=False,
        xaxis=dict(range=[0, 14]),
        yaxis=dict(showticklabels=False),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    st.plotly_chart(fig, use_container_width=True)

def create_swot_analysis(swot):
    """Create simple SWOT analysis with expandable sections"""
    st.markdown("## 🔍 SWOT Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        with st.expander("✅ Strengths", expanded=True):
            for item in swot.get('strengths', []): 
                st.markdown(f"- **{item}**")
        with st.expander("❌ Weaknesses", expanded=True):
            for item in swot.get('weaknesses', []): 
                st.markdown(f"- **{item}**")
    with col2:
        with st.expander("💡 Opportunities", expanded=True):
            for item in swot.get('opportunities', []): 
                st.markdown(f"- **{item}**")
        with st.expander("⚠️ Threats", expanded=True):
            for item in swot.get('threats', []): 
                st.markdown(f"- **{item}**")

def create_market_intelligence_charts(success_prob, competitors, startup_names, competitor_tips, success_analysis):
    """Create specific market intelligence visualizations"""
    st.markdown("## 📊 Real-Time Market Intelligence")
    
    # Success Analysis Display
    st.markdown("### 🎯 Success Probability Analysis")
    st.markdown(f"**Score:** {success_analysis['score']}/100")
    st.markdown(f"**Reason:** {success_analysis['reason']}")
    st.markdown(f"**Explanation:** {success_analysis['explanation']}")
    
    st.markdown("---")
    
    # Startup Naming Suggestions
    st.markdown("### 🏷️ Startup Naming Suggestions")
    cols = st.columns(len(startup_names))
    for i, name in enumerate(startup_names):
        with cols[i % len(cols)]:
            st.info(f"**{name}**")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Top 5 Competitors List
        st.markdown("### 🏆 Top 5 Competitors")
        for i, competitor in enumerate(competitors, 1):
            st.markdown(f"{i}. **{competitor}**")
        
        # Bar Chart - Multiple category ratings
        rating_categories = ['Product-Market Fit', 'Technical Feasibility', 'Market Timing', 'Competitive Advantage']
        ratings = [success_prob + np.random.normal(0, 8) for _ in rating_categories]
        ratings = [max(0, min(100, r)) for r in ratings]
        
        fig_bar = px.bar(
            x=rating_categories,
            y=ratings,
            title="📈 Category Ratings",
            color=ratings,
            color_continuous_scale='viridis'
        )
        fig_bar.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True)
    
    with col2:
        # Competitor Advantage Tips
        st.markdown("### 🎯 Competitor Advantage Tips")
        for i, tip in enumerate(competitor_tips, 1):
            st.success(f"{i}. {tip}")
        
        # Gauge Chart - Single prediction score
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = success_prob,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "🎯 Success Probability"},
            delta = {'reference': 50},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 25], 'color': "lightgray"},
                    {'range': [25, 50], 'color': "gray"},
                    {'range': [50, 75], 'color': "lightgreen"},
                    {'range': [75, 100], 'color': "green"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        fig_gauge.update_layout(height=300)
        st.plotly_chart(fig_gauge, use_container_width=True)
        
        # Line Chart - Market prediction over time with normalized values
        dates = pd.date_range(start=datetime.now(), periods=12, freq='M')
        # Normalize trend to match success probability
        base_value = success_prob
        trend_variation = 10  # Smaller variation range
        market_prediction = [base_value + np.random.normal(0, trend_variation) for _ in range(12)]
        market_prediction = [max(10, min(90, p)) for p in market_prediction]
        
        fig_line = px.line(
            x=dates,
            y=market_prediction,
            title="📈 Market Prediction Over Time",
            labels={'x': 'Time', 'y': 'Market Score (%)'}
        )
        fig_line.update_traces(line_color='#ff7f0e', line_width=3)
        fig_line.update_layout(height=300)
        st.plotly_chart(fig_line, use_container_width=True)
        
        # Monthly trend explanations
        month_names = [date.strftime('%B') for date in dates]
        explanations = generate_monthly_explanations(client, market_prediction, month_names)
        
        st.markdown("**Monthly Trends:**")
        for month, explanation in explanations.items():
            st.markdown(f"**{month}:** {explanation}")

def create_team_hiring_strategy():
    """Create task-based team hiring strategy"""
    st.markdown("## 👥 Team & Hiring Strategy")
    
    # Core hiring principles
    st.markdown("### 🎯 Core Hiring Principles")
    principles = [
        "Identify core tasks your startup must do daily",
        "Convert each core task into a role (tech, marketing, operations, etc.)",
        "Hire for skills, not titles — match skills to your startup needs",
        "Start small with essential roles; expand later",
        "Use freelancers for non-core or temporary work",
        "Prioritize multifunctional people in early stages",
        "Hire someone stronger than you in technical or specialized areas",
        "Check experience in similar industries to reduce training time",
        "Use structured interviews with task-based tests",
        "Build a culture-first team to ensure long-term fit"
    ]
    
    for i, principle in enumerate(principles, 1):
        st.markdown(f"{i}. **{principle}**")
    
    st.markdown("### 💼 Essential Startup Roles")
    
    # Essential roles with tasks
    roles_data = {
        'Role': ['Technical Lead', 'Marketing Lead', 'Operations Lead', 'Sales Lead', 'Product Lead'],
        'Core Tasks': [
            'Product development, technical architecture, code review',
            'Brand building, content creation, digital marketing',
            'Process optimization, vendor management, logistics',
            'Customer acquisition, relationship building, revenue generation',
            'User research, feature planning, roadmap management'
        ],
        'Hire When': [
            'Need technical expertise beyond founder skills',
            'Ready to scale customer acquisition',
            'Daily operations become complex',
            'Product-market fit achieved',
            'Multiple feature requests from users'
        ]
    }
    
    df_roles = pd.DataFrame(roles_data)
    st.dataframe(df_roles, use_container_width=True)

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

# --- 5. Main Streamlit Function ---

def main():
    """Main function to run the Streamlit app"""
    st.title("🚀 AI Startup Success Predictor")
    st.markdown("### Get instant, structured business insights and market trends for your startup idea.")

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

        with st.spinner("🧠 Analyzing your startup idea and checking market trends..."):
            try:
                success_prob = calculate_success_probability(startup_description)
                success_analysis = generate_success_analysis(client, startup_description, success_prob)
                
                # Get top competitors
                with st.spinner("🏆 Identifying top competitors..."):
                    competitors = get_top_competitors(startup_description)
                
                # Get startup names, social media strategy, and competitor tips
                with st.spinner("🏷️ Generating startup names and strategies..."):
                    startup_names = generate_startup_names(client, startup_description)
                    social_strategy = generate_social_media_strategy(client, startup_description)
                    competitor_tips = generate_competitor_tips(client, startup_description)
                
                # Generate industry-specific legal guidance
                with st.spinner("⚖️ Generating industry-specific legal guidance..."):
                    legal_guidance = generate_legal_guidance(client, startup_description)
                
                # --- PROMPT CONSTRUCTION ---
                
                prompt = f"""You are StartupPro, an elite business analyst. Analyze this startup idea and provide a comprehensive business analysis in valid JSON format.

Startup Idea: {startup_description}
Success Probability: {success_prob:.1f}%

Provide your analysis in this exact JSON structure:
{{
  "swot_analysis": {{
    "strengths": ["strength1", "strength2", "strength3", "strength4", "strength5"],
    "weaknesses": ["weakness1", "weakness2", "weakness3", "weakness4", "weakness5"],
    "opportunities": ["opportunity1", "opportunity2", "opportunity3", "opportunity4", "opportunity5"],
    "threats": ["threat1", "threat2", "threat3", "threat4", "threat5"]
  }},
  "roadmap": [
    {{"week": 1, "milestone": "Week 1: 3-5 action steps"}},
    {{"week": 2, "milestone": "Week 2: 3-5 action steps"}},
    {{"week": 3, "milestone": "Week 3: 3-5 action steps"}},
    {{"week": 4, "milestone": "Week 4: 3-5 action steps"}}
  ],
  "team_advice": [
    "CEO - Leads company vision and strategy",
    "CTO - Handles all technical development",
    "Marketing Manager - Grows customer base",
    "Sales Manager - Converts leads to customers",
    "Product Manager - Defines what to build",
    "Developer - Builds the actual product",
    "Designer - Makes product user-friendly",
    "Customer Support - Helps users with issues"
  ],
  "funding_advice": {{
    "funding_strategies": ["strategy1", "strategy2", "strategy3", "strategy4", "strategy5"],
    "legal_considerations": ["legal1", "legal2", "legal3", "legal4", "legal5"]
  }}
}}

For SWOT Analysis: EXACTLY 5 points in each category. Keep each point short (max 8-10 words). Use simple language.

For Team Advice: Give 8 essential roles with 1-line explanation each. Format: "Role - What they do". Keep simple.

For Roadmap: Create EXACTLY 4 weeks. Each week should have 3-5 clear, short action steps.

Respond ONLY with valid JSON. No additional text or explanation."""
                
                analysis = generate_analysis(client, prompt)
                
                if not analysis:
                    return
                
                # --- Display Enhanced Results ---
                st.markdown("---")
                
                # Market Intelligence Charts
                create_market_intelligence_charts(success_prob, competitors, startup_names, competitor_tips, success_analysis)
                
                st.markdown("---")
                
                # Simple SWOT Analysis
                create_swot_analysis(analysis.get('swot_analysis', {}))
                
                st.markdown("---")
                
                # Simple Roadmap Table
                st.markdown("## 🗓️ 4-Week Execution Roadmap")
                roadmap = analysis.get('roadmap', [])
                if roadmap and isinstance(roadmap, list):
                    cleaned_roadmap = [{'Week': item.get('week', i+1), 'Action Steps': item.get('milestone', 'N/A')} 
                                       for i, item in enumerate(roadmap)]
                    df_roadmap = pd.DataFrame(cleaned_roadmap)
                    df_roadmap.set_index('Week', inplace=True)
                    st.dataframe(df_roadmap, use_container_width=True)
                else:
                    st.warning("Roadmap data is missing or incomplete.")
                
                st.markdown("---")
                
                # Task-based Team Hiring Strategy
                st.markdown("## 👥 Team & Hiring Advice")
                for i, role in enumerate(analysis.get('team_advice', []), 1):
                    st.markdown(f"{i}. **{role}**")
                
                st.markdown("## 💰 Funding & Legal Advice")
                funding_advice = analysis.get('funding_advice', {})
                
                st.markdown("### 💰 Funding Strategies")
                for strategy in funding_advice.get('funding_strategies', []): 
                    st.success(f"• {strategy}")
                
                st.markdown("### ⚖️ Legal Considerations")
                st.info("📝 **Note:** These are general guidance points, not legal advice. Consult a lawyer for specific legal matters.")
                for i, legal in enumerate(legal_guidance, 1):
                    st.warning(f"{i}. {legal}")
                
                # Social Media Strategy
                st.markdown("### 📱 Social Media Strategy")
                
                for i, tip in enumerate(social_strategy, 1):
                    st.success(f"{i}. {tip}")
                
                # Download Report
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