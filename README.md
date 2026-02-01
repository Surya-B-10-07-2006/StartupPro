# 🚀 AI Startup Success Predictor

An intelligent web application that analyzes startup ideas and provides comprehensive business insights using AI-powered analysis.

## ✨ Features

- **AI-Powered Analysis**: Uses Groq API with Llama 3.1 for intelligent startup evaluation
- **Success Probability Prediction**: ML-based scoring system with detailed reasoning
- **SWOT Analysis**: Concise strengths, weaknesses, opportunities, and threats assessment
- **4-Week Roadmap**: Practical milestone planning for startup execution
- **Team & Hiring Advice**: 8 essential roles with clear explanations
- **Funding & Legal Guidance**: 5 funding strategies and 5 legal considerations
- **Market Intelligence**: Real-time competitor analysis and naming suggestions
- **Social Media Strategy**: 5 beginner-friendly marketing tips
- **Monthly Trend Analysis**: Market prediction with explanations
- **Downloadable Reports**: Export analysis as text/markdown files

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd StartupPro
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   Create a `.env` file in the project root:
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   ```
   Get your API key from [Groq Console](https://console.groq.com/)

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

## 🚀 Usage

1. Open the application in your browser (typically `http://localhost:8501`)
2. Enter your startup idea in the text area
3. Click "🔍 Analyze My Startup" to generate comprehensive analysis
4. Review the results including:
   - Success probability with reasoning
   - Market intelligence and competitor analysis
   - SWOT analysis with concise points
   - 4-week execution roadmap
   - Team building recommendations with role explanations
   - Funding and legal advice
   - Social media strategy tips
   - Monthly market trend predictions
5. Download the full report for offline reference

## 📋 Requirements

- Python 3.10+
- Streamlit 1.29.0
- Groq API access
- Internet connection for AI analysis

## 🔧 Configuration

The application uses the following environment variables:
- `GROQ_API_KEY`: Your Groq API key (required)

## 📊 Output Components

- **Success Analysis**: Probability scoring with reason and explanation
- **Market Intelligence**: Competitor analysis, startup naming suggestions, advantage tips
- **Visual Analytics**: Gauge charts, bar charts, and trend line graphs
- **Strategic Planning**: Week-by-week milestone roadmap
- **Business Intelligence**: SWOT analysis with concise points
- **Team Guidance**: 8 essential roles with explanations
- **Social Media Tips**: 5 beginner-friendly marketing strategies
- **Monthly Trends**: Market prediction explanations

## 🤖 AI Features

- **Groq Integration**: Fast Llama 3.1 model for natural language processing
- **Market Research**: AI-powered competitor and trend analysis
- **Structured Analysis**: JSON-formatted comprehensive reports
- **Beginner-Friendly**: Simple language suitable for new entrepreneurs

## 📁 Project Structure

```
StartupPro/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies with versions
├── .env               # Environment variables (create this)
├── .gitignore         # Git ignore rules
├── README.md          # Project documentation
└── runtime.txt        # Python version specification
```

## 🔒 Security

- API keys are stored in environment variables
- Sensitive files are properly gitignored
- No hardcoded credentials in source code

## 🐛 Troubleshooting

**API Key Issues:**
- Ensure your Groq API key is valid and active
- Check the `.env` file is in the project root
- Verify the environment variable name matches `GROQ_API_KEY`

**Installation Problems:**
- Use Python 3.10+ for compatibility
- Install dependencies in a virtual environment
- Check internet connection for package downloads

**Runtime Errors:**
- Restart the Streamlit server if issues persist
- Clear browser cache and refresh the page
- Check console logs for detailed error messages

## 📄 License

This project is for educational and demonstration purposes.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

---

**Built with ❤️ using Streamlit and Groq AI**