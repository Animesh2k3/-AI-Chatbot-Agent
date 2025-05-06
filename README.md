
# 🤖 AI Chatbot Agent with Image Generation

A Streamlit-based AI assistant powered by Groq's LLM and Stability AI's image generation, featuring:
- Conversational AI with web search capability
- AI image generation from text prompts
- Customizable system prompts
- Conversation history management


## 🚀 Features

- **Chat Interface**:
  - Powered by Groq's Llama 3 70B model
  - Web search integration (Tavily API)
  - Customizable system prompts
  - Conversation history

- **Image Generation**:
  - Create images from text prompts
  - Powered by Stability AI's Stable Diffusion XL
  - Adjustable parameters (size, creativity, etc.)

- **User Experience**:
  - Streamlit web interface
  - Export conversations (JSON/Text/Markdown)
  - Save/Load chat sessions

## 🛠️ Installation

### Prerequisites
- Python 3.9+
- API keys for:
  - [Groq](https://console.groq.com/)
  - [Tavily](https://tavily.com/)
  - [Stability AI](https://platform.stability.ai/)

### Local Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/ai-chatbot-agent.git
   cd ai-chatbot-agent
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create `.streamlit/secrets.toml`:
   ```toml
   GROQ_API_KEY = "your_groq_api_key"
   TAVILY_API_KEY = "your_tavily_api_key"
   STABILITY_API_KEY = "your_stability_api_key"
   ```

4. Run the app:
   ```bash
   streamlit run app.py
   ```

## 🌐 Deployment

### Streamlit Community Cloud
1. Fork this repository
2. Go to [Streamlit Community Cloud](https://share.streamlit.io/)
3. Click "New app" and connect your repository
4. Set the main file path to `app.py`
5. Add your API keys in the "Secrets" section

### Alternative Deployment
For better performance, consider deploying the backend separately:
- Backend: Deploy `backend.py` on Render/Railway
- Frontend: Update API URLs in `app.py` and deploy to Streamlit

## 🧩 Project Structure

```
ai-chatbot-agent/
├── app.py               # Streamlit frontend
├── backend_functions.py # FastAPI backend logic
├── ai_agent.py          # Core AI functionality
├── requirements.txt     # Python dependencies
├── .streamlit/
│   └── secrets.toml    # API keys configuration
└── README.md           # This file
```

## 🤝 Contributing

Contributions are welcome! Please open an issue or submit a PR for any:
- Bug fixes
- New features
- Documentation improvements

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Groq for the ultra-fast LLM API
- Stability AI for image generation
- LangChain/LangGraph for agent orchestration
- Streamlit for the web interface
```



