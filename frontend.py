# if you dont use pipenv uncomment the following:
# from dotenv import load_dotenv
# load_dotenv()

import streamlit as st
import requests
import time
import json
import base64
from datetime import datetime
from typing import Dict, List
from io import BytesIO
from PIL import Image

# Page configuration
st.set_page_config(
    page_title="LangGraph AI Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
    <style>
        .stTextArea [data-baseweb=textarea] {
            background-color: #f8f9fa;
            border-radius: 8px;
        }
        .stButton button {
            width: 100%;
            background-color: #4CAF50;
            color: white;
            border-radius: 8px;
            padding: 0.5rem;
            font-weight: 500;
            transition: all 0.3s ease;
        }
        .stButton button:hover {
            background-color: #45a049;
            transform: scale(1.02);
        }
        [data-testid="stHorizontalBlock"] {
            gap: 1.5rem;
        }
        .stChatMessage {
            padding: 1rem;
            border-radius: 12px;
            margin-bottom: 1rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .st-emotion-cache-1q7spjk {
            width: 100%;
        }
        .stSpinner > div {
            margin: 0 auto;
        }
        .system-prompt-container {
            background-color: #f0f2f6;
            border-radius: 12px;
            padding: 1rem;
            margin: 1rem 0;
        }
        .preset-prompt {
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 0.5rem;
            margin: 0.25rem 0;
            cursor: pointer;
            transition: all 0.2s;
        }
        .preset-prompt:hover {
            background-color: #f0f0f0;
            transform: translateX(5px);
        }
        .image-generation-container {
            border: 1px solid #eee;
            border-radius: 12px;
            padding: 1rem;
            margin: 1rem 0;
        }
        .generated-image {
            border-radius: 8px;
            margin: 0.5rem 0;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        .sidebar-section {
            margin-bottom: 1.5rem;
        }
        .tab-button {
            border: none;
            background: none;
            padding: 0.5rem 1rem;
            cursor: pointer;
            font-size: 0.9rem;
            transition: all 0.2s;
        }
        .tab-button.active {
            border-bottom: 2px solid #4CAF50;
            font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)

# Constants
MODEL_NAMES_GROQ = ["llama-3.3-70b-versatile"]
API_URL = "https://ai-agent-backend-c1zz.onrender.com/chat"  # For Render
IMAGE_API_URL = "https://ai-agent-backend-c1zz.onrender.com/generate-image"

# Preset system prompts
PRESET_PROMPTS = {
    "Default Assistant": "Act as an AI chatbot who is smart and friendly",
    "Technical Expert": "You are a technical expert with deep knowledge in computer science, programming, and AI. Provide detailed, accurate answers.",
    "Creative Writer": "You are a creative writer with a flair for storytelling. Respond with imaginative and engaging content.",
    "Business Consultant": "You are a business consultant providing professional, structured advice on business strategies and operations.",
    "Language Tutor": "You are a language tutor helping students learn new languages. Provide clear explanations and examples.",
}

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "model_config" not in st.session_state:
    st.session_state.model_config = {
        "provider": "Groq",
        "model": MODEL_NAMES_GROQ[0],
        "system_prompt": PRESET_PROMPTS["Default Assistant"],
        "allow_search": False,
        "conversation_name": f"Conversation {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    }
if "conversations" not in st.session_state:
    st.session_state.conversations = {}
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "chat"
if "image_prompt" not in st.session_state:
    st.session_state.image_prompt = ""
if "negative_prompt" not in st.session_state:
    st.session_state.negative_prompt = ""

# Sidebar for settings
with st.sidebar:
    st.title("⚙️ Settings")
    
    # Conversation management
    with st.container():
        st.markdown("### Conversation")
        st.session_state.model_config["conversation_name"] = st.text_input(
            "Conversation Name",
            value=st.session_state.model_config["conversation_name"]
        )
    
    # Model configuration
    with st.container():
        st.markdown("### Model Configuration")
        provider = st.radio(
            "AI Provider:",
            ("Groq"),
            index=0 if st.session_state.model_config["provider"] == "Groq" else 1,
            key="provider"
        )
        
        if provider == "Groq":
            selected_model = st.selectbox(
                "Groq Model:",
                MODEL_NAMES_GROQ,
                index=MODEL_NAMES_GROQ.index(st.session_state.model_config["model"]) if st.session_state.model_config["model"] in MODEL_NAMES_GROQ else 0,
                key="groq_model"
            )
        
        
        allow_web_search = st.toggle(
            "Enable Web Search",
            value=st.session_state.model_config["allow_search"],
            help="Allow the agent to search the web for answers",
            key="web_search"
        )
    
    # Advanced options
    with st.container():
        st.markdown("### Advanced Options")
        temperature = st.slider(
            "Creativity (Temperature)",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.1,
            help="Higher values make outputs more random"
        )
        
        max_tokens = st.slider(
            "Max Response Length",
            min_value=100,
            max_value=2000,
            value=500,
            step=100,
            help="Maximum number of tokens in the response"
        )
    
    # Export conversation option
    with st.container():
        st.markdown("### 📤 Export Conversation")
        if st.session_state.messages:
            export_format = st.radio(
                "Export format",
                ("JSON", "Text", "Markdown"),
                horizontal=False
            )
            
            if export_format == "JSON":
                export_data = {
                    "metadata": st.session_state.model_config,
                    "messages": [
                        {k: v for k, v in msg.items() if k != "image_data"} 
                        for msg in st.session_state.messages
                    ]
                }
                st.download_button(
                    label="Download JSON",
                    data=json.dumps(export_data, indent=2),
                    file_name=f"{st.session_state.model_config['conversation_name']}.json",
                    mime="application/json"
                )
            elif export_format == "Text":
                text_content = "\n".join(
                    f"{msg['role'].capitalize()}: {msg['content']}" 
                    for msg in st.session_state.messages
                )
                st.download_button(
                    label="Download Text",
                    data=text_content,
                    file_name=f"{st.session_state.model_config['conversation_name']}.txt",
                    mime="text/plain"
                )
            elif export_format == "Markdown":
                md_content = "\n".join(
                    f"**{msg['role'].capitalize()}**: {msg['content']}\n" + 
                    (f"![Generated Image](data:image/png;base64,{base64.b64encode(msg['image_data'].getvalue()).decode() if 'image_data' in msg and hasattr(msg['image_data'], 'getvalue') else ''})\n" 
                     if "image_data" in msg else "\n")
                    for msg in st.session_state.messages
                )
                st.download_button(
                    label="Download Markdown",
                    data=md_content,
                    file_name=f"{st.session_state.model_config['conversation_name']}.md",
                    mime="text/markdown"
                )
        else:
            st.info("No conversation to export")
    
    # Conversation actions
    with st.container():
        st.markdown("### Actions")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("💾 Save Chat", use_container_width=True):
                st.session_state.conversations[st.session_state.model_config["conversation_name"]] = {
                    "messages": st.session_state.messages.copy(),
                    "config": st.session_state.model_config.copy()
                }
                st.toast("Conversation saved!", icon="✅")
        
        with col2:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.messages = []
                st.rerun()
        
        # Load conversation dropdown
        if st.session_state.conversations:
            st.markdown("### Saved Conversations")
            selected_conv = st.selectbox(
                "Select conversation to load",
                options=list(st.session_state.conversations.keys()),
                index=0,
                label_visibility="collapsed"
            )
            
            if st.button("Load Selected", use_container_width=True):
                conv_data = st.session_state.conversations[selected_conv]
                st.session_state.messages = conv_data["messages"].copy()
                st.session_state.model_config = conv_data["config"].copy()
                st.rerun()
    
    st.markdown("---")
    st.markdown("ℹ️ **About**")
    st.caption("This AI chatbot uses advanced language models to answer your questions and generate images.")

# Main chat interface
st.title("🤖 AI Chatbot Agents")
st.caption("Create and interact with AI agents powered by LangGraph")

# Tab selection
col1, col2 = st.columns(2)
with col1:
    if st.button("💬 Chat", key="chat_tab", use_container_width=True):
        st.session_state.active_tab = "chat"
with col2:
    if st.button("🎨 Generate Image", key="image_tab", use_container_width=True):
        st.session_state.active_tab = "image"

# Chat tab content
if st.session_state.active_tab == "chat":
    # Display chat messages
    for message in st.session_state.messages:
        avatar = "🧑" if message["role"] == "user" else "🤖"
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])
            if "timestamp" in message:
                st.caption(message["timestamp"])
            if "image_data" in message:
                st.image(message["image_data"], caption=message.get("image_caption", "Generated image"))
            elif "image_url" in message:
                st.image(message["image_url"], caption=message.get("image_caption", "Generated image"))

    # System Prompt
    with st.container():
        st.markdown("### Agent Configuration")
        
        # Preset prompts quick selection
        st.markdown("**Quick Presets:**")
        cols = st.columns(3)
        for i, (name, prompt) in enumerate(PRESET_PROMPTS.items()):
            with cols[i % 3]:
                if st.button(name, use_container_width=True):
                    st.session_state.model_config["system_prompt"] = prompt
                    st.rerun()
        
        # System prompt editor
        system_prompt = st.text_area(
            "System Prompt:",
            value=st.session_state.model_config["system_prompt"],
            height=150,
            placeholder="Define how your AI should behave...",
            help="This guides how the AI responds to you",
            key="sys_prompt"
        )

    # Chat input
    query = st.chat_input("Ask me anything!")
    if query:
        # Update model config in session state
        st.session_state.model_config = {
            "provider": provider,
            "model": selected_model,
            "system_prompt": system_prompt,
            "allow_search": allow_web_search,
            "conversation_name": st.session_state.model_config["conversation_name"]
        }
        
        # Add user message to chat history with timestamp
        timestamp = datetime.now().strftime("%H:%M")
        st.session_state.messages.append({
            "role": "user", 
            "content": query,
            "timestamp": timestamp
        })
        
        # Display user message
        with st.chat_message("user", avatar="🧑"):
            st.markdown(query)
            st.caption(timestamp)
        
        # Display assistant response
        with st.chat_message("assistant", avatar="🤖"):
            message_placeholder = st.empty()
            full_response = ""
            
            with st.spinner("Thinking..."):
                try:
                    start_time = time.time()
                    response = requests.post(
                        API_URL,
                        json={
                            "model_name": selected_model,
                            "model_provider": provider,
                            "system_prompt": system_prompt,
                            "messages": [msg["content"] for msg in st.session_state.messages if msg["role"] == "user"],
                            "allow_search": allow_web_search,
                            "temperature": temperature,
                            "max_tokens": max_tokens
                        },
                        timeout=30
                    )
                    
                    response.raise_for_status()
                    response_time = time.time() - start_time
                    
                    if response.status_code == 200:
                        response_data = response.json()
                        if isinstance(response_data, str):
                            full_response = response_data
                        elif "error" in response_data:
                            full_response = f"❌ Error: {response_data['error']}"
                        else:
                            full_response = response_data
                        
                        # Simulate streaming response
                        for chunk in full_response.split():
                            full_response += chunk + " "
                            time.sleep(0.05)
                            message_placeholder.markdown(full_response + "▌")
                        message_placeholder.markdown(full_response)
                        
                        # Add assistant message with timestamp and metrics
                        timestamp = datetime.now().strftime("%H:%M")
                        st.caption(f"{timestamp} • {len(full_response.split())} words • {response_time:.2f}s")
                        
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": full_response,
                            "timestamp": timestamp,
                            "metrics": {
                                "response_time": response_time,
                                "word_count": len(full_response.split())
                            }
                        })
                        
                        st.toast("Response received!", icon="✅")
                
                except requests.exceptions.RequestException as e:
                    error_msg = f"Failed to connect to the API: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": error_msg,
                        "timestamp": datetime.now().strftime("%H:%M")
                    })
                    st.info("Please ensure the backend server is running.")
                except Exception as e:
                    error_msg = f"An unexpected error occurred: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": error_msg,
                        "timestamp": datetime.now().strftime("%H:%M")
                    })

# Image generation tab content
elif st.session_state.active_tab == "image":
    with st.container():
        st.markdown("### Image Generation Settings")
        
        # Image prompt input
        st.session_state.image_prompt = st.text_area(
            "Describe the image you want to generate:",
            value=st.session_state.image_prompt,
            height=150,
            placeholder="A beautiful sunset over mountains...",
            key="image_prompt_input"
        )
        
        st.session_state.negative_prompt = st.text_area(
            "What to exclude from the image (optional):",
            value=st.session_state.negative_prompt,
            height=100,
            placeholder="blurry, low quality, text...",
            key="negative_prompt_input"
        )
        
        # Image settings
        st.markdown("### Image Settings")
        col1, col2, col3 = st.columns(3)
        with col1:
            width = st.selectbox("Width", [512, 768, 1024], index=2)
        with col2:
            height = st.selectbox("Height", [512, 768, 1024], index=2)
        with col3:
            samples = st.selectbox("Number of Images", [1, 2, 3, 4], index=0)
        
        col1, col2 = st.columns(2)
        with col1:
            cfg_scale = st.slider("Creativity (CFG Scale)", 1.0, 20.0, 7.0, 0.5)
        with col2:
            steps = st.slider("Generation Steps", 10, 150, 30, 5)
        
        # Generate button
        if st.button("Generate Image", key="generate_image_main", use_container_width=True):
            if st.session_state.image_prompt.strip():
                with st.spinner("Generating image..."):
                    try:
                        response = requests.post(
                            IMAGE_API_URL,
                            json={
                                "prompt": st.session_state.image_prompt,
                                "negative_prompt": st.session_state.negative_prompt,
                                "width": width,
                                "height": height,
                                "steps": steps,
                                "cfg_scale": cfg_scale,
                                "samples": samples
                            },
                            timeout=60
                        )
                        
                        response.raise_for_status()
                        result = response.json()
                        
                        if "images" in result:
                            for i, image in enumerate(result["images"]):
                                img_data = base64.b64decode(image["base64"])
                                img = Image.open(BytesIO(img_data))
                                
                                img_bytes = BytesIO()
                                img.save(img_bytes, format="PNG")
                                img_bytes.seek(0)
                                
                                timestamp = datetime.now().strftime("%H:%M")
                                st.session_state.messages.append({
                                    "role": "assistant",
                                    "content": f"Generated image: {st.session_state.image_prompt}",
                                    "image_data": img,
                                    "image_caption": f"Image {i+1}: {st.session_state.image_prompt}",
                                    "timestamp": timestamp
                                })
                                
                                # Display the generated image
                                st.image(img, caption=f"Image {i+1}: {st.session_state.image_prompt}")
                                st.success(f"Image {i+1} generated successfully!")
                                st.session_state.active_tab = "chat"
                                st.rerun()
                    except requests.exceptions.HTTPError as e:
                        error_detail = e.response.json().get('detail', str(e))
                        st.error(f"Image generation failed: {error_detail}")
                    except requests.exceptions.RequestException as e:
                        st.error(f"Failed to connect to the image generation service: {str(e)}")
                        st.info("Please ensure the backend server is running and the STABILITY_API_KEY is set.")
                    except Exception as e:
                        st.error(f"An unexpected error occurred: {str(e)}")
            else:
                st.warning("Please enter an image description")
