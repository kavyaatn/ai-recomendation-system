# 🤖 Conversational Bot using Mistral-7B

A conversational chatbot powered by Mistral-7B-Instruct-v0.3 using Hugging Face Transformers and Streamlit. This bot maintains conversational history, generates intelligent replies, and offers follow-up suggestions for a more interactive and guided chat experience.

## 🔍 Project Overview

This app provides a simple yet powerful interface to interact with a large language model using conversational prompts. It is ideal for educational, research, and demonstration purposes to explore how modern LLMs can hold a context-aware dialogue and assist users intelligently.

## ⚙️ Features

- 💬 Conversational Memory – Keeps track of user and assistant messages
- 💡 Smart Suggestions – Generates follow-up questions after each response
- 🚀 Streamlit UI – Interactive web-based chat interface
- 🧠 Mistral-7B Model – Uses a powerful transformer-based model from Hugging Face
- 🔁 Reset Chat – Start a fresh conversation anytime

## 🧱 Tech Stack

| Tool/Library | Purpose |
|--------------|---------|
| Streamlit | Web UI framework for the chatbot |
| Hugging Face transformers | LLM pipeline and model handling |
| PyTorch | Backend for the model inference |
| Mistral-7B-Instruct | The LLM used for chat and generation |

## Prerequisites

- Python 3.8 or higher
- pip package manager

## Installation

1. Clone the repository:
```bash
git clone https://github.com/kavyaatn/ai-recomendation-system.git
cd ai-recomendation-system
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the Streamlit app:
```bash
streamlit run app.py
```

2. Open your browser and navigate to the local URL provided by Streamlit (typically http://localhost:8501)

3. Start chatting with the bot!

## Technical Details

- **Framework**: Streamlit
- **Model**: Mistral-7B-Instruct-v0.3
- **Main Features**:
  - Contextual response generation
  - Dynamic suggestion generation
  - Conversation state management
  - Markdown support for formatted responses

## File Structure

```
.
├── app.py              # Main application file
├── requirements.txt    # Project dependencies
├── .env               # Environment variables
└── README.md          # Project documentation
```

## Environment Variables

Required environment variables for the project:

```env
HUGGINGFACE_TOKEN=your_token_here
```

## Dependencies

- streamlit
- transformers
- torch
- python-dotenv

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.