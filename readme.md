# 🤖 Bot vs Bot Debate Arena

Welcome to the **Bot vs Bot Debate Arena**, a lightweight Streamlit app that pits two reasoning-capable LLMs against each other in a fully autonomous debate. OpenAI’s `o4-mini` goes head-to-head with Anthropic’s `Claude 3.7 Sonnet`, challenging each other on classic debate topics—or whatever spicy prompt you throw at them.

---

## 🚀 What This App Does
- Spins up a clean UI using Streamlit
- Runs a structured, turn-based debate between two LLMs
- Logs each turn's visible response and (if applicable) reasoning summary
- Outputs a full transcript and a JSON reasoning log at the end

---

## 🛠️ Tech Stack
- **Frontend**: Streamlit
- **Backend**: Python
- **Models Used**:
  - OpenAI: `o4-mini`
  - Anthropic: `Claude 3.7 Sonnet`

---

## ⚙️ Requirements
- Python 3.9+
- API key from OpenAI (`o4-mini` access)
- API key from Anthropic (`claude-3-7-sonnet` access)

---

## 🔐 Setup Instructions

1. **Clone the repo**  
```bash
git clone https://github.com/yourusername/bot-vs-bot-debate.git
cd bot-vs-bot-debate
```

2. **Add your API keys to a `.env` file**  
```env
OPENAI_API_KEY=your-openai-key
ANTHROPIC_API_KEY=your-anthropic-key
```

3. **Install dependencies**  
```bash
python -m venv venv
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows
pip install -r requirements.txt
```

4. **Run the app**  
```bash
streamlit run app.py
```

---

## 📁 Output Files
- `debate_transcript_log.json` – structured conversation log
- `debate_reasoning_log.json` – LLM reasoning steps (if available)
- `debate_transcript_final.txt` – cleaned, human-readable transcript

---

## 🧹 Customization Ideas
- Use voices via ElevenLabs
- Animate debates with avatars
- Swap out system prompts or positions
- Use a single model as a sparring partner

---

## 📄 License
MIT License – modify, remix, redistribute freely.

---

## 💬 Community
Want to contribute or just curious how this works?  
Come hang out on [AI for Normal People](https://www.youtube.com/@aifornormalpeoples)
