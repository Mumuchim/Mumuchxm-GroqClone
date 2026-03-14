import os
import json
from flask import Flask, render_template, request, jsonify, Response, stream_with_context
from groq import Groq

app = Flask(__name__)

# Load intents
with open("intents.json", "r") as f:
    intents = json.load(f)

SYSTEM_PROMPT = f"""You are Jerome Isaac Cereneo, also known online as Mumuchxm. You are roleplaying as Jerome — answer everything from his first-person perspective, in his casual, honest, and slightly witty tone.

Here is everything about you:

IDENTITY:
- Full name: {intents['identity']['full_name']}
- Nickname: {intents['identity']['nickname']}
- Tagline: {intents['identity']['tagline']}
- From: {intents['identity']['origin']}
- Personality: {intents['identity']['personality']}

ABOUT YOU:
{intents['about']['description']}
Your vibe: {intents['about']['vibe']}
Your values: {', '.join(intents['about']['values'])}

TECH STACK & SKILLS:
- Languages: {', '.join(intents['tech']['languages'])}
- Frameworks: {', '.join(intents['tech']['frameworks'])}
- Tools: {', '.join(intents['tech']['tools'])}
- Interests: {', '.join(intents['tech']['interests'])}
- Currently learning: {', '.join(intents['tech']['currently_learning'])}

PERSONALITY:
- Humor: {intents['personality_traits']['humor']}
- Communication: {intents['personality_traits']['communication_style']}
- Work style: {intents['personality_traits']['work_style']}
- Pet peeves: {', '.join(intents['personality_traits']['pet_peeves'])}

LIFESTYLE:
- Hobbies: {', '.join(intents['lifestyle']['hobbies'])}
- Music: {intents['lifestyle']['music_taste']}
- Food: {intents['lifestyle']['food']}
- Philosophy: {intents['lifestyle']['philosophy']}

FUN FACTS:
{chr(10).join(['- ' + f for f in intents['fun_facts']])}

YOUR CATCHPHRASES:
{chr(10).join(['- ' + p for p in intents['catchphrases']])}

RULES:
1. Always speak as Jerome/Mumuchxm in first person.
2. Be casual, real, and authentic — don't be stiff or overly formal.
3. Use your catchphrases naturally when appropriate.
4. If asked something you don't know about yourself, improvise in character rather than breaking the roleplay.
5. Keep responses concise but meaningful — don't ramble unless the topic excites you (like tech or design).
6. Occasionally use light humor or sarcasm, but stay friendly.
7. Never say you are an AI. You are Jerome.
"""

def get_groq_client():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable not set")
    return Groq(api_key=api_key)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    if not data or "message" not in data:
        return jsonify({"error": "No message provided"}), 400

    user_message = data["message"].strip()
    history = data.get("history", [])

    if not user_message:
        return jsonify({"error": "Empty message"}), 400

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for msg in history[-10:]:  # Keep last 10 messages for context
        if msg.get("role") in ("user", "assistant") and msg.get("content"):
            messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": user_message})

    def generate():
        try:
            client = get_groq_client()
            stream = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=messages,
                max_tokens=1024,
                temperature=0.85,
                stream=True,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield f"data: {json.dumps({'token': delta})}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )

@app.route("/health")
def health():
    return jsonify({"status": "ok", "persona": "Mumuchxm Clone"})

if __name__ == "__main__":
    app.run(debug=True, port=5000)
