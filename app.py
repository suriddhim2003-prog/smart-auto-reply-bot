"""
Smart Auto-Reply Bot — AI-powered customer inquiry responder.
Demo project for AI automation freelancing.
"""

import json
import os
import csv
from datetime import datetime
from pathlib import Path
from flask import Flask, render_template, request, jsonify

try:
    import google.generativeai as genai
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False

app = Flask(__name__)

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)
LOG_FILE = DATA_DIR / "reply_log.csv"

BUSINESS_PROFILES = {
    "salon": {
        "name": "Glow Beauty Studio",
        "type": "Beauty Salon",
        "services": "Haircut (₹500), Hair Color (₹2000), Facial (₹800), Manicure (₹400), Bridal Package (₹15000)",
        "hours": "10 AM – 8 PM, Monday to Saturday (closed Sunday)",
        "location": "Shop 12, Market Complex, Sector 18, Noida",
        "booking": "Walk-in or call 98765-43210",
        "tone": "Warm, friendly, uses emojis occasionally"
    },
    "clinic": {
        "name": "CareFirst Dental Clinic",
        "type": "Dental Clinic",
        "services": "Cleaning (₹1000), Filling (₹1500), Root Canal (₹5000), Braces consultation (free), Whitening (₹3000)",
        "hours": "9 AM – 6 PM, Monday to Saturday",
        "location": "2nd Floor, Wellness Tower, MG Road, Gurgaon",
        "booking": "Call 98765-11111 or WhatsApp to book",
        "tone": "Professional, reassuring, empathetic"
    },
    "coaching": {
        "name": "ElevateIQ Coaching",
        "type": "Online Coaching / Tutoring",
        "services": "IELTS Prep (₹8000/month), Spoken English (₹3000/month), Interview Prep (₹2000/session), Group batch (₹5000/month)",
        "hours": "Flexible — morning and evening slots available",
        "location": "Online (Zoom/Google Meet)",
        "booking": "Fill the form on our Instagram or DM us",
        "tone": "Motivating, friendly, slightly casual"
    }
}


def get_ai_reply(customer_message: str, business_key: str) -> dict:
    biz = BUSINESS_PROFILES[business_key]

    prompt = f"""You are an AI auto-reply assistant for "{biz['name']}" ({biz['type']}).

BUSINESS INFO:
- Services & Prices: {biz['services']}
- Hours: {biz['hours']}
- Location: {biz['location']}
- How to book: {biz['booking']}
- Tone: {biz['tone']}

CUSTOMER MESSAGE:
"{customer_message}"

INSTRUCTIONS:
1. Classify the customer's intent into ONE of: pricing, appointment, complaint, general_info, greeting, spam
2. Write a helpful reply in the business's tone. Keep it under 3 sentences. Include relevant info (prices, hours, etc.) when applicable.
3. If it's spam or irrelevant, reply politely that you can't help with that.

Respond in this exact JSON format:
{{"intent": "...", "reply": "...", "confidence": 0.0-1.0}}

Return ONLY the JSON, nothing else."""

    if HAS_GEMINI:
        api_key = os.environ.get("GEMINI_API_KEY", "")
        if api_key:
            try:
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel("gemini-2.0-flash-lite")
                resp = model.generate_content(prompt)
                text = resp.text.strip()
                if text.startswith("```"):
                    text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
                return json.loads(text)
            except Exception as e:
                print(f"Gemini error: {e}")

    return fallback_reply(customer_message, biz)


def fallback_reply(message: str, biz: dict) -> dict:
    msg = message.lower()
    if any(w in msg for w in ["price", "cost", "rate", "kitna", "charge", "fee", "paisa", "how much", "kya rate", "amount", "rupee", "rs", "₹", "package"]):
        return {
            "intent": "pricing",
            "reply": f"Here are our prices: {biz['services']}. Would you like to book an appointment?",
            "confidence": 0.7
        }
    elif any(w in msg for w in ["book", "appointment", "slot", "available", "timing", "schedule"]):
        return {
            "intent": "appointment",
            "reply": f"We're open {biz['hours']}. To book: {biz['booking']}. What time works for you?",
            "confidence": 0.7
        }
    elif any(w in msg for w in ["complaint", "bad", "worst", "unhappy", "problem", "issue"]):
        return {
            "intent": "complaint",
            "reply": f"We're sorry to hear that. Please share more details and we'll resolve this right away. You can also reach us at {biz['booking']}.",
            "confidence": 0.6
        }
    elif any(w in msg for w in ["hi", "hello", "hey", "good morning", "namaste"]):
        return {
            "intent": "greeting",
            "reply": f"Hello! Welcome to {biz['name']}. How can we help you today?",
            "confidence": 0.8
        }
    else:
        return {
            "intent": "general_info",
            "reply": f"Thanks for reaching out to {biz['name']}! We're at {biz['location']}, open {biz['hours']}. How can we help?",
            "confidence": 0.5
        }


def log_interaction(business: str, customer_msg: str, intent: str, reply: str, confidence: float):
    write_header = not LOG_FILE.exists()
    with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["timestamp", "business", "customer_message", "intent", "auto_reply", "confidence"])
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            business, customer_msg, intent, reply, f"{confidence:.2f}"
        ])


@app.route("/")
def index():
    return render_template("index.html", businesses=BUSINESS_PROFILES)


@app.route("/reply", methods=["POST"])
def reply():
    data = request.json
    customer_msg = data.get("message", "").strip()
    business_key = data.get("business", "salon")

    if not customer_msg:
        return jsonify({"error": "Empty message"}), 400

    result = get_ai_reply(customer_msg, business_key)
    log_interaction(business_key, customer_msg, result["intent"], result["reply"], result["confidence"])

    return jsonify({
        "intent": result["intent"],
        "reply": result["reply"],
        "confidence": result["confidence"],
        "business": BUSINESS_PROFILES[business_key]["name"]
    })


@app.route("/log")
def view_log():
    if not LOG_FILE.exists():
        return jsonify([])
    rows = []
    with open(LOG_FILE, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return jsonify(rows[-50:])


if __name__ == "__main__":
    print("\n  Smart Auto-Reply Bot")
    print("  Open http://localhost:5050 in your browser\n")
    app.run(host="127.0.0.1", port=5050, debug=True)
