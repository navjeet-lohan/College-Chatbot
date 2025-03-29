from flask import Flask, render_template, request, jsonify
from chatbot import CollegeChatbot  # Import your existing class
import os

app = Flask(__name__)

# Initialize your chatbot exactly as in chatbot.py
bot = CollegeChatbot()

@app.route("/")
def home():
    return render_template("base.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        text = data.get("message", "").strip()
        
        if not text:
            return jsonify({"answer": "Please enter a valid question"})
        
        # Use your existing process_query method
        responses = bot.process_query(text)
        
        # Format response for web (same as your console output)
        if not responses:
            return jsonify({"answer": "No response available"})
        
        # Combine all responses into one answer
        answer = []
        for response in responses:
            if isinstance(response, str):
                answer.append(response)
            elif isinstance(response, dict):
                answer.append(response.get("text", ""))
                if "url" in response:
                    answer.append(f"More info: {response['url']}")
        
        return jsonify({
            "answer": "\n".join(answer),
            "confidence": bot.predict_intent(text)[1]  # Include confidence score
        })
        
    except Exception as e:
        print(f"Error in predict endpoint: {str(e)}")
        return jsonify({"answer": "Sorry, I encountered an error processing your request"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)