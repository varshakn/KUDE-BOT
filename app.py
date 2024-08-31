from flask import Flask, send_from_directory, request, jsonify
from kude import generate_combined_response  # Import the function from kude.py

app = Flask(__name__)

@app.route("/")
def serve_index():
    return send_from_directory('kude_chatbot', 'index.html')

@app.route("/get_response", methods=["POST"])
def get_response():
    user_input = request.json.get("user_input")
    print(f"User input received: {user_input}")  # Debugging: Check user input
    
    try:
        response = generate_combined_response(user_input)
        print(f"Generated response: {response}")  # Debugging: Check generated response
    except Exception as e:
        print(f"Error generating response: {e}")  # Debugging: Print error message
        response = "An error occurred while processing your request."
    
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=False)  # Set debug=False to avoid automatic restarts

