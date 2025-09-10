from flask import Flask, render_template, request, jsonify
from ai_backend import get_ai_response

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message')
    if not user_input:
        return jsonify({'response': 'Please enter a message.'})

    response = get_ai_response(user_input)
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)