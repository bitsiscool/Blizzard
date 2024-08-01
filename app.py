import cohere
from flask import Flask, render_template, request, jsonify, session
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired
import secrets

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)  # Set a secret key for CSRF protection

# Initialize Cohere client
co = cohere.Client('wAFnIR9jjaDJlWAP1tFbxSdTCzfWhJNmn9rPnjVM')

class Form(FlaskForm):
    text = StringField('Enter text to search', validators=[DataRequired()])
    submit = SubmitField('Submit')

@app.route('/', methods=['GET', 'POST'])
def home():
    form = Form()
    # Initialize the chat history in the session
    if 'chat_history' not in session:
        session['chat_history'] = []

    if form.validate_on_submit():
        text = form.text.data
        # Append user input to chat history
        session['chat_history'].append({"role": "user", "content": text})

        response = co.generate(
            model='command-nightly',
            prompt=get_prompt(session['chat_history']),
            max_tokens=300,
            temperature=0.9,
            k=0,
            p=0.75,
            stop_sequences=[],
            return_likelihoods='NONE'
        )
        
        # Extract the bot's response
        output = response.generations[0].text.strip()
        
        # Append bot response to chat history
        session['chat_history'].append({"role": "bot", "content": output})

        return render_template('home.html', form=form, output=output)

    return render_template('home.html', form=form, output=None)

@app.route('/ask', methods=['POST'])
def ask():
    if 'chat_history' not in session:
        session['chat_history'] = []

    user_input = request.form.get('query')
    if not user_input:
        return jsonify({"answer": "Please enter a valid input."})

    # Append user input to chat history
    session['chat_history'].append({"role": "user", "content": user_input})

    response = co.generate(
        model='command-nightly',
        prompt=get_prompt(session['chat_history']),
        max_tokens=300,
        temperature=0.9,
        k=0,
        p=0.75,
        stop_sequences=[],
        return_likelihoods='NONE'
    )

    # Extract the bot's response
    answer = response.generations[0].text.strip()

    # Append bot response to chat history
    session['chat_history'].append({"role": "bot", "content": answer})

    return jsonify({"answer": answer})

def get_prompt(chat_history):
    """
    Generate a prompt for the Cohere API based on the chat history.
    """
    history_text = "\n".join(
        f"{entry['role'].capitalize()}: {entry['content']}" for entry in chat_history
    )
    return history_text

if __name__ == "__main__":
    app.run(debug=True)
