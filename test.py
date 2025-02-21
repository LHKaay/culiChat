from flask import Flask, request

VERIFY_TOKEN = "LHKishandsome"

app = Flask(__name__)

@app.route('/')
def home():
    return "Successful running"

@app.route('/webhook', methods=['GET'])
def verify_webhook():
    if request.args.get("hub.verify_token") == VERIFY_TOKEN:
        return request.args.get("hub.challenge")
    return "Verification token mismatch", 403

if __name__ == '__main__':
    app.run(port=5000)