from flask import Flask, render_template, request
from Transformer_bot import predict

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("home.html")

@app.route("/get")
def get_bot_response_transformer():
    userText = request.args.get('msgTransformer')
    results = predict(userText)

    return str(results)


if __name__ == "__main__":
    app.run(debug=True)
