from flask import Flask, jsonify
import os

app = Flask(__name__)


@app.route("/output")
def index():
    return jsonify({"message": "Let's go MLA!"})


if __name__ == "__main__":
    app.run(debug=True, port=os.getenv("PORT", default=8000))
