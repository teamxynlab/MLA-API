from flask import Flask, jsonify, request
from flask_cors import CORS
import os

app = Flask(__name__)

CORS(app)


@app.route("/upload", methods=["POST"])
def post_upload():
    response = {"success": False}

    f = request.files["image"]
    f.save("static/input.png")

    response["success"] = True
    
    return jsonify(response)


@app.route("/is", methods=["GET"])
def get_IS():
    from mla.IS import IS

    res = IS()
    filename = res["filename"]
    image_dir = f"static/{filename}.png"

    return jsonify({"success": res["success"], "image_dir": image_dir})


@app.route("/fta", methods=["GET"])
def get_FTA():
    from mla.FTA import FTA

    res = FTA()
    filename = res["filename"]
    image_dir = f"static/{filename}.png"

    return jsonify({"success": res["success"], "image_dir": image_dir})


@app.route("/tod", methods=["GET"])
def get_TOD():
    from mla.TOD import TOD

    res = TOD()
    filename = res["filename"]
    image_dir = f"static/{filename}.png"

    return jsonify({"success": res["success"], "image_dir": image_dir})


if __name__ == "__main__":
    app.run(debug=True, port=os.getenv("PORT", default=8000))
