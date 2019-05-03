"""
Temporary solution: heavy refactoring needed!
"""

from predict import load_model, predict
from flask import Flask, request, jsonify
from constants import BEST_MODEL_PATH

app = Flask(__name__)
model, data_loader = load_model(BEST_MODEL_PATH)
    

@app.route('/api', methods=['POST'])
def predict_toxicity():
    data = request.get_json(force=True).items()
    data = {key: str(value) for key, value in data if len(str(value)) > 0}
    if len(data) > 0:
        keys = list(data.keys())
        values = list(data.values())
        predictions = predict(values, model, data_loader)
        respond = {key: prediction for key, prediction in zip(keys, predictions)}
        return jsonify(respond)
    else:
        return jsonify([])


@app.route('/test', methods=['GET'])
def test():
    return 'connection ok'


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
