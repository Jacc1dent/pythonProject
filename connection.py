from flask import Flask, jsonify
import groupProject

app = Flask(__name__)


@app.route('/run_predictions', methods=['GET'])
def run_predictions():
    results = groupProject.run_predictions()
    return jsonify(results)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
