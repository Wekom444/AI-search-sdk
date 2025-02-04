from flask import Flask, request, jsonify
from transformers import pipeline

app = Flask(__name__)

# Load the AI model
search_model = pipeline('question-answering', model="distilbert-base-cased-distilled-squad")

@app.route('/search', methods=['POST'])
def search():
    try:
        # Get request data
        data = request.get_json()
        query = data.get("query")
        context = data.get("context")

        # Validate input
        if not query or not context:
            return jsonify({"error": "Both 'query' and 'context' are required"}), 400

        # Run the AI model
        response = search_model(question=query, context=context)
        return jsonify({"result": response['answer']}), 200

    except KeyError as e:
        return jsonify({"error": f"Missing key: {str(e)}"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
