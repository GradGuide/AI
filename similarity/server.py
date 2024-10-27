#!/usr/bin/env python
from flask import Flask, request, jsonify
from flask_cors import CORS


import json
import bert
import sbert
import cosine_similarity

app = Flask(__name__)
CORS(app)


@app.route('/')
def index():
    return '<h1>HI!</h1>'

@app.route('/api/compare', methods=['POST'])
def compare():
    data = request.get_json()

    if not ('text1' in data and 'text2' in data):
        return jsonify({'error': 'Missing text1 or text2 in request'}), 400

    alg = 'cosine-similarity'
    if 'alg' in data:
        alg = data['alg']
        
    paragraphs = [data['text1'], data['text2']]

    sim = 0.
    
    match alg:
        case 'sbert':
            sim = sbert.sbert(paragraphs)
        case 'bert':
            sim = bert.bert(paragraphs)
        case 'cosine-similarity':
            sim = cosine_similarity.similarity(paragraphs)
        case _:
            sim = cosine_similarity.similarity(paragraphs)

    return jsonify({"similarity": json.dumps(sim.tolist()[0][1])})


app.run(debug=True)
