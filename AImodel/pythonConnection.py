from flask import Flask, request, jsonify
from generateGraph import main as generate_graph

app = Flask(__name__)

@app.route('/generateGraph', methods=['GET'])
def searchGraph():
    keyword = request.args.get('keyword')
    result = generate_graph([keyword])

    return jsonify(result)


@app.route('/compareGraph', methods=['GET'])
def compareGraph():
    keywords_str = request.args.get('keywords')
    keywords = keywords_str.split(', ')
    print(keywords)
    result = generate_graph(keywords)

    return jsonify(result)


if __name__ == '__main__':
    app.run(port=5000, debug=True)


