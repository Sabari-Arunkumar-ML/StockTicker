import faiss
import processing, constants
from transformers import BertTokenizer, TFBertForSequenceClassification

from flask import Flask, request, jsonify

app = Flask(__name__)
from transformers import BertTokenizer, TFBertModel

# Just for this example,  we load index, models,tokenizer at start.
index = faiss.read_index(constants.faiss_index_bin)
model = TFBertModel.from_pretrained(constants.base_bert_model)
tokenizer = BertTokenizer.from_pretrained(constants.base_bert_model)

@app.route('/search', methods=['POST'])
def search():
    query_text = request.json['query']
    encoded_input = tokenizer(query_text, padding=True, truncation=True, return_tensors='tf')
    query_embedding = processing.get_cls_embeddings(encoded_input, model)

    D, I = index.search(query_embedding, k=constants.top_matching_results_to_return)  # Searching for top-5 similar vectors
    neighbours = []
    # matching_docs = []
    for idx in I[0]:
        neighbours.append(constants.training_dataset[idx])
        # matching_docs.append(constants.training_dataset[idx])
        # print(f"Neighbor: {constants.training_dataset[idx]}, Label: {constants.training_dataset_label[idx]}")
    return jsonify({'distances': D.tolist(), "nearest": neighbours})

if __name__ == '__main__':
    app.run(debug=True)