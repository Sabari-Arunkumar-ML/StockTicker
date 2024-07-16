# Stock Ticker

## Instructions to run Embedding/Indexer Program
1. Define the training datatest in constants.py, and if needed modify other params like top_matching_results_to_return etc.,
2. Run data_ingestion.py and it will generate faiss_index.bin. This is the vector DB which will have the generated vector embedding
3. Run search_faiss_index.py and it will listen on 8080.
   we can run our query "curl --location 'http://localhost:5000/search' --header 'Content-Type: application/json' --data '{"query": "technology backing conserns associated with food"} " and see the matching vectors