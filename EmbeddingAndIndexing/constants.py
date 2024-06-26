trained_model_location = "./trainedModel/fine_tuned_bert"
faiss_index_bin = "faiss_index.bin"
base_bert_model = "bert-base-uncased"
training_dataset = [
    "I loved 'Inception'! The acting was superb.",
    "'The Room' was boring and predictable.",
    "The plot twists in 'The Sixth Sense' were unexpected and thrilling.",
    "The cinematography in 'Blade Runner 2049' was breathtaking.",
    "The dialogue in 'Batman & Robin' felt unnatural and forced.",
    "'Battlefield Earth' was a waste of time.",
    "'The Godfather' is a masterpiece! Will watch again.",
    "'Transformers: Age of Extinction' was poorly executed and too long.",
    "'La La Land' was an absolute joy to watch from start to finish.",
    "'Cats' was a terrible film, would not recommend."
]
training_dataset_label = [1, 0, 1, 1, 0, 0, 1, 0, 1, 0]
top_matching_results_to_return = 4
