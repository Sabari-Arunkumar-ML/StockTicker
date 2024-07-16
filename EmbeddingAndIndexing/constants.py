trained_model_location = "./trainedModel/fine_tuned_bert"
faiss_index_bin = "faiss_index.bin"
base_bert_model = "bert-base-uncased"
max_token_length = 300
buffer_for_token_size_threshold = 10
training_dataset = [
    "The advent of CRISPR technology has ushered in a new era in the field of biological research. By allowing precise edits to the DNA of various organisms, scientists can now target specific genes responsible for hereditary traits. This breakthrough has paved the way for the development of genetically engineered crops that are more resistant to pests and environmental stresses. Additionally, it holds promise for eliminating genetic disorders by correcting mutations at their source, offering a potential cure for diseases that were previously deemed untreatable.",
    "Recent advancements in medical science have led to the development of novel therapies for managing chronic illnesses. Among these, the use of monoclonal antibodies has shown significant efficacy in treating autoimmune conditions. These biologics work by targeting specific proteins in the immune system, thereby reducing inflammation and preventing tissue damage. Clinical trials are also exploring their potential in combating various forms of cancer, offering new hope to patients with limited treatment options.",
    "Modern agricultural practices are being revolutionized by the integration of cutting-edge technologies. Precision farming, which utilizes satellite imagery and sensors, allows farmers to monitor crop health in real-time and optimize resource usage. This approach not only increases yield but also minimizes the environmental impact by reducing the need for chemical fertilizers and pesticides. Furthermore, advancements in hydroponics and vertical farming are enabling the cultivation of crops in urban areas, thereby addressing food security concerns in densely populated regions.",
]
# training_dataset_label = [1,2,3]
top_matching_results_to_return = 2
