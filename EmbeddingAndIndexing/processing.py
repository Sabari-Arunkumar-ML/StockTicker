from transformers import BertTokenizer, TFBertForSequenceClassification
import tensorflow as tf # type: ignore
import os
import numpy as np
import faiss
import EmbeddingAndIndexing.constants as constants
from transformers import BertTokenizer, TFBertModel



def process_text(text, label):
    # Preprocess texts by converting in to Tokens, and their paddings
    # We are considering bert opensoure model as the base for tokenization
    tokenizer = BertTokenizer.from_pretrained(constants.base_bert_model)
    encoded_input = tokenizer(text, padding=True, truncation=True, return_tensors='tf')
    
    # Fine-tune or update model with new insight
    model = fine_tune_model(encoded_input,label)
    
    # Generate embeddings
    embeddings = get_cls_embeddings(encoded_input, model)
    
    # Index embeddings in FAISS
    index_embeddings_in_faiss(embeddings)

def fine_tune_model(encoded_input, label):
    # Load a local fine tuned bert model, file can be kept in any durable storage
    model = None
    if not os.path.exists(constants.trained_model_location):
        model = TFBertForSequenceClassification.from_pretrained(constants.base_bert_model, num_labels=2)
    else:
        model = TFBertForSequenceClassification.from_pretrained(constants.trained_model_location, num_labels=2)
    
    # Just for example, keeping the batch size as one
    input_ids = encoded_input['input_ids']
    attention_mask = encoded_input['attention_mask']

    dataset = tf.data.Dataset.from_tensor_slices(({'input_ids': input_ids, 'attention_mask': attention_mask}, [label]))
    dataset = dataset.batch(2)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(dataset, epochs=1)
    
    # Save the fine-tuned model
    model.save_pretrained(constants.trained_model_location)
    return model



def get_cls_embeddings(encoded_input, model):
    # TFBertForSequenceClassification [Param: Model] currently doesnt support hidden states loading, as of now, we can load it from path and optimize it in future
    fine_tuned_model = TFBertModel.from_pretrained(constants.trained_model_location)
    output = fine_tuned_model(encoded_input)
    embeddings = output.last_hidden_state  

    cls_embeddings = embeddings[:, 0, :].numpy()
    return cls_embeddings


index = None

def index_embeddings_in_faiss(embeddings):
    global index
    if index is None:
        d = embeddings.shape[1] 
        index = faiss.IndexFlatL2(d)
    index.add(embeddings)
    faiss.write_index(index, constants.faiss_index_bin)