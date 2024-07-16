from transformers import BertTokenizer, TFBertForSequenceClassification
import tensorflow as tf # type: ignore
import os
import numpy as np
import faiss
import constants as constants
from transformers import BertTokenizer, TFBertModel



def process_text(text, label):
    # Preprocess texts by converting in to Tokens, and their paddings
    # We are considering bert opensoure model as the base for tokenization
    tokenizer = BertTokenizer.from_pretrained(constants.base_bert_model)
    chunks = []
    current_chunk = []
    current_length = 0
    for sentence in text.split('.'):
        current_length += len(tokenizer.tokenize(sentence.strip()))
        # Let us use buffer_for_token_size_threshold in later stages
        if current_length >= constants.max_token_length:
            chunks.append(' '.join(current_chunk))
            current_chunk = []
            current_length = 0
        current_chunk.append(sentence.strip())
    if len(current_chunk) != 0:
        chunks.append(' '.join(current_chunk))
    
    input_ids = []
    attention_masks = []
    for chunk in chunks:
        tokens = tokenizer.encode_plus(chunk, add_special_tokens=True, max_length=constants.max_token_length, padding='max_length', truncation=True, return_tensors='tf')
        input_ids.append(tokens['input_ids'])
        attention_masks.append(tokens['attention_mask'])
    input_ids = tf.concat(input_ids, axis=0)
    attention_masks = tf.concat(attention_masks, axis=0)
    
    
    # print(input_ids, attention_masks)
    # input_ids = encoded_input['input_ids']
    # attention_mask = encoded_input['attention_mask']
    # Fine-tune or update model with new insight
    # model = fine_tune_model(encoded_input,label)
    
    encoded_input = {}
    encoded_input['input_ids'] = input_ids
    encoded_input['attention_mask']= attention_masks   
    # Generate embeddings
    
    embeddings = get_cls_embeddings(encoded_input, None)
    
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

    fine_tuned_model = TFBertModel.from_pretrained(constants.base_bert_model)
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