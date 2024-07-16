import time
import processing
import constants
# Mocking some random data which will be a streaming ticker insights in our actual use case
def fetch_new_texts():
    insight_texts = constants.training_dataset
    # insight_labels = constants.training_dataset_label
    for insight_text in insight_texts:
        processing.process_text(insight_text, None)
fetch_new_texts()