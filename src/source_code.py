#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# %% [markdown]
# # Overview
# 
# It only takes one toxic comment to sour an online discussion. The Conversation AI team, a research initiative founded by [Jigsaw](https://jigsaw.google.com/) and Google, builds technology to protect voices in conversation. A main area of focus is machine learning models that can identify toxicity in online conversations, where toxicity is defined as anything *rude, disrespectful or otherwise likely to make someone leave a discussion*. Our API, [Perspective](http://perspectiveapi.com/), serves these models and others in a growing set of languages (see our [documentation](https://github.com/conversationai/perspectiveapi/blob/master/2-api/models.md#all-attribute-types) for the full list). If these toxic contributions can be identified, we could have a safer, more collaborative internet.
# 
# In this competition, we'll explore how models for recognizing toxicity in online conversations might generalize across different languages. Specifically, in this notebook, we'll demonstrate this with a multilingual BERT (m-BERT) model. Multilingual BERT is pretrained on monolingual data in a variety of languages, and through this learns multilingual representations of text. These multilingual representations enable *zero-shot cross-lingual transfer*, that is, by fine-tuning on a task in one language, m-BERT can learn to perform that same task in another language (for some examples, see e.g. [How multilingual is Multilingual BERT?](https://arxiv.org/abs/1906.01502)).
# 
# We'll study this zero-shot transfer in the context of toxicity in online conversations, similar to past competitions we've hosted ([[1]](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification), [[2]](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge)). But rather than analyzing toxicity in English as in those competitions, here we'll ask you to do it in several different languages. For training, we're including the (English) datasets from our earlier competitions, as well as a small amount of new toxicity data in other languages.

# %% [markdown]
# # Benchmark notebook
# 
# We'll begin by importing TensorFlow, our datasets, and TensorFlow Hub, which has a pretrained multilingual model we'll use.

# %% [code]
import os, time
import pandas
import tensorflow as tf
import tensorflow_hub as hub
from kaggle_datasets import KaggleDatasets
print(tf.version.VERSION)

# %% [markdown]
# Detect TPUs or GPUs:

# %% [code]
# Detect hardware, return appropriate distribution strategy
try:
    # TPU detection. No parameters necessary if TPU_NAME environment variable is
    # set: this is always the case on Kaggle.
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print('Running on TPU ', tpu.master())
except ValueError:
    tpu = None

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
else:
    # Default distribution strategy in Tensorflow. Works on CPU and single GPU.
    strategy = tf.distribute.get_strategy()

print("REPLICAS: ", strategy.num_replicas_in_sync)

# %% [markdown]
# Set maximum sequence length and path variables.

# %% [code]
SEQUENCE_LENGTH = 128

# Note that private datasets cannot be copied - you'll have to share any pretrained models 
# you want to use with other competitors!
GCS_PATH = KaggleDatasets().get_gcs_path('jigsaw-multilingual-toxic-comment-classification')
BERT_GCS_PATH = KaggleDatasets().get_gcs_path('bert-multi')
BERT_GCS_PATH_SAVEDMODEL = BERT_GCS_PATH + "/bert_multi_from_tfhub"

# %% [markdown]
# Define the model. We convert m-BERT's output to a final probabilty estimate. We're using an [m-BERT model from TensorFlow Hub](https://tfhub.dev/tensorflow/bert_multi_cased_L-12_H-768_A-12/1).

# %% [code]
def multilingual_bert_model(max_seq_length=SEQUENCE_LENGTH, trainable_bert=True):
    """Build and return a multilingual BERT model and tokenizer."""
    input_word_ids = tf.keras.layers.Input(
        shape=(max_seq_length,), dtype=tf.int32, name="input_word_ids")
    input_mask = tf.keras.layers.Input(
        shape=(max_seq_length,), dtype=tf.int32, name="input_mask")
    segment_ids = tf.keras.layers.Input(
        shape=(max_seq_length,), dtype=tf.int32, name="all_segment_id")
    
    # Load a SavedModel on TPU from GCS. This model is available online at 
    # https://tfhub.dev/tensorflow/bert_multi_cased_L-12_H-768_A-12/1. You can use your own 
    # pretrained models, but will need to add them as a Kaggle dataset.
    bert_layer = tf.saved_model.load(BERT_GCS_PATH_SAVEDMODEL)
    # Cast the loaded model to a TFHub KerasLayer.
    bert_layer = hub.KerasLayer(bert_layer, trainable=trainable_bert)

    pooled_output, _ = bert_layer([input_word_ids, input_mask, segment_ids])
    output = tf.keras.layers.Dense(32, activation='relu')(pooled_output)
    output = tf.keras.layers.Dense(1, activation='sigmoid', name='labels')(output)

    return tf.keras.Model(inputs={'input_word_ids': input_word_ids,
                                  'input_mask': input_mask,
                                  'all_segment_id': segment_ids},
                          outputs=output)

# %% [markdown]
# Load the preprocessed dataset. See the demo notebook for sample code for performing this preprocessing.

# %% [code]
def parse_string_list_into_ints(strlist):
    s = tf.strings.strip(strlist)
    s = tf.strings.substr(
        strlist, 1, tf.strings.length(s) - 2)  # Remove parentheses around list
    s = tf.strings.split(s, ',', maxsplit=SEQUENCE_LENGTH)
    s = tf.strings.to_number(s, tf.int32)
    s = tf.reshape(s, [SEQUENCE_LENGTH])  # Force shape here needed for XLA compilation (TPU)
    return s

def format_sentences(data, label='toxic', remove_language=False):
    labels = {'labels': data.pop(label)}
    if remove_language:
        languages = {'language': data.pop('lang')}
    # The remaining three items in the dict parsed from the CSV are lists of integers
    for k,v in data.items():  # "input_word_ids", "input_mask", "all_segment_id"
        data[k] = parse_string_list_into_ints(v)
    return data, labels

def make_sentence_dataset_from_csv(filename, label='toxic', language_to_filter=None):
    # This assumes the column order label, input_word_ids, input_mask, segment_ids
    SELECTED_COLUMNS = [label, "input_word_ids", "input_mask", "all_segment_id"]
    label_default = tf.int32 if label == 'id' else tf.float32
    COLUMN_DEFAULTS  = [label_default, tf.string, tf.string, tf.string]

    if language_to_filter:
        insert_pos = 0 if label != 'id' else 1
        SELECTED_COLUMNS.insert(insert_pos, 'lang')
        COLUMN_DEFAULTS.insert(insert_pos, tf.string)

    preprocessed_sentences_dataset = tf.data.experimental.make_csv_dataset(
        filename, column_defaults=COLUMN_DEFAULTS, select_columns=SELECTED_COLUMNS,
        batch_size=1, num_epochs=1, shuffle=False)  # We'll do repeating and shuffling ourselves
    # make_csv_dataset required a batch size, but we want to batch later
    preprocessed_sentences_dataset = preprocessed_sentences_dataset.unbatch()
    
    if language_to_filter:
        preprocessed_sentences_dataset = preprocessed_sentences_dataset.filter(
            lambda data: tf.math.equal(data['lang'], tf.constant(language_to_filter)))
        #preprocessed_sentences.pop('lang')
    preprocessed_sentences_dataset = preprocessed_sentences_dataset.map(
        lambda data: format_sentences(data, label=label,
                                      remove_language=language_to_filter))

    return preprocessed_sentences_dataset

# %% [markdown]
# Set up our data pipelines for training and evaluation.

# %% [code]
def make_dataset_pipeline(dataset, repeat_and_shuffle=True):
    """Set up the pipeline for the given dataset.
    
    Caches, repeats, shuffles, and sets the pipeline up to prefetch batches."""
    cached_dataset = dataset.cache()
    if repeat_and_shuffle:
        cached_dataset = cached_dataset.repeat().shuffle(2048)
    cached_dataset = cached_dataset.batch(32 * strategy.num_replicas_in_sync)
    cached_dataset = cached_dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return cached_dataset

# Load the preprocessed English dataframe.
preprocessed_en_filename = (
    GCS_PATH + "/jigsaw-toxic-comment-train-processed-seqlen{}.csv".format(
        SEQUENCE_LENGTH))

# Set up the dataset and pipeline.
english_train_dataset = make_dataset_pipeline(
    make_sentence_dataset_from_csv(preprocessed_en_filename))

# Process the new datasets by language.
preprocessed_val_filename = (
    GCS_PATH + "/validation-processed-seqlen{}.csv".format(SEQUENCE_LENGTH))

nonenglish_val_datasets = {}
for language_name, language_label in [('Spanish', 'es'), ('Italian', 'it'),
                                      ('Turkish', 'tr')]:
    nonenglish_val_datasets[language_name] = make_sentence_dataset_from_csv(
        preprocessed_val_filename, language_to_filter=language_label)
    nonenglish_val_datasets[language_name] = make_dataset_pipeline(
        nonenglish_val_datasets[language_name])

nonenglish_val_datasets['Combined'] = tf.data.experimental.sample_from_datasets(
        (nonenglish_val_datasets['Spanish'], nonenglish_val_datasets['Italian'],
         nonenglish_val_datasets['Turkish']))

# %% [markdown]
# Compile our model. We'll first evaluate it on our new toxicity dataset in the 
# different languages to see its performance. After that, we'll train it on one of our English datasets, and then again evaluate its performance on the new multilingual toxicity data. As our metric, we'll use the [AUC](https://www.tensorflow.org/api_docs/python/tf/keras/metrics/AUC).

# %% [code]
with strategy.scope():
    multilingual_bert = multilingual_bert_model()

    # Compile the model. Optimize using stochastic gradient descent.
    multilingual_bert.compile(
        loss=tf.keras.losses.BinaryCrossentropy(),
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),
        metrics=[tf.keras.metrics.AUC()])

multilingual_bert.summary()

# %% [code]
# Test the model's performance on non-English comments before training.
for language in nonenglish_val_datasets:
    results = multilingual_bert.evaluate(nonenglish_val_datasets[language],
                                         steps=100, verbose=0)
    print('{} loss, AUC before training:'.format(language), results)

results = multilingual_bert.evaluate(english_train_dataset,
                                     steps=100, verbose=0)
print('\nEnglish loss, AUC before training:', results)

print()
# Train on English Wikipedia comment data.
history = multilingual_bert.fit(
    # Set steps such that the number of examples per epoch is fixed.
    # This makes training on different accelerators more comparable.
    english_train_dataset, steps_per_epoch=4000/strategy.num_replicas_in_sync,
    epochs=5, verbose=1, validation_data=nonenglish_val_datasets['Combined'],
    validation_steps=100)
print()

# Re-evaluate the model's performance on non-English comments after training.
for language in nonenglish_val_datasets:
    results = multilingual_bert.evaluate(nonenglish_val_datasets[language],
                                         steps=100, verbose=0)
    print('{} loss, AUC after training:'.format(language), results)

results = multilingual_bert.evaluate(english_train_dataset,
                                     steps=100, verbose=0)
print('\nEnglish loss, AUC after training:', results)

# %% [markdown]
# # Generate predictions
# 
# Finally, we'll use our trained multilingual model to generate predictions for the test data.

# %% [code]
import numpy as np

TEST_DATASET_SIZE = 63812

print('Making dataset...')
preprocessed_test_filename = (
    GCS_PATH + "/test-processed-seqlen{}.csv".format(SEQUENCE_LENGTH))
test_dataset = make_sentence_dataset_from_csv(preprocessed_test_filename, label='id')
test_dataset = make_dataset_pipeline(test_dataset, repeat_and_shuffle=False)

print('Computing predictions...')
test_sentences_dataset = test_dataset.map(lambda sentence, idnum: sentence)
probabilities = np.squeeze(multilingual_bert.predict(test_sentences_dataset))
print(probabilities)

print('Generating submission file...')
test_ids_dataset = test_dataset.map(lambda sentence, idnum: idnum).unbatch()
test_ids = next(iter(test_ids_dataset.batch(TEST_DATASET_SIZE)))[
    'labels'].numpy().astype('U')  # All in one batch

np.savetxt('submission.csv', np.rec.fromarrays([test_ids, probabilities]),
           fmt=['%s', '%f'], delimiter=',', header='id,toxic', comments='')
get_ipython().system('head submission.csv')

# %% [markdown]
# # Future thoughts
# 
# Our results show improvements in recognizing toxicity in every language, even though we only trained in English. But this notebook is only as a basic demonstration of cross-lingual model transfer, and how it applies to toxicity in online comments — we've left a lot for you to explore in the competition! Here are a few starter thoughts for how to improve upon what we've done:
# 1.   How might toxicity vary by language? The demo illustrates that some properties are shared, i.e. we do learn toxicity in other languages from toxicity in English. But our final AUC values vary significantly by language. What factors might contribute to this? Are there ways you could use this?
# 2.   The scores in the `jigsaw-unintended-bias-train` dataset reflect the *probability* that an individual human rater will find a comment toxic (see [our blog post](https://medium.com/the-false-positive/creating-labeled-datasets-and-exploring-the-role-of-human-raters-56367b6db298) for more). However, for the dataset `jigsaw-toxic-comment-train` and for the non-English data, we only have binary labels. What information is there in the non-binary scores, i.e. in terms of rater agreement and disagreement?
# 3.   There are a lot of things you could try to improve upon the demo (different models, maximum sequence lengths, optimizers, learning rates and learning rate schedules, etc). You could also include the dataset from [our first competition](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data) (which we've included, and preprocessed, in this one) as part of your training and see what improvements that might lead to, or consider using information from some of the other labels in the datasets.
# 
# We're excited to see the ways you can implement or expand upon these ideas, as well as the new ones you'll come up with!

