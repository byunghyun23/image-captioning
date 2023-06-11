import click
import json
from glove import Corpus, Glove
import numpy as np
import pickle
from tensorflow.keras.preprocessing.text import text_to_word_sequence

@click.command()
@click.option('--json_file_name', default='MSCOCO_train_val_Korean.json', help='Json file name')
@click.option('--data_size', default=120000, help='Data size')
@click.option('--window', default=20, help='Embedding window')
@click.option('--no_components', default=200, help='Embedding dimension')
@click.option('--learning_rate', default=0.01, help='Embedding learning_rate')
@click.option('--epochs', default=50, help='Embedding epochs')
@click.option('--model_name', default='glove.model', help='Embedding model name')
def run(json_file_name, data_size, window, no_components, learning_rate, epochs, model_name):
    with open(json_file_name, 'r', encoding='UTF-8') as f:
        json_data = json.load(f)

    # Tokenization
    token = []
    for i in range(data_size):
        for j in json_data[i]['captions']:
            token.append(text_to_word_sequence(j))
    print('Length of token... ', len(token))

    # Glove embedding
    corpus = Corpus()
    corpus.fit(token, window=window)

    glove = Glove(no_components=no_components, learning_rate=learning_rate)
    glove.fit(corpus.matrix, epochs=epochs, no_threads=4, verbose=False)
    glove.add_dictionary(corpus.dictionary)
    glove.save(model_name)
    glove_model = Glove.load(model_name)

    word_dict = {}
    for word in glove_model.dictionary.keys():
        word_dict[word] = glove_model.word_vectors[glove_model.dictionary[word]]
    print('Length of word dict:', len(word_dict))

    embeddings_index = {}
    for word in glove_model.dictionary.keys():
        embeddings = word_dict[word]
        coefs = np.asarray(embeddings, dtype='float32')
        embeddings_index[word] = coefs

    with open('embeddings_index.pkl', 'wb') as f:
        pickle.dump(embeddings_index, f)

    information = {}
    information['embedding_dim'] = no_components
    with open('information.pkl', 'wb') as f:
        pickle.dump(information, f)


if __name__ == '__main__':
    run()