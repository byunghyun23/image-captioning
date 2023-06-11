import click
import json
from tensorflow.keras.preprocessing.text import text_to_word_sequence
import os
from time import time
from tqdm import tqdm
import pickle
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.python.keras.applications.inception_v3 import InceptionV3
from tensorflow.python.keras.models import Model


def permutation_train_test_split(data_list, test_size=0.2, shuffle=True, random_state=1004):
    X = np.array(data_list)
    test_num = int(X.shape[0] * test_size)
    train_num = X.shape[0] - test_num

    if shuffle:
        np.random.seed(random_state)
        shuffled = np.random.permutation(X.shape[0])
        X = X[shuffled]
        X_train = X[:train_num]
        X_test = X[train_num:]
    else:
        X_train = X[:train_num]
        X_test = X[train_num:]

    return X_train.tolist(), X_test.tolist()


def hms_string(sec_elapsed):
    h = int(sec_elapsed / (60 * 60))
    m = int((sec_elapsed % (60 * 60)) / 60)
    s = sec_elapsed % 60

    return f'{h}:{m:>02}:{s:>05.2f}'


def encode_image(img, width, height, encode_model, preprocess_input, output_dim):
    img = img.resize((width, height), Image.ANTIALIAS)
    x = tf.keras.preprocessing.image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    x = encode_model.predict(x)
    x = np.reshape(x, output_dim)

    return x


@click.command()
@click.option('--json_file_name', default='MSCOCO_train_val_Korean.json', help='Json file name')
@click.option('--data_dir', default='data/', help='Data directory')
@click.option('--data_size', default=120000, help='Data size')
@click.option('--test_size', default=0.2, help='Test size')
def run(json_file_name, data_dir, data_size, test_size):
    with open(json_file_name, 'r', encoding='UTF-8') as f:
        json_data = json.load(f)

    # Split data to train and test
    train_json, test_json = permutation_train_test_split(json_data[:data_size], test_size)

    # Captioning
    max_len = 0
    train_img = []
    train_lookup = dict()
    for i in range(len(train_json)):
        id = train_json[i]['file_path'][-16:-4]
        train_img.append(train_json[i]['file_path'][-16:])

        caption_list = []
        for caption in train_json[i]['captions']:
            # new_caption = okt.nouns(caption)
            new_caption = text_to_word_sequence(caption)
            max_len = max(max_len, len(new_caption))
            new_caption = ' '.join(new_caption)
            caption_list.append(new_caption)
        train_lookup[id] = caption_list
    # lex = set(word)

    test_img = []
    test_lookup = dict()
    for i in range(len(test_json)):
        id = test_json[i]['file_path'][-16:-4]
        test_img.append(test_json[i]['file_path'][-16:])

        caption_list = []
        for caption in test_json[i]['captions']:
            new_caption = text_to_word_sequence(caption)
            max_len = max(max_len, len(new_caption))
            new_caption = ' '.join(new_caption)
            caption_list.append(new_caption)
        test_lookup[id] = caption_list

    # print(max_len)
    # print(len(train_lookup))
    # print(train_lookup['000000391895'])

    # Labeling
    train_captions = {}
    for key, value in train_lookup.items():
        caption_list = []

        for v in value:
            caption_list.append(f'startseq {v} endseq')
        train_captions[key] = caption_list
    # print(train_captions['000000391895'])

    all_train_captions = []
    for key, val in train_captions.items():
        for cap in val:
            all_train_captions.append(cap)
    # print(len(all_train_captions))

    test_captions = {}
    for key, value in test_lookup.items():
        caption_list = []

        for v in value:
            caption_list.append(f'startseq {v} endseq')
        test_captions[key] = caption_list

    with open('train_captions.pkl', 'wb') as f:
        pickle.dump(train_captions, f)
    with open('test_captions.pkl', 'wb') as f:
        pickle.dump(test_captions, f)

    # Exclude words with frequencies less than 10
    word_count_threshold = 10
    word_counts = {}
    for caption in all_train_captions:
        for word in caption.split(' '):
            word_counts[word] = word_counts.get(word, 0) + 1

    vocab = [word for word in word_counts if word_counts[word] >= word_count_threshold]
    print('preprocessed words %d ==> %d' % (len(word_counts), len(vocab)))

    # Indexing
    idx_to_word = {}  # {index:word}
    word_to_idx = {}  # {word:index}

    idx = 1
    for w in vocab:
        word_to_idx[w] = idx
        idx_to_word[idx] = w
        idx += 1

    vocab_size = len(idx_to_word) + 1
    max_len += 2  # include start, end token
    print(max_len)
    print(word_to_idx)

    with open('idx_to_word.pkl', 'wb') as f:
        pickle.dump(idx_to_word, f)
    with open('word_to_idx.pkl', 'wb') as f:
        pickle.dump(word_to_idx, f)

    # Encoding
    encode_model = InceptionV3(weights='imagenet')
    encode_model = Model(encode_model.input, encode_model.layers[-2].output)
    width = 299
    height = 299
    output_dim = 2048
    preprocess_input = tf.keras.applications.inception_v3.preprocess_input
    # encode_model.summary()

    train_path = 'train_encoding.pkl'
    if not os.path.exists(train_path):
        start = time()
        train_encoding = {}

        for id in tqdm(train_img):
            image_path = data_dir + id
            img = tf.keras.preprocessing.image.load_img(image_path,
                                                        target_size=(height, width))
            train_encoding[id] = encode_image(img, width, height, encode_model, preprocess_input, output_dim)

        with open(train_path, 'wb') as f:
            pickle.dump(train_encoding, f)
        print(f'Generating training set took:, {hms_string(time() - start)}')
    else:
        with open(train_path, 'rb') as f:
            train_encoding = pickle.load(f)

    test_path = 'test_encoding.pkl'
    if not os.path.exists(test_path):
        start = time()
        test_encoding = {}

        for id in tqdm(test_img):
            image_path = data_dir + id
            img = tf.keras.preprocessing.image.load_img(image_path,
                                                        target_size=(height, width))
            test_encoding[id] = encode_image(img, width, height, encode_model, preprocess_input, output_dim)

        with open(test_path, 'wb') as fp:
            pickle.dump(test_encoding, fp)
        print(f'Generating testing set took:, {hms_string(time() - start)}')
    else:
        with open(test_path, 'rb') as fp:
            test_encoding = pickle.load(fp)

    information = {}
    with open('information.pkl', 'rb') as f:
        try:
            info = pickle.load(f)
            information['embedding_dim'] = info['embedding_dim']
        except Exception as e:
            pass

    information['output_dim'] = output_dim
    information['vocab_size'] = vocab_size
    information['max_len'] = max_len
    information['width'] = width
    information['height'] = height
    information['preprocess_input'] = preprocess_input

    with open('information.pkl', 'wb') as f:
        pickle.dump(information, f)


if __name__ == '__main__':
    run()