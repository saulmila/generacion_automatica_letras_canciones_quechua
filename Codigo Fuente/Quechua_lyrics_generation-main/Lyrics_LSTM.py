#
# Quechua Lyrics Generation using Bidirectional LSTM:
# this proyect is based on the text generation proyects using LSTM
#
# based in text generation proyect focusing  on character sequences
#
# source :https://github.com/keras-team/keras/blob/master/examples/lstm_text_generation.py
#
# segun los tests: 20 epocas seran suficientes
# para tener un resultado decente.
# usamos un generado para evitar cargar toda la data
# en memoria (en entrenamiento y test).
# guardamos los pesos y el mdoelo en cada fin de epoca.
#
# Autores:
#     saul - UNSAAC saulvk8@gmail.com
#     mila - UNSAAC miraclemaza@gmail.com
#
# Compilacion:
#
# python Lyrics_LSTM.py data_collection.txt ejemplo.txt vocabulario.txt
#
#
# Descripcion:
#
# ejemplo.txt     : es el documento donde se guardan los ejemplos del test (de training)
# vocabulario.txt :  es el documento donde se guardan las palabras no repetidas(total)

from __future__ import print_function

#import tensorflow as tf

from tensorflow.keras.callbacks import LambdaCallback, ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, LSTM, Bidirectional
import numpy as np
import sys
import io
import os
import codecs


# Just disables the warning, doesn't take advantage of AVX/FMA to run faster
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# configuracion de variables globales
# para el experimento
#

SEQUENCE_LEN = 10
MIN_WORD_FREQUENCY = 3
STEP = 1
BATCH_SIZE = 32

# funcion guardar vocabulary
def print_vocabulary(words_file_path, words_set):
  words_file = codecs.open(words_file_path, 'w', encoding='utf8')
  for w in words_set:
    if w != "\n":
      words_file.write(w+"\n")
    else:
      words_file.write(w)
  words_file.close()

# Generador de palabras ONE-HOT ENCODDING
def generator(sentence_list, next_word_list, batch_size):
    index = 0
    while True:
        x = np.zeros((batch_size, SEQUENCE_LEN, len(words)), dtype=np.bool)
        y = np.zeros((batch_size, len(words)), dtype=np.bool)
        for i in range(batch_size):
            for t, w in enumerate(sentence_list[index % len(sentence_list)]):
                x[i, t, word_indices[w]] = 1
            y[i, word_indices[next_word_list[index % len(sentence_list)]]] = 1
            index = index + 1
        yield x, y


# Dividimos la data para el training y el test
def shuffle_and_split_training_set(sentences_original, next_original, percentage_test=2):
    # shuffle sentences
    print('combinacion de sentences')

    tmp_sentences = []
    tmp_next_word = []
    for i in np.random.permutation(len(sentences_original)):
        tmp_sentences.append(sentences_original[i])
        tmp_next_word.append(next_original[i])

    cut_index = int(len(sentences_original) * (1.-(percentage_test/100.)))
    x_train, x_test = tmp_sentences[:cut_index], tmp_sentences[cut_index:]
    y_train, y_test = tmp_next_word[:cut_index], tmp_next_word[cut_index:]

    print("Size de trainingset = %d" % len(x_train))
    print("Size de testingset = %d" % len(y_test))
    return (x_train, y_train), (x_test, y_test)

# vamos a implementar la version get_model
def get_model(dropout=0.2):
    print('Construyendo Modelo...')
    model = Sequential() #
    model.add(Bidirectional(LSTM(128), input_shape=(SEQUENCE_LEN, len(words))))
    if dropout > 0:
        model.add(Dropout(dropout))
    model.add(Dense(len(words)))
    model.add(Activation('softmax'))
    return model

# Functions from keras-team/keras/blob/master/examples/lstm_text_generation.py
def sample(preds, temperature=1.0):
    #  funcion q muestrea un indice de un array d probabilidad
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def on_epoch_end(epoch, logs):
    # funcion que se llama al final de cada epoca : escribe texto generado
    examples_file.write('\n----- Generating text after Epoch: %d\n' % epoch)

    # elegir y entregar una secuencia aleatoria
    seed_index = np.random.randint(len(sentences+sentences_test))
    seed = (sentences+sentences_test)[seed_index]

    for diversity in [0.3, 0.4, 0.5, 0.6, 0.7]:
        sentence = seed
        examples_file.write('----- Diversidad:' + str(diversity) + '\n')
        examples_file.write('----- Generando con seed:\n"' + ' '.join(sentence) + '"\n')
        examples_file.write(' '.join(sentence))

        for i in range(50):
            x_pred = np.zeros((1, SEQUENCE_LEN, len(words)))
            for t, word in enumerate(sentence):
                x_pred[0, t, word_indices[word]] = 1.

            preds = model.predict(x_pred, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_word = indices_word[next_index]

            sentence = sentence[1:]
            sentence.append(next_word)

            examples_file.write(" "+next_word)
        examples_file.write('\n')
    examples_file.write('='*80 + '\n')
    examples_file.flush()

# aqui vamos a llamar a todos nuestro codigo
if __name__=="__main__":

    print("Iniciando")
    # primero tenemos que llammar a compilacion

    # check arg de terminal
    if len(sys.argv) != 4:
        print('\033[91m' + 'Error de Argumento!\nUsage: python3 Lyrics_LSTM.py '
                           '<path_to_corpus> <examples_txt> <vocabulary_txt>' + '\033[0m')
        exit(1)
    if not os.path.isfile(sys.argv[1]):
        print('\033[91mERROR: ' + sys.argv[1] + ' no es un archivo!' + '\033[0m')
        exit(1)

    # asiganmos args de compilador a una variable
    corpus = sys.argv[1]
    examples = sys.argv[2]
    vocabulary = sys.argv[3]

    # comprobamos si existe una carpeta 'checkpoint'
    # si no existe creamos
    if not os.path.isdir('./checkpoints/'):
        os.makedirs('./checkpoints/')

    # leeemos el documento caracter por caracters
    with io.open(corpus, encoding='utf-8') as f:
        text = f.read().lower().replace('\n', ' \n ')
    print('Corpus length in characters:', len(text))

    # vamos a agrupar las palabras en base a un espacio
    text_in_words = [w for w in text.split(' ') if w.strip() != '' or w == '\n']
    print('Corpus length in words:', len(text_in_words))

    # Calculamos la frecuencia
    word_freq = {}
    for word in text_in_words:
        word_freq[word] = word_freq.get(word, 0) + 1

    ignored_words = set()
    for k, v in word_freq.items():
        if word_freq[k] < MIN_WORD_FREQUENCY:
            ignored_words.add(k)

    words = set(text_in_words)
    print('Palabras unicas antes de ingonar:', len(words))
    print('Ignorar palabras con frecuencia <', MIN_WORD_FREQUENCY)
    words = sorted(set(words) - ignored_words)
    print('palabras despues de ignorar:', len(words))

    # guardamos el conjunto de palabras no repetidas
    # y lo llamamos vocabulario
    print_vocabulary(vocabulary, words)

    # creamos indices para cada lista de palabras y hacemos un dict={key:"valor"}
    word_indices = dict((c, i) for i, c in enumerate(words))
    indices_word = dict((i, c) for i, c in enumerate(words))

    # cortamos el texto con  semi-redundant sequencias de la SEQUENCE_LEN words
    sentences = []
    next_words = []

    ignored = 0

    for i in range(0, len(text_in_words) - SEQUENCE_LEN, STEP):
        # Only add the sequences where no word is in ignored_words
        if len(set(text_in_words[i: i+SEQUENCE_LEN+1]).intersection(ignored_words)) == 0:
            sentences.append(text_in_words[i: i + SEQUENCE_LEN])
            next_words.append(text_in_words[i + SEQUENCE_LEN])
        else:
            ignored = ignored + 1
    print('Secuencias ingonaradas:', ignored)
    print('Secuencias restantes:', len(sentences))

    # x, y, x_test, y_test
    (train_sentences, train_next_words), (sentences_test, next_words_test) = shuffle_and_split_training_set(
        sentences, next_words
    )
    #print("sentences: ", train_sentences)
    #print("next_wrods: ", train_next_words)
    # print("sentences_test: ",sentences_test)
    # print("senteces _test_nextword: ", next_words_test)


    model = get_model()

    # load file model

    filename = "/home/wilderd/Documents/Lyrics_generation/Text_generation_me/saved_model.pb"

    model.load_weights(filename)
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    '''
    model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])

    file_path = "./checkpoints/LSTM_LYRICS-epoch{epoch:03d}-words%d-sequence%d-minfreq%d-" \
                "loss{loss:.4f}-val_loss{val_loss:.4f}" % \
                (len(words), SEQUENCE_LEN, MIN_WORD_FREQUENCY)
    # file_path = "./checkpoints/LSTM_LYRICS-epoch{epoch:03d}-words%d-sequence%d-minfreq%d-" \
    #             "loss{loss:.4f}-accuracy{accuracy:.4f}-val_loss{val_loss:.4f}-val_accuracy{val_accuracy:.4f}" % \
    #             (len(words), SEQUENCE_LEN, MIN_WORD_FREQUENCY)

    #print("file_path check points: ", file_path)

    checkpoint = ModelCheckpoint(file_path, monitor='val_accuracy', save_best_only=True)
    print_callback = LambdaCallback(on_epoch_end=on_epoch_end)
    # early_stopping = EarlyStopping(monitor='val_accuracy', patience=5)
    callbacks_list = [checkpoint, print_callback]#, early_stopping]

    examples_file = open(examples, "w")
    print("len(sentences_test)/ BATCH_SIZE",len(sentences_test), BATCH_SIZE)
    model.fit_generator(generator(sentences, next_words, BATCH_SIZE),
                        steps_per_epoch=int(len(sentences)/BATCH_SIZE) + 1,
                        epochs=60,
                        callbacks=callbacks_list,
                        validation_data=generator(sentences_test, next_words_test, BATCH_SIZE),
                        validation_steps=int(len(sentences_test)/BATCH_SIZE) + 1
                        )
'''
