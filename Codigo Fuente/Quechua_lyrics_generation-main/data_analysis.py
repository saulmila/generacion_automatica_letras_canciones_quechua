from __future__ import print_function
# import punctuation symbols
from string import punctuation

import pandas
import sys
import io

# compilation
# python data_analysis.py data.txt

if __name__ == "__main__":
    # # check de argument...
    corpus = sys.argv[1]
    print("nombre de corpus(dataset): ", corpus)
    # Este pado paso lee todos los caracteres de entrada
    with io.open(corpus, encoding='utf-8')as f:
        text = f.read().lower().replace('\n',' \n ').replace('/','').replace('.','').replace(',','')
    #print('ejemplo 01: ', text[30])
    print('Corpus length in characters',len(text))

    # array de signos de puntuacion
    '''
    ['¡','!', '"', '#', '$', '%', '&', "'",""", '(', ')', '*', '+', ',', '-', '.', '/', ':', ';', '<', '-','=', '>', ,'¿','?', '@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~']

    '''
    non_words = list(punctuation)
    # agregamos simbolos usados en quechua
    non_words.extend(['¿','!'])
    # agregamos tmb numeros
    non_words.extend( map(str,range(10)) )
    # show the non_words to cleaning
    # print("NON WORDS: ",non_words)
    # NECESITAMOS UNA CUSTOM LIST PARA ELIMINAR LA DATA INECESARIA
    # counting how many words there is in all data_collection
    text_in_words = []
    for w in text.split(' '):
        if (w.strip()!='' or w =='\n'):
            # if(w!='//'):
            text_in_words.append(w)

    # text_in_words = [w for w in text.split(' ') if ( w.strip() !='' or w =='\n')]
    print("numero de palabras: length total ", len(text_in_words))
    print("ejemplo :", text_in_words[0:40])
    file_name = "data_q_saul.txt"
    f = open(file_name, "w")
    f.write(' '.join(text_in_words))
    f.close()
