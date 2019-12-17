# -*- coding: utf-8 -*-
from __future__ import print_function
from data import *
from glove import *
import tensorflow as tf
#import tensorflow.nn.rnn_cell as rnn_cell
from tensorflow.contrib import rnn
from sklearn.model_selection import train_test_split
import glob
import numpy as np
import h5py
import keras

from random import random
from numpy import array
from numpy import cumsum
#from matplotlib import pyplot
from pandas import DataFrame

from keras.models import Sequential
from keras.layers import Bidirectional
from keras.losses import cosine_proximity   

from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from keras.layers.merge import concatenate
from keras.layers.merge import average
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import optimizers
from keras import regularizers
import keras.backend as K





import csv

from numpy.random import seed
seed(1)

tf.compat.v1.set_random_seed(2)


#from keras.losses import categorical_crossentropy

# rm old log files
for file in glob.glob('/home/matteo/tmp/tf.log/*'):
    os.remove(file)


# config
train_path = '/data/senseval2/eng-lex-sample.training.xml'
test_path = '/data/senseval2/eng-lex-samp.evaluation.xml'

# load data
#train_data = load_senteval2_data(train_path, True)
#test_data = load_senteval2_data(test_path, False)
train_data_ = load_train_data(23)
test_data = load_test_data(2)
print('Dataset size (train/test): %d / %d' % (len(train_data_), len(test_data)))

#print(train_data_)

mappasensi = {}
for istanza in train_data_:
    #print(istanza)
    mappasensi[istanza['id']] = istanza['target_sense']

#print(mappasensi)

EMBEDDING_DIM = 100
print('Embedding vector: %d' % EMBEDDING_DIM)

# build vocab utils
word_to_id = build_vocab(train_data_)
target_word_to_id, target_sense_to_id, n_words, n_senses_from_target_id = build_sense_ids(train_data_)
#print(word_to_id)
  
with open('gloveList.txt', 'w') as f:
    for key in word_to_id:
        f.write(key + '\n')
print('Vocabulary size: %d' % len(word_to_id))

#build context vocab of the target sense
train_target_sense_to_context = build_context(train_data_, word_to_id)

#build context embeddings of the target sense
embedding_matrix = fill_with_gloves(word_to_id, 100)
target_sense_to_context_embedding = build_embedding(train_target_sense_to_context, embedding_matrix, len(word_to_id), EMBEDDING_DIM)

#sense_embeddings_ = get_embeddingBIS("data/unito/LessLex.cut.csv")

#print(sense_embeddings_)


#with open('test.csv', 'w') as f:
#    for key in target_sense_to_context_embedding.keys():
        #print('[' + ' '.join(map(str, target_sense_to_context_embedding[key])) + ']')
#        f.write("%s\t%s\n"%(key,'[' + ' '.join(map(str, target_sense_to_context_embedding[key])) + ']'))

#with open('senseList.txt', 'w') as f:
#    for key in target_sense_to_context_embedding.keys():
#        #print('[' + ' '.join(map(str, target_sense_to_context_embedding[key])) + ']')
#        f.write(key + "\n")

#sense_embeddings_ = get_embeddingBIS("test.csv")

#print(sense_embeddings_)

# make numeric
train_ndata = convert_to_numeric(train_data_, word_to_id, target_word_to_id, target_sense_to_id, n_senses_from_target_id, target_sense_to_context_embedding, is_training = True)
test_ndata = convert_to_numeric(test_data, word_to_id, target_word_to_id, target_sense_to_id, n_senses_from_target_id, target_sense_to_context_embedding, is_training = False)
#train_ndata = convert_to_numeric(train_data_, word_to_id, target_word_to_id, target_sense_to_id, n_senses_from_target_id)
#test_ndata = convert_to_numeric(test_data, word_to_id, target_word_to_id, target_sense_to_id, n_senses_from_target_id)

#print(test_ndata[0])

n_step_f = 40
n_step_b = 40
print('n_step forward/backward: %d / %d' % (n_step_f, n_step_b))
MAX_SEQUENCE_LENGTH = 40
act = 'relu'
#STAMP = 'lstm_%d_%d_%.2f_%.2f'%(100, 2, 0.2, 0.5)

#bst_model_path = h5py.File("weights.best.hdf5", "w")

def cos_distance(y_true, y_pred):
    y_true = K.l2_normalize(y_true, axis=-1)
    y_pred = K.l2_normalize(y_pred, axis=-1)
    return K.mean(1 - K.sum((y_true * y_pred), axis=1)) #mettere 0 o 1

def cos_distance_bis(y_true, y_pred):
    def l2_normalize(x, axis):
        norm = K.sqrt(K.sum(K.square(x), axis=axis, keepdims=True))
        return K.sign(x) * K.maximum(K.abs(x), K.epsilon()) / K.maximum(norm, K.epsilon())
    y_true = l2_normalize(y_true, axis=-1)
    y_pred = l2_normalize(y_pred, axis=-1)
    return K.mean(y_true * y_pred, axis=-1)

def cos2(a, b):

    dot = np.dot(a, b)
    norma = np.linalg.norm(a)
    normb = np.linalg.norm(b)
    cos = dot / (norma * normb)

    #return 1-(np.dot(vA, vB) / (np.sqrt(np.dot(vA,vA)) * np.sqrt(np.dot(vB,vB))))
    return 1-cos

#####################################################################################################################
sense_embeddings_ = get_embeddingBIS("test.csv")
#####################################################################################################################

def own_model(train_forward_data, train_backward_data, train_sense_embedding, 
              val_forward_data=None, val_backward_data=None, val_sense_embedding=None,
              n_units=100, dense_unints=200, is_training=True, EMBEDDING_DIM=100, epochs=50, batch_size=2048, init_word_vecs=None):

    #is_training = False

    '''model = Sequential()
    model.add(Input(shape=(MAX_SEQUENCE_LENGTH, ), dtype='int32', name='forward_input'))
    model.add(Embedding(len(word_to_id),
                                EMBEDDING_DIM,
                                weights=[init_word_vecs],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False))
    model.add(LSTM(n_units, dropout=0.5, recurrent_dropout=0.5))
    #model.add(LSTM(n_units, dropout=0.5, recurrent_dropout=0.5))
    if use_dropout:
        model.add(Dropout(0.5))
        
    model.add(BatchNormalization())
    model.add(Dense(units=dense_unints, activation=act))
    if use_dropout:
        model.add(Dropout(0.5))
        
    model.add(BatchNormalization())
    model.add(Dense(units=EMBEDDING_DIM, activation=None))'''
    
    embedding_layer1 = Embedding(len(word_to_id),
                                EMBEDDING_DIM,
                                weights=[init_word_vecs],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False)
    '''embedding_layer2 = Embedding(len(word_to_id),
                                EMBEDDING_DIM,
                                weights=[init_word_vecs],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False)'''

    lstm_layer1 = LSTM(n_units, dropout=0.5, recurrent_dropout=0.5)
    lstm_layer2 = LSTM(n_units, dropout=0.5, recurrent_dropout=0.5)
    
    forward_input = Input(shape=(MAX_SEQUENCE_LENGTH, ), dtype='int32', name='forward_input')
    embedded_forward = embedding_layer1(forward_input)
    forward_lstm = lstm_layer1(embedded_forward)
    
    backward_input = Input(shape=(MAX_SEQUENCE_LENGTH, ), dtype='int32', name='backward_input')
    embedded_backward = embedding_layer1(backward_input)
    backward_lstm = lstm_layer2(embedded_backward)

    merged = average([forward_lstm, backward_lstm])     
    #merged = lstm_layer(merged)
           
    merged = Dropout(0.5)(merged) if is_training else merged
    merged = BatchNormalization()(merged)

    #merged = Dense(units=dense_unints, activation=act)(merged)
    #merged = Dense(units=dense_unints, activation=act, kernel_regularizer=regularizers.l2(0.1), activity_regularizer=regularizers.l1(0.1))(forward_lstm)
    #merged = Dropout(0.5)(merged) if is_training else merged
    #merged = BatchNormalization()(merged)

    #merged = Dense(units=dense_unints, activation=act)(merged)
    #merged = Dense(units=dense_unints, activation=act, kernel_regularizer=regularizers.l2(0.1), activity_regularizer=regularizers.l1(0.1))(forward_lstm)
    #merged = Dropout(0.5)(merged) if is_training else merged
    #merged = BatchNormalization()(merged)


    merged = Dense(units=dense_unints)(merged)
    #merged = Dense(units=dense_unints, activation=act, kernel_regularizer=regularizers.l2(0.1), activity_regularizer=regularizers.l1(0.1))(forward_lstm)
    #merged = Dropout(0.5)(merged) if is_training else merged
    #merged = BatchNormalization()(merged)

    
    preds = Dense(units=EMBEDDING_DIM, activation='softmax')(merged) #activation=None'''
    
    ## train the model 
    model = Model(inputs=[forward_input, backward_input], outputs=preds)
    #model = Model(inputs=[forward_input], outputs=preds)

    #loss = keras.losses.categorical_crossentropy(train_sense_embedding, preds, from_logits=True, label_smoothing=0)
    # Define custom loss
    def custom_loss():

        # Create a loss function that adds the MSE loss to the mean of all squared activations of a specific layer
        def loss(y_true,y_pred):
            loss = cos_distance(y_true,y_pred)
            #loss = keras.losses.cosine_proximity(y_true, y_pred, axis=1)
            return loss

            '''minimo_true = list(range(10000.0,10000.0,len(y_true)))
            minimo_pred = list(range(10000.0,10000.0,len(y_pred)))
            minimo_true_vec = list(str(' ') * len(y_true))
            minomo_pred_vec = list(str(' ') * len(y_pred))

            indice = -1

            for h in y_true:
            	indice = indice + 1
            	for j in sense_embeddings_:
            		d = cos2(sense_embeddings_[j],h)
                        if d < minimo_true[indice]:
                            minimo_true[indice] = d
                            minimo_true_vec[indice] = j

            indice = -1

            for z in y_pred:
            	indice = indice + 1
            	for j in sense_embeddings_:
            		d = cos2(sense_embeddings_[j],z)
                        if d < minimo_pred[indice]:
                            minimo_pred[indice] = d
                            minimo_pred_vec[indice] = j'''

            #print_op = tf.print("veri: \n", y_true, "\n" "e predetti: \n" ,y_pred)
            #with tf.control_dependencies([print_op]):
                #return K.identity(loss)	
       
        # Return a function
        return loss

    

    #nadam = optimizers.Adam(clipnorm=1.) #, clipvalue=0.5
    #model.compile(loss=custom_loss(), optimizer=nadam)
    model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
    #model.compile(loss=cos_distance(train_sense_embedding,preds), optimizer=nadam)
    #model.compile(loss=keras.losses.cosine_proximity(train_sense_embedding, preds), optimizer=nadam)
    
    
    #early_stopping =EarlyStopping(monitor='val_loss', patience=10)
    #bst_model_path = STAMP + '.h5'
    bst_model_path = "weights.best.hdf5"
    model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True, save_weights_only=True, verbose=1)

    print(model.summary())
    #model.load_weights(bst_model_path)
      
    '''hist = model.fit([train_forward_data, train_backward_data], train_sense_embedding, 
                     validation_data=([val_forward_data, val_backward_data], val_sense_embedding), 
                     epochs=epochs, batch_size=batch_size, shuffle=True, 
                     callbacks=[model_checkpoint])'''
    hist = model.fit([train_forward_data, train_backward_data], train_sense_embedding,
    #hist = model.fit(train_forward_data, train_sense_embedding, 
                     validation_split=0.3, 
                     epochs=epochs, batch_size=batch_size, shuffle=True, 
                     callbacks=[model_checkpoint])
    
    model.load_weights(bst_model_path)
    #bst_val_score = min(hist.history['val_loss'])
    #print(hist.history.keys())
    #print(hist)
    
    #print('min val loss is: %f' % (bst_val_score))

    print('___________________________________________________')


    conf = {
        'batch_size': 100,
        'n_step_f': 40,
        'n_step_b': 40,
        'n_lstm_units': 74,
        'n_layers': 1,
        'emb_base_std': 0.2,
        'input_keep_prob': 0.5,
        'keep_prob': 0.5,
        'embedding_size': 100,
        'train_embeddings': True,
        #'forget_bias': 0.0,
        'state_size': 200,
        'train_init_state': False,
        'permute_input_order': False,
        'word_drop_rate': 0.1,
        'w_penalty': False,
        'freeze_emb_n_iter': 0
    }


    class Answer:
        pass

    correct = 0
    counted = 0
    correct_b = 0
    counted_b = 0
    print ('Evaluating')
    result = []
    numero = 0
    for batch_id, batch in enumerate(batch_generator(False, conf['batch_size'], test_ndata, word_to_id['<pad>'], conf['n_step_f'], conf['n_step_b'], pad_last_batch=True)):
        xfs, xbs, target_ids, sense_ids, instance_ids = batch

        print(xfs)
        #feed = {
        #    model.inputs_f: xfs,
        #    model.inputs_b: xbs,
        #    model.train_target_ids: target_ids,
        #    model.train_sense_ids: sense_ids
        #}
        #print(target_ids)
        #print(len(target_ids))
        #print(len(xfs))
        #print(len(xbs))
        predictions = model.predict([xfs, xbs], verbose=1, batch_size=conf['batch_size'])
        #predictions = model.predict(xfs)
        print(predictions)

        #predictions = session.run(model.predictions, feed_dict=feed)
        lexelts = get_lexelts(2)
        target_id_to_word = {id: word for (word, id) in target_word_to_id.iteritems()}
        target_word_to_lexelt = target_to_lexelt_map(target_word_to_id.keys(), lexelts)
        if 'colourless' in target_word_to_lexelt:
            target_word_to_lexelt['colorless'] = target_word_to_lexelt['colourless']
        target_id_to_sense_id_to_sense = [{sense_id: sense for (sense, sense_id) in sense_to_id.iteritems()} for (target_id, sense_to_id) in enumerate(target_sense_to_id)]

        #print(target_id_to_sense_id_to_sense)
        #sense_embeddings_ = get_embedding(sense_embedding_file)
        keys = get_keys("Senseval2.csv")

        #print("##########################################################")
        #print(keys)
        sense_embeddings_ = get_embeddingBIS("test.csv")

        for i, predicted_sense_emb in enumerate(predictions):
            
            if batch_id * conf['batch_size'] + i < len(test_ndata):
                a = Answer()
                a.target_word = target_id_to_word[target_ids[i]]
                a.lexelt = target_word_to_lexelt[a.target_word]
                a.instance_id = instance_ids[i]
                minimo_v = 1000.00
                minimo = np.float64(minimo_v)
                minimo_vec = 'None'
                for j in sense_embeddings_:
                    parola = str(a.instance_id).split(".")[0]
                    #print("la parola da trovare è " + parola)
                    #print(j)
                    if str(j).find(parola) != -1:
                        d = cos2(sense_embeddings_[j],predicted_sense_emb)
                        if d < minimo:
                            minimo = d
                            minimo_vec = j
                    #for h in word_to_id:
                    #    if word_to_id
                    #toconfront = 
                a.predicted_sense = minimo_vec
                a.distanza = minimo
                result.append(a)
                '''print(a.target_word)
                print(a.lexelt)
                print(a.instance_id)'''
                #print(a.predicted_sense)

        print ('Writing to file')
        path = './tmp/result_' + str(numero)
        numero += 1
        with open(path, 'w') as file:
            for a in result:
                #first = a.lexelt if se_to_eval == 3 else a.target_word
                counted = counted + 1
                counted_b = counted_b + 1
                if str(keys[str(a.instance_id)]).find(str(a.predicted_sense)) != -1:
                    correct = correct + 1
                    correct_b = correct_b + 1
                first = a.lexelt
                file.write('%s %s %s %f %s\n' % (first, a.instance_id, a.predicted_sense, a.distanza, keys[str(a.instance_id)]))

        result = []
        print("corretti " + str(correct_b) + " su " + str(counted_b))
        correct_b = 0
        counted_b = 0

    print("corretti " + str(correct) + " su " + str(counted))


if __name__ == '__main__':

    grouped_by_target = group_by_target(train_ndata)
    train_data_b, val_data = split_grouped(grouped_by_target, frac=0.2)

    print('ALLORA')
    print("---------------------------------")
    print(len(val_data))


    init_emb = fill_with_gloves(word_to_id, EMBEDDING_DIM)
    
    train_forward_data, train_backward_data, train_target_sense_ids, train_sense_embedding = get_data(train_data_b, n_step_f, n_step_b)
    val_forward_data, val_backward_data, val_target_sense_ids, val_sense_embedding = get_data(val_data, n_step_f, n_step_b)
        
    own_model(train_forward_data, train_backward_data, train_sense_embedding, 
              val_forward_data, val_backward_data, val_sense_embedding, 
              init_word_vecs=init_emb)
    