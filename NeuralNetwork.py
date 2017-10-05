#usr/bin/python3
#
#
#
#install:
#keras,pandas,theano,h5py
#https://keras.io/backend/

import pandas as ps
import numpy as np
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense

def extract_labels(data,col):
    l = [(l, v) for v,l in zip(ps.get_dummies(data[col]).values.argmax(1),data[col])]
    l_dict = dict(l)
    return l_dict

def translate_results(y):
    if	type(y) is str:
        return True
    if  attack_labels['normal.'] == y:
        return False
    else:
        return True

if __name__=="__main__":
    #load data
    c1 = ps.read_csv('kddcup.data_10_percent_corrected', header=None)
    c1_nbr_rows = len(c1.index)
    c2 = ps.read_csv('corrected', header=None)
    c3 = c1.append(c2, ignore_index=True)

    #get labels and replace them in the train set and in the test set
    attack_labels = extract_labels(c1,41)
    protocol_labels = extract_labels(c3,1)
    service_labels = extract_labels(c3,2)
    flag_labels = extract_labels(c3,3)

    c1[41].replace(attack_labels, inplace=True)
    c1[1].replace(protocol_labels, inplace=True)
    c1[2].replace(service_labels, inplace=True)
    c1[3].replace(flag_labels, inplace=True)
    c2[41].replace(attack_labels, inplace=True)
    c2[1].replace(protocol_labels, inplace=True)
    c2[2].replace(service_labels, inplace=True)
    c2[3].replace(flag_labels, inplace=True)

    #Translate to numpy array
    train_data = np.array(c1)
    test_data = np.array(c2)

    #split the datasets isolating the last column

    X_train = train_data[:, 0:-1]
    Y_train = train_data[:,-1]
    Y_train = to_categorical(Y_train)	

    X_valid = test_data[:,0:-1]
    Y_valid = list(test_data[:,-1])

    model = Sequential()

    model.add(Dense(9, activation='tanh', input_dim=41))
    model.add(Dense(9, activation='tanh'))
    model.add(Dense(len(attack_labels), activation='softmax'))

    print("Compiling model...")

    model.compile(
        optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, Y_train)

    print("TRAINED!")
    print("Evaluation:")

    result = model.predict(X_valid)
    res_prop = [np.argmax(x) for x in result]

    print("Accuracy: ")

    print(
        len([True for y, y_proper in zip(res_prop, Y_valid)
        if translate_results(y) == translate_results(y_proper)]) / len(Y_valid))



