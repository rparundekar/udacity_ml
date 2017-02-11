#Define some useful Functions
import h5py
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range
import matplotlib.pyplot as plt
import cv2
import math
import random

#CONSTANTS
imgSize = 54

def get_box_data(index, hdf5_data):
    """
    get `left, top, width, height` of each picture
    :param index:
    :param hdf5_data:
    :return:
    """
    meta_data = dict()
    meta_data['height'] = []
    meta_data['label'] = []
    meta_data['left'] = []
    meta_data['top'] = []
    meta_data['width'] = []

    def print_attrs(name, obj):
        vals = []
        if obj.shape[0] == 1:
            vals.append(obj[0][0])
        else:
            for k in range(obj.shape[0]):
                vals.append(int(hdf5_data[obj[k][0]][0][0]))
        meta_data[name] = vals

    box = hdf5_data['/digitStruct/bbox'][index]
    hdf5_data[box[0]].visititems(print_attrs)
    return meta_data

def get_name(index, hdf5_data):
    name = hdf5_data['/digitStruct/name']
    return ''.join([chr(v[0]) for v in hdf5_data[name[index][0]].value])

def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])

def oneHot(num, length):
    arr = np.zeros(length)
    arr[num-1]=1
    return arr
def maybeLoadData(folder, variations):
    import os.path
    import pickle
    file_path=folder + '.pk'
    if(os.path.exists(file_path) is False):
        imageData, imageLengths, imageDigits = loadData(folder, variations)
        data = { 'imageData': imageData, 'imageLengths': imageLengths, 'imageDigits': imageDigits}
        pickle.dump(data , open( file_path, "wb" ))

    data = pickle.load( open( file_path, "rb" ) );
    return data['imageData'], data['imageLengths'], data['imageDigits']

def loadData(folder, variations):
    #First load the data using h5py
    f = h5py.File(folder + '/' + 'digitStruct.mat')
    #Get the number of images to iterate through them
    length = len(f['/digitStruct/name'])

    #length = 10;   #TestLength

    imageData = np.zeros([length, imgSize,imgSize,1]).astype(np.float32)
    imageLengths = np.zeros([length, 5]).astype(np.int)
    imageDigits = np.zeros([length,5,11]).astype(np.int)

    #Iterate through the images
    for i in range(0,length):
        if(i%500==0): #In case of error, comment this line
            print("Loaded {} out of {}".format(i,length))

        #Read the image
        imageFile = folder + '/' + get_name(i,f)
        img = cv2.imread(imageFile)

        #Read the box data & get the bounding box for all characters (using first and last digit)
        boxData=get_box_data(i, f)

        firstTop = int(boxData['top'][0])
        firstLeft = int(boxData['left'][0])
        firstRight = int(boxData['left'][0]) + int(boxData['width'][0])
        firstBottom = int(boxData['top'][0]) + int(boxData['height'][0])

        l = len(boxData['top'])
        lastTop = int(boxData['top'][l-1])
        lastLeft = int(boxData['left'][l-1])
        lastRight = int(boxData['left'][l-1]) + int(boxData['width'][l-1])
        lastBottom = int(boxData['top'][l-1]) + int(boxData['height'][l-1])

        top = min(firstTop, lastTop)
        left = min(firstLeft, lastLeft)
        right = max(firstRight, lastRight)
        bottom = max(firstBottom, lastBottom)

        height = bottom-top
        width = right-left
        vertMiddle = (bottom+top)//2
        horCenter = (left+right)//2

        if(variations==True):
            top = vertMiddle - ((1.3*height)//2)
            bottom = vertMiddle + ((1.3*height)//2)
            left = horCenter - ((1.3*width)//2)
            right = horCenter + ((1.3*width)//2)

        top = max(top, 0)
        left = max(left, 0)
        right = min(right, img.shape[1])
        bottom = min(bottom, img.shape[0])

            #One image has incorrect label length of  6
#         if(len(boxData['label'])>5):
#             cv2.imshow('image',img)
#             cv2.waitKey(0)

        #Check to see the bounding box
        #cv2.rectangle(img,(left,top),(right, bottom),(0,255,0),3)

        #Extract only the RoI for faster pre-processing
        img = img[top:bottom, left:right]

        #Convert to gray scale if in color
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        #Histogram correction
#         clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(6,6))
#         img = clahe.apply(img)

        #Length of digits
        numberOfDigits = len(boxData['label'])

        #Save the original for comparison if needed
        #orig = cv2.resize(img,(imgSize, imgSize), interpolation = cv2.INTER_LANCZOS4)

        #Enhance edges
#         edge_enhancement_kernel = np.array([[-1,-1,-1,-1,-1],
#                              [-1,2,2,2,-1],
#                              [-1,2,8,2,-1],
#                              [-1,2,2,2,-1],
#                              [-1,-1,-1,-1,-1]]) / 8.0


#         img = cv2.filter2D(img, -1, edge_enhancement_kernel)
#         img = cv2.filter2D(img, -1, edge_enhancement_kernel)

        #Resize the image to 64x64
        if(variations==True):
            img = cv2.resize(img,(64, 64), interpolation = cv2.INTER_LANCZOS4)
            leftStart=random.randint(0,9)
            topStart=random.randint(0,9)
            img = img[topStart:(topStart+imgSize), leftStart:(leftStart+imgSize)]
        else:
            img = cv2.resize(img,(imgSize, imgSize), interpolation = cv2.INTER_LANCZOS4)

        #Copy the data
        oneImageData = np.resize(img, (imgSize,imgSize,1)).astype(np.float32)

        oneImageData=oneImageData/255.0
        oneImageData=oneImageData-0.5

        imageData[i] = oneImageData
        first=0
        if(numberOfDigits>5):
            numberOfDigits=5
            print(boxData['label'])
            first=1

        imageLengths[i] = oneHot(numberOfDigits,5)

        for k in range(0,5):
            if(k<numberOfDigits):
                imageDigits[i,k,:]=oneHot(int(boxData['label'][int(k+first)]),11)
            else:
                imageDigits[i,k,10]=1


        #Show the original image
        #cv2.imshow('image',orig)
        #cv2.waitKey(0)

        #Show the processed image
        #cv2.imshow('image',img)
        #cv2.waitKey(0)

    shuffledIndexes  = np.arange(length)
    np.random.shuffle(shuffledIndexes)

    imageData = imageData[shuffledIndexes,:]
    imageLengths = imageLengths[shuffledIndexes,:]
    imageDigits = imageDigits[shuffledIndexes,:,:]
    return imageData,imageLengths, imageDigits

print("Loading data")
trainImageData, trainImageLengths,trainImageDigits = maybeLoadData('train', True)
trainImageData = trainImageData.reshape((-1,54,54,1))
print("Training data images: {}".format(trainImageData.shape))
print("              length: {}".format(trainImageLengths.shape))
print("              digits: {}".format(trainImageDigits.shape))
extraImageData, extraImageLengths,extraImageDigits = maybeLoadData('extra', True)
print("Extra data    images: {}".format(extraImageData.shape))
print("              length: {}".format(extraImageLengths.shape))
print("              digits: {}".format(extraImageDigits.shape))

trainImageData  = np.concatenate((trainImageData, extraImageData))
trainImageLengths  = np.concatenate((trainImageLengths, extraImageLengths))
trainImageDigits  = np.concatenate((trainImageDigits, extraImageDigits))
print("Training data images: {}".format(trainImageData.shape))
print("              length: {}".format(trainImageLengths.shape))
print("              digits: {}".format(trainImageDigits.shape))


print("Loading test & validation data")
folderImageData, folderImageLengths,folderImageDigits = maybeLoadData('test', False)
folderImageData=folderImageData.reshape((-1,54,54,1))
print("Folder test data images: {}".format(folderImageData.shape))
print("          length: {}".format(folderImageLengths.shape))
print("          digits: {}".format(folderImageDigits.shape))

half = len(folderImageData)//2
validationImageData = folderImageData[0:half,:]
validationImageLengths = folderImageLengths[0:half,:]
validationImageDigits= folderImageDigits[0:half,:,:]
print("Validation data images: {}".format(validationImageData.shape))
print("                length: {}".format(validationImageLengths.shape))
print("                digits: {}".format(validationImageDigits.shape))

testImageData = folderImageData[half:,:]
testImageLengths = folderImageLengths[half:,:]
testImageDigits= folderImageDigits[half:,:,:]
print("Test data images: {}".format(testImageData.shape))
print("          length: {}".format(testImageLengths.shape))
print("          digits: {}".format(testImageDigits.shape))

print("Data loaded.")

#Confirm Data
d=6135
#plt.imshow(trainImageData[d].reshape((imgSize,imgSize)), cmap='gray')
#plt.show()
print(trainImageLengths[d])
print(trainImageDigits[d])

trainDigit0  = trainImageDigits[:,0,:]
trainDigit1  = trainImageDigits[:,1,:]
trainDigit2  = trainImageDigits[:,2,:]
trainDigit3  = trainImageDigits[:,3,:]
trainDigit4  = trainImageDigits[:,4,:]

print(trainDigit0[d])
print(trainDigit1[d])
print(trainDigit2[d])
print(trainDigit3[d])
print(trainDigit4[d])

from keras.models import Model
from keras.layers import Dense, Activation, Reshape, AveragePooling2D, MaxPooling2D, Input, Flatten, merge, Convolution2D, Dropout, LocallyConnected2D
from keras.regularizers import l2
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU

inputSize=imgSize*imgSize
num_labels=5

batch_size = 100
def getCnnModel():
    x = Input(batch_shape=(None, imgSize, imgSize,1))

    conv = Convolution2D(48, 5, 5,border_mode='same', W_regularizer=l2(0.01))(x)
    conv = PReLU()(conv)
    conv = MaxPooling2D(pool_size = (2,2), strides=(1,1))(conv)
    conv = BatchNormalization()(conv)
    conv = Dropout(0.5)(conv)

    conv = Convolution2D(64, 5,5,border_mode='same', W_regularizer=l2(0.01))(conv)
    conv = PReLU()(conv)
    #conv = Activation('relu')(conv)
    conv = AveragePooling2D(pool_size = (2,2), strides=(2,2))(conv)
    conv = BatchNormalization()(conv)
    conv = Dropout(0.5)(conv)

    conv = Convolution2D(128, 5,5, border_mode='same', W_regularizer=l2(0.01))(conv)
    conv = PReLU()(conv)
    #conv = Activation('relu')(conv)
    conv = MaxPooling2D(pool_size = (2,2), strides=(1,1))(conv)
    conv = BatchNormalization()(conv)
    conv = Dropout(0.5)(conv)

    conv = Convolution2D(160,5,5, border_mode='same', W_regularizer=l2(0.01))(conv)
    conv = PReLU()(conv)
    #conv = Activation('relu')(conv)
    conv = AveragePooling2D(pool_size = (2,2), strides=(2,2))(conv)
    conv = BatchNormalization()(conv)
    conv = Dropout(0.5)(conv)

    conv = Convolution2D(192,5,5, border_mode='same', W_regularizer=l2(0.01))(conv)
    #conv = Activation('relu')(conv)
    conv = PReLU()(conv)
    conv = MaxPooling2D(pool_size = (2,2), strides=(1,1))(conv)
    conv = BatchNormalization()(conv)
    conv = Dropout(0.5)(conv)

    conv = Convolution2D(192,5,5, border_mode='same', W_regularizer=l2(0.01))(conv)
    #conv = Activation('relu')(conv)
    conv = PReLU()(conv)
    conv = AveragePooling2D(pool_size = (2,2), strides=(2,2))(conv)
    conv = BatchNormalization()(conv)
    conv = Dropout(0.5)(conv)

    conv = Convolution2D(192,5,5,border_mode='same', W_regularizer=l2(0.01))(conv)
    #conv = Activation('relu')(conv)
    conv = PReLU()(conv)
    conv = MaxPooling2D(pool_size = (2,2), strides=(1,1))(conv)
    conv = BatchNormalization()(conv)
    conv = Dropout(0.5)(conv)

    conv = Convolution2D(192,5,5,border_mode='same', W_regularizer=l2(0.01))(conv)
    #conv = Activation('relu')(conv)
    conv = PReLU()(conv)
    conv = AveragePooling2D(pool_size = (2,2), strides=(2,2))(conv)
    conv = BatchNormalization()(conv)
    conv = Dropout(0.5)(conv)

    conv = Convolution2D(192,5,5, W_regularizer=l2(0.01), border_mode='same')(conv)
    conv = PReLU()(conv)
    conv = MaxPooling2D(pool_size = (2,2), strides=(1,1))(conv)
    #conv = Activation('relu')(conv)
    conv = BatchNormalization()(conv)
    conv = Dropout(0.5)(conv)

    flat = Flatten()(conv)

    dense = Dense(3072, W_regularizer=l2(0.01))(flat)
    #dense = Activation('relu')(dense)
    dense = PReLU()(dense)
    dense = BatchNormalization()(dense)
    dense = Dropout(0.5)(dense)
    dense = Dense(3072, W_regularizer=l2(0.01))(dense)
    dense = PReLU()(dense)
    #dense = Activation('relu')(dense)
    dense = BatchNormalization()(dense)
    dense = Dropout(0.5)(dense)
    dense = Dense(3074, W_regularizer=l2(0.01))(dense)
    dense = PReLU()(dense)
    #dense = Activation('relu')(dense)
    dense = BatchNormalization()(dense)
    dense = Dropout(0.5)(dense)
    outL = Dense(5, W_regularizer=l2(0.01))(dense)
    outL = Activation('softmax', name="Length")(outL)

    #merged = merge([outL, dense], mode='concat')
    #merged = Dense(1024, W_regularizer=l2(0.01))(merged)
    #merged = Activation('relu')(merged)

    outD0 = Dense(11, W_regularizer=l2(0.01))(dense)
    outD0 = Activation('softmax', name="Digit0")(outD0)
    outD1 = Dense(11, W_regularizer=l2(0.01))(dense)
    outD1 = Activation('softmax', name="Digit1")(outD1)
    outD2 = Dense(11, W_regularizer=l2(0.01))(dense)
    outD2 = Activation('softmax', name="Digit2")(outD2)
    outD3 = Dense(11, W_regularizer=l2(0.01))(dense)
    outD3 = Activation('softmax', name="Digit3")(outD3)
    outD4 = Dense(11, W_regularizer=l2(0.01))(dense)
    outD4 = Activation('softmax', name="Digit4")(outD4)
    model = Model(input=x, output=[outL, outD0, outD1, outD2, outD3, outD4])
    return model


def strongLengthBias():
    x = Input(batch_shape=(None, imgSize, imgSize,1))
    #flat = Flatten()(x)


    conv = Convolution2D(10, 5, 5, border_mode='same', W_regularizer=l2(0.05))(x)
    #conv = Activation('relu')(conv)
    conv = MaxPooling2D(pool_size = (2,2), strides=(2,2))(conv)
    conv = BatchNormalization()(conv)
    conv = Dropout(0.4)(conv)
    conv = Convolution2D(20, 5,5, border_mode='same', W_regularizer=l2(0.05))(conv)
    #conv = Activation('relu')(conv)
    conv = MaxPooling2D(pool_size = (2,2), strides=(2,2))(conv)
    conv = BatchNormalization()(conv)
    conv = Dropout(0.4)(conv)
    conv = Convolution2D(30,5,5, border_mode='same', W_regularizer=l2(0.05))(conv)
    #conv = Activation('relu')(conv)
    conv = MaxPooling2D(pool_size = (2,2), strides=(2,2))(conv)
    conv = BatchNormalization()(conv)
    flat = Flatten()(conv)
    dense = Dropout(0.4)(flat)
    dense = Dense(1024, W_regularizer=l2(0.05))(dense)
    dense = BatchNormalization()(dense)
    dense = Activation('relu')(dense)
    dense = Dropout(0.4)(dense)
    dense = Dense(1024, W_regularizer=l2(0.05))(dense)
    dense = BatchNormalization()(dense)
    dense = Activation('relu')(dense)
    outL = Dense(5, W_regularizer=l2(0.05))(dense)
    outL = Activation('softmax', name="Length")(outL)

    conv = Convolution2D(10,5,5, border_mode='same', W_regularizer=l2(0.05))(x)
    #conv = Activation('relu')(conv)
    conv = MaxPooling2D(pool_size = (2,2), strides=(2,2))(conv)
    conv = BatchNormalization()(conv)
    conv = Dropout(0.4)(conv)
    conv = Convolution2D(20,5,5, border_mode='same', W_regularizer=l2(0.05))(conv)
    #conv = Activation('relu')(conv)
    conv = MaxPooling2D(pool_size = (2,2), strides=(2,2))(conv)
    conv = BatchNormalization()(conv)
    conv = Dropout(0.4)(conv)
    conv = Convolution2D(30,5,5, border_mode='same', W_regularizer=l2(0.05))(conv)
    #conv = Activation('relu')(conv)
    conv = MaxPooling2D(pool_size = (2,2), strides=(2,2))(conv)
    conv = BatchNormalization()(conv)
    merged0 = Flatten()(conv)
    merged0 = merge([outL, merged0], mode='concat')
    merged0 = Dropout(0.4)(merged0)
    merged0 = Dense(1024, W_regularizer=l2(0.05))(merged0)
    merged0 = BatchNormalization()(merged0)
    merged0 = Activation('relu')(merged0)
    merged0 = Dropout(0.4)(merged0)
    merged0 = Dense(1024, W_regularizer=l2(0.05))(merged0)
    merged0 = BatchNormalization()(merged0)
    merged0 = Activation('relu')(merged0)
    outD0 = Dense(11, W_regularizer=l2(0.05))(merged0)
    outD0 = Activation('softmax', name="Digit0")(outD0)

    conv = Convolution2D(10,5,5, border_mode='same', W_regularizer=l2(0.05))(x)
    #conv = Activation('relu')(conv)
    conv = MaxPooling2D(pool_size = (2,2), strides=(2,2))(conv)
    conv = BatchNormalization()(conv)
    conv = Dropout(0.4)(conv)
    conv = Convolution2D(20,5,5, border_mode='same', W_regularizer=l2(0.05))(conv)
    #conv = Activation('relu')(conv)
    conv = MaxPooling2D(pool_size = (2,2), strides=(2,2))(conv)
    conv = BatchNormalization()(conv)
    conv = Dropout(0.4)(conv)
    conv = Convolution2D(30,5,5, border_mode='same', W_regularizer=l2(0.05))(conv)
    #conv = Activation('relu')(conv)
    conv = MaxPooling2D(pool_size = (2,2), strides=(2,2))(conv)
    conv = BatchNormalization()(conv)
    merged1 = Flatten()(conv)
    merged1 = merge([outL, merged1], mode='concat')
    merged1 = Dropout(0.4)(merged1)
    merged1 = Dense(1024, W_regularizer=l2(0.05))(merged1)
    merged1 = BatchNormalization()(merged1)
    merged1 = Activation('relu')(merged1)
    merged1 = Dropout(0.4)(merged1)
    merged1 = Dense(1024, W_regularizer=l2(0.05))(merged1)
    merged1 = BatchNormalization()(merged1)
    merged1 = Activation('relu')(merged1)
    outD1 = Dense(11, W_regularizer=l2(0.05))(merged1)
    outD1 = Activation('softmax', name="Digit1")(outD1)

    conv = Convolution2D(10,5,5, border_mode='same', W_regularizer=l2(0.05))(x)
    #conv = Activation('relu')(conv)
    conv = MaxPooling2D(pool_size = (2,2), strides=(2,2))(conv)
    conv = BatchNormalization()(conv)
    conv = Dropout(0.4)(conv)
    conv = Convolution2D(20,5,5, border_mode='same', W_regularizer=l2(0.05))(conv)
    #conv = Activation('relu')(conv)
    conv = MaxPooling2D(pool_size = (2,2), strides=(2,2))(conv)
    conv = BatchNormalization()(conv)
    conv = Dropout(0.4)(conv)
    conv = Convolution2D(30,5,5, border_mode='same', W_regularizer=l2(0.05))(conv)
    #conv = Activation('relu')(conv)
    conv = MaxPooling2D(pool_size = (2,2), strides=(2,2))(conv)
    conv = BatchNormalization()(conv)
    merged2 = Flatten()(conv)
    merged2 = merge([outL, merged2], mode='concat')
    merged2 = Dropout(0.4)(merged2)
    merged2 = Dense(1024, W_regularizer=l2(0.05))(merged2)
    merged2 = BatchNormalization()(merged2)
    merged2 = Activation('relu')(merged2)
    merged2 = Dropout(0.4)(merged2)
    merged2 = Dense(1024, W_regularizer=l2(0.05))(merged2)
    merged2 = BatchNormalization()(merged2)
    merged2 = Activation('relu')(merged2)
    outD2 = Dense(11, W_regularizer=l2(0.05))(merged2)
    outD2 = Activation('softmax', name="Digit2")(outD2)

    conv = Convolution2D(10,5,5, border_mode='same', W_regularizer=l2(0.05))(x)
    #conv = Activation('relu')(conv)
    conv = MaxPooling2D(pool_size = (2,2), strides=(2,2))(conv)
    conv = BatchNormalization()(conv)
    conv = Dropout(0.4)(conv)
    conv = Convolution2D(20,5,5, border_mode='same', W_regularizer=l2(0.05))(conv)
    #conv = Activation('relu')(conv)
    conv = MaxPooling2D(pool_size = (2,2), strides=(2,2))(conv)
    conv = BatchNormalization()(conv)
    conv = Dropout(0.4)(conv)    
    conv = Convolution2D(30,5,5, border_mode='same', W_regularizer=l2(0.05))(conv)
    #conv = Activation('relu')(conv)
    conv = MaxPooling2D(pool_size = (2,2), strides=(2,2))(conv)
    conv = BatchNormalization()(conv)
    merged3 = Flatten()(conv)
    merged3 = merge([outL, merged3], mode='concat')
    merged3 = Dropout(0.4)(merged3)
    merged3 = Dense(1024, W_regularizer=l2(0.05))(merged3)
    merged3 = BatchNormalization()(merged3)
    merged3 = Activation('relu')(merged3)
    merged3 = Dropout(0.4)(merged3)
    merged3 = Dense(1024, W_regularizer=l2(0.05))(merged3)
    merged3 = BatchNormalization()(merged3)
    merged3 = Activation('relu')(merged3)
    outD3 = Dense(11, W_regularizer=l2(0.05))(merged3)
    outD3 = Activation('softmax', name="Digit3")(outD3)

    conv = Convolution2D(10,5,5, border_mode='same', W_regularizer=l2(0.05))(x)
    #conv = Activation('relu')(conv)
    conv = MaxPooling2D(pool_size = (2,2), strides=(2,2))(conv)
    conv = BatchNormalization()(conv)
    conv = Dropout(0.4)(conv)
    conv = Convolution2D(20,5,5, border_mode='same', W_regularizer=l2(0.05))(conv)
    #conv = Activation('relu')(conv)
    conv = MaxPooling2D(pool_size = (2,2), strides=(2,2))(conv)
    conv = BatchNormalization()(conv)
    conv = Dropout(0.4)(conv)
    conv = Convolution2D(30,5,5, border_mode='same', W_regularizer=l2(0.05))(conv)
    #conv = Activation('relu')(conv)
    conv = MaxPooling2D(pool_size = (2,2), strides=(2,2))(conv)
    conv = BatchNormalization()(conv)
    merged4 = Flatten()(conv)
    merged4 = merge([outL, merged4], mode='concat')
    merged4 = Dropout(0.4)(merged4)
    merged4 = Dense(1024, W_regularizer=l2(0.05))(merged4)
    merged4 = BatchNormalization()(merged4)
    merged4 = Activation('relu')(merged4)
    merged4 = Dropout(0.4)(merged4)
    merged4 = Dense(1024, W_regularizer=l2(0.05))(merged4)
    merged4 = BatchNormalization()(merged4)
    merged4 = Activation('relu')(merged4)
    outD4 = Dense(11, W_regularizer=l2(0.0))(merged4)
    outD4 = Activation('softmax', name="Digit4")(outD4)

    model = Model(input=x, output=[outL, outD0, outD1, outD2, outD3, outD4]) 
    return model

import os.path
if os.path.isfile('actualModel.h5') is False:
    epochs=40

    #model = strongLengthBias()
    model = getCnnModel()

    model.compile(loss='categorical_crossentropy',
                  optimizer='adagrad',
                  metrics=['accuracy'])#, loss_weights=[1.5,1.,1.,1.,0.4,0.4])

    trainDigit0  = trainImageDigits[:,0,:]
    trainDigit1  = trainImageDigits[:,1,:]
    trainDigit2  = trainImageDigits[:,2,:]
    trainDigit3  = trainImageDigits[:,3,:]
    trainDigit4  = trainImageDigits[:,4,:]

    validationDigit0  = validationImageDigits[:,0,:]
    validationDigit1  = validationImageDigits[:,1,:]
    validationDigit2  = validationImageDigits[:,2,:]
    validationDigit3  = validationImageDigits[:,3,:]
    validationDigit4  = validationImageDigits[:,4,:]


    model.fit(trainImageData, [trainImageLengths, trainDigit0, trainDigit1, trainDigit2, trainDigit3, trainDigit4], nb_epoch=epochs, batch_size=batch_size, validation_data=(validationImageData,[validationImageLengths,validationDigit0,validationDigit1,validationDigit2,validationDigit3,validationDigit4]))
    model.save('actualModel.h5')
else:
    print('Loading saved model')
    from keras.models import load_model
    model = load_model('actualModel.h5')

testDigit0  = testImageDigits[:,0,:]
testDigit1  = testImageDigits[:,1,:]
testDigit2  = testImageDigits[:,2,:]
testDigit3  = testImageDigits[:,3,:]
testDigit4  = testImageDigits[:,4,:]

score = model.evaluate(testImageData, [testImageLengths, testDigit0, testDigit1, testDigit2, testDigit3, testDigit4], batch_size=32)
print('Test score:', score)
