from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input,ConvLSTM2D,TimeDistributed,concatenate, Add, Bidirectional,Concatenate, dot, add, multiply, Permute
from tensorflow.keras.layers import Activation, Reshape, Dense, RepeatVector, Dropout
from tensorflow.keras.layers import Conv3D,Conv2D,SeparableConv2D,Cropping3D,Conv2DTranspose
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Lambda
import tensorflow.keras.backend as K


def LSTM_Conv2D_KDD():
    encoder1_inputs = Input(shape=(None, 29, 33, 4), name='encoder1_inputs')  #(batch_size,6,29,33,3)
    encoder1_conv2d = TimeDistributed(Conv2D(filters=64,kernel_size=(7, 7),strides=(2, 2),padding='same'),name='en1_conv2d')(encoder1_inputs)
    encoder1_conv2d = TimeDistributed(Activation('relu'))(encoder1_conv2d)

    encoder1_convlstm,h1,c1 = ConvLSTM2D(filters=128, kernel_size=(5, 5),
                    return_state=True, padding='same', return_sequences=True,name='en1_convlstm')(encoder1_conv2d)
    # --------------------------------------------------------------------------------
    encoder2_inputs = Input(shape=(None, 29, 33, 1), name='encoder2_inputs')  #(batch_size,3,29,33,1)
    encoder2_conv2d = TimeDistributed(Conv2D(filters=4, kernel_size=(7, 7), strides=(2, 2), padding='same'),name='en2_conv2d')(encoder2_inputs)
    encoder2_conv2d = TimeDistributed(Activation('relu'))(encoder2_conv2d)

    encoder2_convlstm,h2,c2 = ConvLSTM2D(filters=8, kernel_size=(5, 5),
                    return_state=True, padding='same', return_sequences=True,name='en2_convlstm')(encoder2_conv2d)
    # --------------------------------------------------------------------------------
    h = concatenate([h1, h2],axis=-1)
    c = concatenate([c1, c2],axis=-1)
    h = Conv2D(filters=64, kernel_size=(1, 1), padding="same", name='h_conv2d', activation='relu')(h)
    c = Conv2D(filters=64, kernel_size=(1, 1), padding="same", name='c_conv2d', activation='relu')(c)
    
    # --------------------------------------------------------------------------------

    _decoder_inputs = Input(shape=(None, 29, 33, 1), name='decoder_inputs')  #(batch_size,1,29,33,1)
    decoder_inputs = _decoder_inputs
    de_conv2d = TimeDistributed(Conv2D(filters=4, kernel_size=(7, 7), strides=(2, 2),padding='same'), name='de_conv2d')
    de_convlstm = ConvLSTM2D(filters=64, kernel_size=(5, 5), name='de_convlstm',
                            return_state=True, padding='same', return_sequences=True)
    de_conv2dT = TimeDistributed(Conv2DTranspose(filters=64, kernel_size=(7, 7), strides=(2, 2),padding='same'), name='de_conv2dT')
    de_out_conv2d = TimeDistributed(Conv2D(filters=1, kernel_size=(1, 1), padding="same"), name='de_out_conv2d')
    
    # ----------------------------------------------------------

    relu = Activation('relu')
    sigmoid = Activation('sigmoid')
    cropper = Cropping3D(cropping=((0, 0), (0, 1), (0, 1)))

    # decoder: data flow-----------------------------------------
    
    decoder_outputs = []
    for i in range(6):
        decoder_conv2d = de_conv2d(decoder_inputs)
        decoder_conv2d = relu(decoder_conv2d)

        decoder_convlstm, h, c = de_convlstm([decoder_conv2d, h, c])

        decoder_conv2dT = de_conv2dT(decoder_convlstm)
        decoder_conv2dT = relu(decoder_conv2dT)

        decoder_out_conv2d = de_out_conv2d(decoder_conv2dT)  
        decoder_output = cropper(decoder_out_conv2d)  
        decoder_outputs.append(decoder_output)
        decoder_output = sigmoid(decoder_output)
        decoder_inputs = decoder_output

    decoder_outputs = Lambda(lambda x: K.concatenate(x, axis=1))(decoder_outputs)   
    decoder_outputs = Reshape((-1, 29 * 33, 1), input_shape=(-1, 29, 33, 1))(decoder_outputs)
    
    # ----------------------------------------------------------
    
    return Model([encoder1_inputs, encoder2_inputs, _decoder_inputs], decoder_outputs, name='ConvLSTM-Conv2d-KDD')