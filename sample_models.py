from keras import backend as K
from keras.models import Model
from keras.layers import (BatchNormalization, Conv1D, Dense, Input, LeakyReLU, AveragePooling1D,
    TimeDistributed, Activation, Bidirectional, SimpleRNN, GRU, LSTM, MaxPooling1D, Dropout, GaussianNoise)

def simple_rnn_model(input_dim, output_dim=29):
    """ Build a recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add recurrent layer
    simp_rnn = GRU(output_dim, return_sequences=True, 
                 implementation=2, name='rnn')(input_data)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(simp_rnn)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def rnn_model(input_dim, units, activation, output_dim=29):
    """ Build a recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add recurrent layer
    simp_rnn = GRU(units, activation=activation,
        return_sequences=True, implementation=2, name='rnn')(input_data)
    # TODO: Add batch normalization 
    bn_rnn = BatchNormalization()(simp_rnn)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model


def cnn_rnn_model(input_dim, filters, kernel_size, conv_stride,
    conv_border_mode, units, output_dim=29):
    """ Build a recurrent + convolutional network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add convolutional layer
    conv_1d = Conv1D(filters, kernel_size, 
                     strides=conv_stride, 
                     padding=conv_border_mode,
                     activation='relu',
                     name='conv1d')(input_data)
    # Add batch normalization
    bn_cnn = BatchNormalization(name='bn_conv_1d')(conv_1d)
    # Add a recurrent layer
    simp_rnn = GRU(units, activation='relu',
        return_sequences=True, implementation=2, name='rnn')(bn_cnn)
    # TODO: Add batch normalization
    bn_rnn = BatchNormalization()(simp_rnn)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: cnn_output_length(
        x, kernel_size, conv_border_mode, conv_stride)
    print(model.summary())
    return model

def cnn_output_length(input_length, filter_size, border_mode, stride,
                       pool_size, n_conv, dilation=1, max_pooling=False):
    """ Compute the length of the output sequence after 1D convolution along
        time. Note that this function is in line with the function used in
        Convolution1D class from Keras.
    Params:
        input_length (int): Length of the input sequence.
        filter_size (int): Width of the convolution kernel.
        border_mode (str): Only support `same` or `valid`.
        stride (int): Stride size used in 1D convolution.
        dilation (int)
    """
    if input_length is None:
        return None
    assert border_mode in {'same', 'valid'}
    dilated_filter_size = filter_size + (filter_size - 1) * (dilation - 1)
    if border_mode == 'same':
        output_length = input_length
    elif border_mode == 'valid':
        output_length = input_length
        i = 0
        for i in range(n_conv):
        #while i < n_conv:
            output_length = output_length - dilated_filter_size + 1
            output_length = (output_length + stride - 1) // stride
            i += 1
        if max_pooling == True:
            output_length = (output_length - pool_size + 1) // stride
    return output_length

def deep_rnn_model(input_dim, units, recur_layers, output_dim=29):
    """ Build a deep recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # TODO: Add recurrent layers, each with batch normalization
    rnn1 = GRU(units, activation='relu',
        return_sequences=True, implementation=2, name='rnn1')(input_data)
    bn1 = BatchNormalization(name='bn1')(rnn1)
    rnn2 = GRU(units, activation='relu',
        return_sequences=True, implementation=2, name='rnn2')(bn1)
    bn2 = BatchNormalization(name='bn2')(rnn2)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bn2)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def bidirectional_rnn_model(input_dim, units, output_dim=29):
    """ Build a bidirectional recurrent network for speech
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # TODO: Add bidirectional recurrent layer
    bidir_rnn = Bidirectional(GRU(units, activation='relu',
                                  return_sequences=True, implementation=2, name='bdrnn'))(input_data)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bidir_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def optional_model(input_dim, units, conv_stride, 
                   filters, kernel_size,conv_border_mode, output_dim=29):
    # includes CNN->BN->Bd(GRU)->BN->TD(Dense)
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add CNN
    conv_1d = Conv1D(filters, kernel_size, 
                     strides=conv_stride,
                     padding=conv_border_mode,
                     activation='relu',
                     name='conv1d')(input_data)
    # Add batch normalization
    bn_cnn = BatchNormalization(name='bn_cnn_1d')(conv_1d)
    # Add bidirectional GRU
    bd_rnn = Bidirectional(GRU(units,
                               return_sequences=True, implementation=2,
                               name='bdrnn'))(bn_cnn)
    # Add batch normalization
    bn_rnn = BatchNormalization(name='bn_rnn')(bd_rnn)
    relu_rnn = Activation('relu', name='relu')(bn_rnn)
    # Add dropout
    dropout1 = Dropout(rate=0.2)(relu_rnn)
    time_dense = TimeDistributed(Dense(output_dim))(dropout1)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: cnn_output_length(
        x, kernel_size, conv_border_mode, conv_stride)
    print(model.summary())
    return model

def final_model(input_dim, filters, kernel_size, conv_stride,
    conv_border_mode, units,dilation_rate, pool_size, output_dim=29):
    """ Build a deep network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Apply additive zero-centered Gaussian noise on input data to prevent overfitting
    gaussian = GaussianNoise(0)(input_data)
    # Add convolutional layer
    conv_1d = Conv1D(filters, kernel_size,
                     dilation_rate=dilation_rate,
                     strides=conv_stride,
                     padding=conv_border_mode,
                     activation='relu',
                     name='conv1d')(gaussian)
    #conv_2d = Conv1D(filters, kernel_size,
     #                dilation_rate=dilation_rate,
      #               strides=conv_stride,
       #              padding=conv_border_mode,
        #             activation='relu',
         #            name='conv2d')(conv_1d)
    # Add max pooling layer
    mp_conv = MaxPooling1D(pool_size=pool_size)(conv_1d)
    # Add batch normalization
    bn_cnn1 = BatchNormalization(name='bn_conv1d')(mp_conv)
    # Dropout
    drp_conv = Dropout(rate=0.5)(bn_cnn1)
    
    # BiDirectional RNNs
    bdrnn1 = Bidirectional(GRU(units, return_sequences=True, implementation=2, name='bdrnn1'))(drp_conv)
    bn1 = BatchNormalization(name='bn1')(bdrnn1)
    act1 = Activation('relu', name='act1relu')(bn1)
    dropout1 = Dropout(rate=0.5)(act1)
    #act1 = LeakyReLU(alpha=1, name='leakyRelu1')(dropout1)
                     
    bdrnn2 = Bidirectional(GRU(units,return_sequences=True, implementation=2,name='bdrnn2'))(dropout1)
    bn2 = BatchNormalization(name='bn2')(bdrnn2)
    act2 = Activation('relu', name='act2relu')(bn2)
    dropout2 = Dropout(rate=0.5)(act2)
    #act2 = LeakyReLU(alpha=1, name='leakyReLU2')(dropout2)
    
    #bdrnn3 = Bidirectional(GRU(units, return_sequences=True, implementation=2,name='bdrnn3'))(act2)
    #bn3 = BatchNormalization(name='bn3')(bdrnn3)
    #act3 = Activation('relu', name='act3relu')(bn3)
    #dropout3 = Dropout(rate=0.3)(act3)
    #act3 = LeakyReLU(alpha=1, name='leakyRelu3')(dropout3)
    
    time_dense = TimeDistributed(Dense(output_dim))(dropout2)
    # TODO: Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    # TODO: Specify model.output_length
    model.output_length = lambda x: cnn_output_length(x, kernel_size, conv_border_mode,
                                                      conv_stride, n_conv=1, max_pooling=True,
                                                      pool_size=pool_size) 
    print(model.summary())
    return model