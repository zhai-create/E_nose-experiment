from tensorflow.python.keras import layers as kl
from tensorflow.python.keras import models as km
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras import backend as K
try:  # tf2
    import tensorflow.compat.v1 as tf
    from tensorflow.compat.v1 import ConfigProto
    from tensorflow.compat.v1 import InteractiveSession
except ImportError:
    import tensorflow as tf
from utils_data import loadmat_1, acc_calc, loadmat_5
import numpy as np
import inspect
from functools import partial
import random


def _param_input(dataset, setting):
    if dataset is None:
        dataset = input('[INPUT] Select dataset (1 for 10boards or 2 for 4months):')
        dataset = int(dataset[0])
    assert dataset == 1 or dataset == 2, 'Please input correct dataset number.'
    if setting is None:
        setting = input('[INPUT] Select setting (1 or 2):')
        setting = int(setting[0])
    assert setting == 1 or setting == 2, 'Please input correct setting.'
    if dataset == 1:  # dataset 1
        param = {'insize': 8, 'sensornum': 16, 'classnum': 6, 'batchnum': 10}
        loaddata = partial(loadmat_1, norm=True)
    else:  # dataset 2
        param = {'insize': 50, 'sensornum': 21, 'classnum': 7, 'batchnum': 3}
        loaddata = loadmat_5
    return dataset, setting, param, loaddata


def _dataloader(dataset, setting, param, loaddata, tbatch=None):
    if setting == 1:
        sdata, slabel = loaddata(1, shuffle=True)
        if dataset == 2:
            sdata = sdata.swapaxes(0, 1).swapaxes(1, 2)
            sdata = sdata.squeeze(axis=(4,))  # reduce dim
        tdata = np.ndarray((0, param['insize'], param['sensornum'], 1))
        tlabel = np.ndarray((0, param['classnum']))
        for tbatch in range(2, param['batchnum'] + 1):
            data, label = loaddata(batch=tbatch, shuffle=True)
            if dataset == 2:
                data = data.swapaxes(0, 1).swapaxes(1, 2)
                data = data.squeeze(axis=(4,))  # reduce dim
            tdata = np.concatenate((tdata, data), axis=0)
            tlabel = np.concatenate((tlabel, label), axis=0)
    else:
        sdata, slabel = loaddata(tbatch - 1, shuffle=True)
        tdata, tlabel = loaddata(tbatch, shuffle=True)
        if dataset == 2:
            sdata = sdata.swapaxes(0, 1).swapaxes(1, 2)
            sdata = sdata.squeeze(axis=(4,))  # reduce dim
            tdata = tdata.swapaxes(0, 1).swapaxes(1, 2)
            tdata = tdata.squeeze(axis=(4,))  # reduce dim
    return sdata, slabel, tdata, tlabel


def auto_solver(dataset):
    netlist = [network_GoogLeNet, network_Resnet34, network_ODCNN,
               network_GasNet, network_AlexNet,
               network_LeNet, network_SniffConv, network_SniffMultinose]
    for net in netlist:
        for s in [1, 2]:
            net(dataset, s)
    # for VGG net
    for vgg in [1, 2, 3]:
        for s in [1, 2]:
            network_VGG(dataset, s, vgg)


def network_GoogLeNet(dataset=None, setting=None):
    def Conv2d_BN(x, nb_filter, kernel_size, padding='same', strides=(1, 1), name=None):
        if name is not None:
            bn_name = name + '_bn'
            conv_name = name + '_conv'
        else:
            bn_name = None
            conv_name = None
        x = kl.Conv2D(nb_filter, kernel_size, padding=padding, strides=strides, activation='relu', name=conv_name)(x)
        x = kl.BatchNormalization(axis=3, name=bn_name)(x)
        return x

    def Inception(x, nb_filter):
        branch1x1 = Conv2d_BN(x, nb_filter, (1, 1), padding='same', strides=(1, 1), name=None)
        branch3x3 = Conv2d_BN(x, nb_filter, (1, 1), padding='same', strides=(1, 1), name=None)
        branch3x3 = Conv2d_BN(branch3x3, nb_filter, (3, 3), padding='same', strides=(1, 1), name=None)
        branch5x5 = Conv2d_BN(x, nb_filter, (1, 1), padding='same', strides=(1, 1), name=None)
        branch5x5 = Conv2d_BN(branch5x5, nb_filter, (1, 1), padding='same', strides=(1, 1), name=None)
        branchpool = kl.MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(x)
        branchpool = Conv2d_BN(branchpool, nb_filter, (1, 1), padding='same', strides=(1, 1), name=None)
        x = kl.Concatenate(axis=3)([branch1x1, branch3x3, branch5x5, branchpool])
        return x

    dataset, setting, param, loaddata = _param_input(dataset, setting)
    print("[INFO] ({}) Using dataset {}, setting {}.".format(inspect.stack()[0][3], dataset, setting))
    # create network
    inpt = kl.Input(shape=(param['insize'], param['sensornum'], 1))
    # padding = 'same'，填充为(步长-1）/2,还可以用ZeroPadding2D((3,3))
    x = Conv2d_BN(inpt, 64, (7, 7), strides=(2, 2), padding='same')
    if dataset == 2:
        x = kl.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
    x = Conv2d_BN(x, 192, (3, 3), strides=(1, 1), padding='same')
    if dataset == 2:
        x = kl.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
    x = Inception(x, 64)  # 256
    x = Inception(x, 120)  # 480
    if dataset == 2:
        x = kl.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
    x = Inception(x, 128)  # 512
    x = Inception(x, 128)
    x = Inception(x, 128)
    x = Inception(x, 132)  # 528
    x = Inception(x, 208)  # 832
    x = kl.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
    x = Inception(x, 208)
    x = Inception(x, 256)  # 1024
    # x = kl.AveragePooling2D(pool_size=(7, 7), strides=(7, 7), padding='same')(x)
    x = kl.Flatten()(x)
    x = kl.Dropout(0.4)(x)
    x = kl.Dense(1000, activation='relu')(x)
    x = kl.Dense(param['classnum'], activation='softmax')(x)
    model = km.Model(inpt, x, name='inception')
    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    # model.summary()
    # predict
    acc = []
    if setting == 1:
        sdata, slabel, tdata, tlabel = _dataloader(dataset, setting, param, loaddata)
        history = model.fit(sdata, slabel, validation_data=[tdata, tlabel], batch_size=30, epochs=100, verbose=0)
        acc.append(history.history['val_acc'][-1])
    else:
        for tbatch in range(2, param['batchnum'] + 1):
            sdata, slabel, tdata, tlabel = _dataloader(dataset, setting, param, loaddata, tbatch)
            history = model.fit(sdata, slabel, validation_data=[tdata, tlabel], batch_size=30, epochs=100, verbose=0)
            acc.append(history.history['val_acc'][-1])
    print("[INFO] ({}) Complete dataset {} setting {}, Average accuracy: {}.".format(inspect.stack()[0][3], dataset, setting, np.mean(acc)))
    # K.clear_session()


def network_Resnet34(dataset=None, setting=None):
    def Conv2d_BN(x, nb_filter, kernel_size, strides=(1, 1), padding='same', name=None):
        if name is not None:
            bn_name = name + '_bn'
            conv_name = name + '_conv'
        else:
            bn_name = None
            conv_name = None

        x = kl.Conv2D(nb_filter, kernel_size, padding=padding, strides=strides, activation='relu', name=conv_name)(x)
        x = kl.BatchNormalization(axis=3, name=bn_name)(x)
        return x

    def Conv_Block(inpt, nb_filter, kernel_size, strides=(1, 1), with_conv_shortcut=False):
        x = Conv2d_BN(inpt, nb_filter=nb_filter, kernel_size=kernel_size, strides=strides, padding='same')
        x = Conv2d_BN(x, nb_filter=nb_filter, kernel_size=kernel_size, padding='same')
        if with_conv_shortcut:
            shortcut = Conv2d_BN(inpt, nb_filter=nb_filter, strides=strides, kernel_size=kernel_size)
            x = kl.add([x, shortcut])
            return x
        else:
            x = kl.add([x, inpt])
            return x

    dataset, setting, param, loaddata = _param_input(dataset, setting)
    print("[INFO] ({}) Using dataset {}, setting {}.".format(inspect.stack()[0][3], dataset, setting))
    # create network
    inpt = kl.Input(shape=(param['insize'], param['sensornum'], 1))
    x = kl.ZeroPadding2D((3, 3))(inpt)
    x = Conv2d_BN(x, nb_filter=64, kernel_size=(7, 7), strides=(2, 2), padding='valid')
    if dataset == 2:
        x = kl.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
    # (56,56,64)
    x = Conv_Block(x, nb_filter=64, kernel_size=(3, 3))
    x = Conv_Block(x, nb_filter=64, kernel_size=(3, 3))
    x = Conv_Block(x, nb_filter=64, kernel_size=(3, 3))
    # (28,28,128)
    x = Conv_Block(x, nb_filter=128, kernel_size=(3, 3), strides=(2, 2), with_conv_shortcut=True)
    x = Conv_Block(x, nb_filter=128, kernel_size=(3, 3))
    x = Conv_Block(x, nb_filter=128, kernel_size=(3, 3))
    x = Conv_Block(x, nb_filter=128, kernel_size=(3, 3))
    # (14,14,256)
    x = Conv_Block(x, nb_filter=256, kernel_size=(3, 3), strides=(2, 2), with_conv_shortcut=True)
    x = Conv_Block(x, nb_filter=256, kernel_size=(3, 3))
    x = Conv_Block(x, nb_filter=256, kernel_size=(3, 3))
    x = Conv_Block(x, nb_filter=256, kernel_size=(3, 3))
    x = Conv_Block(x, nb_filter=256, kernel_size=(3, 3))
    x = Conv_Block(x, nb_filter=256, kernel_size=(3, 3))
    # (7,7,512)
    x = Conv_Block(x, nb_filter=512, kernel_size=(3, 3), strides=(2, 2), with_conv_shortcut=True)
    x = Conv_Block(x, nb_filter=512, kernel_size=(3, 3))
    x = Conv_Block(x, nb_filter=512, kernel_size=(3, 3))
    # if dataset == 2:
    #     x = kl.AveragePooling2D(pool_size=(7,7))(x)
    x = kl.Flatten()(x)
    x = kl.Dense(param['classnum'], activation='softmax')(x)
    model = km.Model(inputs=inpt, outputs=x)
    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    # model.summary()
    # predict
    acc = []
    if setting == 1:
        sdata, slabel, tdata, tlabel = _dataloader(dataset, setting, param, loaddata)
        history = model.fit(sdata, slabel, validation_data=[tdata, tlabel], batch_size=30, epochs=100, verbose=0)
        acc.append(history.history['val_acc'][-1])
    else:
        for tbatch in range(2, param['batchnum'] + 1):
            sdata, slabel, tdata, tlabel = _dataloader(dataset, setting, param, loaddata, tbatch)
            history = model.fit(sdata, slabel, validation_data=[tdata, tlabel], batch_size=30, epochs=100, verbose=0)
            acc.append(history.history['val_acc'][-1])
    print("[INFO] ({}) Complete dataset {} setting {}, Average accuracy: {}.".format(inspect.stack()[0][3], dataset, setting, np.mean(acc)))
    # K.clear_session()


def network_ODCNN(dataset=None, setting=None):
    dataset, setting, param, loaddata = _param_input(dataset, setting)
    print("[INFO] ({}) Using dataset {}, setting {}.".format(inspect.stack()[0][3], dataset, setting))
    # create network
    inputs = kl.Input(shape=(param['insize'], param['sensornum'], 1))
    if dataset == 1:
        bone = kl.Conv2D(filters=32, kernel_size=(2, 1), padding='same', activation='relu', strides=(1, 1))(inputs)
    else:
        bone = kl.BatchNormalization(1)(inputs)
        bone = kl.Conv2D(filters=32, kernel_size=(2, 1), padding='same', activation='relu', strides=(1, 1))(bone)
    bone = kl.MaxPool2D(pool_size=(2, 1), strides=(2, 1), padding='same')(bone)
    bone = kl.Conv2D(filters=16, kernel_size=(2, 1), padding='same', activation='relu', strides=(1, 1))(bone)
    bone = kl.MaxPool2D(pool_size=(2, 1), strides=(2, 1), padding='same')(bone)
    bone = kl.Conv2D(filters=16, kernel_size=(2, 1), padding='same', activation='relu', strides=(1, 1))(bone)
    bone = kl.Conv2D(filters=16, kernel_size=(2, 1), padding='same', activation='relu', strides=(1, 1))(bone)
    bone = kl.Conv2D(filters=16, kernel_size=(2, 1), padding='same', activation='relu', strides=(1, 1))(bone)
    bone = kl.Flatten()(bone)
    bone = kl.Dense(units=1024, activation='relu')(bone)
    outputs = kl.Dense(units=param['classnum'], activation='softmax')(bone)
    model = km.Model(inputs=inputs, outputs=outputs)
    # model.summary()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    # predict
    acc = []
    if setting == 1:
        sdata, slabel, tdata, tlabel = _dataloader(dataset, setting, param, loaddata)
        history = model.fit(sdata, slabel, validation_data=[tdata, tlabel], batch_size=80, epochs=100, verbose=0)
        acc.append(history.history['val_acc'][-1])
    else:
        for tbatch in range(2, param['batchnum'] + 1):
            sdata, slabel, tdata, tlabel = _dataloader(dataset, setting, param, loaddata, tbatch)
            history = model.fit(sdata, slabel, validation_data=[tdata, tlabel], batch_size=80, epochs=100, verbose=0)
            acc.append(history.history['val_acc'][-1])
    print("[INFO] ({}) Complete dataset {} setting {}, Average accuracy: {}.".format(inspect.stack()[0][3], dataset, setting, np.mean(acc)))
    # K.clear_session()


def network_GasNet(dataset=None, setting=None):
    dataset, setting, param, loaddata = _param_input(dataset, setting)
    print("[INFO] ({}) Using dataset {}, setting {}.".format(inspect.stack()[0][3], dataset, setting))
    # create network
    # input
    inputs = kl.Input(shape=(param['insize'], param['sensornum'], 1))
    # block 1
    block1 = kl.Conv2D(filters=32, kernel_size=(3, 3), padding='same', strides=(1, 1))(inputs)
    block1 = kl.BatchNormalization(1)(block1)
    block1 = kl.Activation('relu')(block1)
    block1 = kl.Conv2D(filters=32, kernel_size=(3, 3), padding='same', strides=(1, 1))(block1)
    block1 = kl.BatchNormalization(1)(block1)
    block1 = kl.Activation('relu')(block1)
    # block 2
    block2 = kl.Conv2D(filters=32, kernel_size=(3, 3), padding='same', strides=(1, 1))(block1)
    block2 = kl.BatchNormalization(1)(block2)
    block2 = kl.Activation('relu')(block2)
    block2 = kl.Conv2D(filters=32, kernel_size=(3, 3), padding='same', strides=(1, 1))(block2)
    block2 = kl.BatchNormalization(1)(block2)
    block2 = kl.Activation('relu')(block2)
    block2 = kl.Add()([block1, block2])  # shortcut
    # maxpooling 1
    maxp1 = kl.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(block2)
    # block 3
    block3 = kl.Conv2D(filters=64, kernel_size=(3, 3), padding='same', strides=(1, 1))(maxp1)
    block3 = kl.BatchNormalization(1)(block3)
    block3 = kl.Activation('relu')(block3)
    block3 = kl.Conv2D(filters=64, kernel_size=(3, 3), padding='same', strides=(1, 1))(block3)
    block3 = kl.BatchNormalization(1)(block3)
    block3 = kl.Activation('relu')(block3)
    block2 = kl.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(block2)  # match dimension: 64, 1*1, 2
    block2 = kl.Conv2D(filters=64, kernel_size=(1, 1), padding='valid', strides=(1, 1))(block2)  # match dimension: 64, 1*1, 2
    block3 = kl.Add()([block2, block3])  # shortcut
    # block 4
    block4 = kl.Conv2D(filters=64, kernel_size=(3, 3), padding='same', strides=(1, 1))(block3)
    block4 = kl.BatchNormalization(1)(block4)
    block4 = kl.Activation('relu')(block4)
    block4 = kl.Conv2D(filters=64, kernel_size=(3, 3), padding='same', strides=(1, 1))(block4)
    block4 = kl.BatchNormalization(1)(block4)
    block4 = kl.Activation('relu')(block4)
    block4 = kl.Add()([block3, block4])  # shortcut
    # maxpooling 2
    maxp2 = kl.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(block4)
    # block 5
    block5 = kl.Conv2D(filters=128, kernel_size=(3, 3), padding='same', strides=(1, 1))(maxp2)
    block5 = kl.BatchNormalization(1)(block5)
    block5 = kl.Activation('relu')(block5)
    block5 = kl.Conv2D(filters=128, kernel_size=(3, 3), padding='same', strides=(1, 1))(block5)
    block5 = kl.BatchNormalization(1)(block5)
    block5 = kl.Activation('relu')(block5)
    block4 = kl.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(block4)  # match dimension: 128, 1*1, 2
    block4 = kl.Conv2D(filters=128, kernel_size=(1, 1), padding='valid', activation=None, strides=(1, 1))(block4)  # match dimension: 128, 1*1, 2
    block5 = kl.Add()([block4, block5])  # shortcut
    # block 6
    block6 = kl.Conv2D(filters=128, kernel_size=(3, 3), padding='same', strides=(1, 1))(block5)
    block6 = kl.BatchNormalization(1)(block6)
    block6 = kl.Activation('relu')(block6)
    block6 = kl.Conv2D(filters=128, kernel_size=(3, 3), padding='same', strides=(1, 1))(block6)
    block6 = kl.BatchNormalization(1)(block6)
    block6 = kl.Activation('relu')(block6)
    block6 = kl.Add()([block5, block6])  # shortcut
    # Global Average Pooling(GAP)
    GAP = kl.GlobalAveragePooling2D(data_format='channels_last')(block6)
    # output
    outputs = kl.Dense(units=param['classnum'], activation='softmax')(GAP)
    model = km.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='SGD', loss='categorical_crossentropy', metrics=['accuracy'])
    # predict
    acc = []
    if setting == 1:
        sdata, slabel, tdata, tlabel = _dataloader(dataset, setting, param, loaddata)
        history = model.fit(sdata, slabel, validation_data=[tdata, tlabel], batch_size=80, epochs=100, verbose=0)
        acc.append(history.history['val_acc'][-1])
    else:
        for tbatch in range(2, param['batchnum'] + 1):
            sdata, slabel, tdata, tlabel = _dataloader(dataset, setting, param, loaddata, tbatch)
            history = model.fit(sdata, slabel, validation_data=[tdata, tlabel], batch_size=80, epochs=100, verbose=0)
            acc.append(history.history['val_acc'][-1])
    print("[INFO] ({}) Complete dataset {} setting {}, Average accuracy: {}.".format(inspect.stack()[0][3], dataset, setting, np.mean(acc)))
    # K.clear_session()


def network_VGG(dataset=None, setting=None, vgg=None):
    dataset, setting, param, loaddata = _param_input(dataset, setting)
    if vgg is None:
        vgg = input('Select network: 1 for VGG13, 2 for VGG16, 3 for VGG19.')
        vgg = int(vgg[0])
    assert vgg == 1 or vgg == 2 or vgg == 3, 'Please input correct number.'
    print("[INFO] ({}) Using dataset {}, setting {}, VGG {}.".format(inspect.stack()[0][3], dataset, setting, vgg))
    # create network
    model = Sequential()
    model.add(kl.Conv2D(64, (3, 3), strides=(1, 1), input_shape=(param['insize'], param['sensornum'], 1),
                        padding='same', activation='relu', kernel_initializer='uniform'))
    model.add(kl.Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))

    # if dataset == 2:
    #     model.add(kl.MaxPooling2D(pool_size=(2, 2)))
    model.add(kl.Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
    model.add(kl.Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))

    # if dataset == 2:
    #     model.add(kl.MaxPooling2D(pool_size=(2, 2)))
    model.add(kl.Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
    model.add(kl.Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
    if vgg == 2:
        model.add(kl.Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
    elif vgg == 3:
        model.add(kl.Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
        model.add(kl.Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))

    model.add(kl.MaxPooling2D(pool_size=(2, 2)))
    model.add(kl.Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
    model.add(kl.Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
    if vgg == 2:
        model.add(kl.Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
    elif vgg == 3:
        model.add(kl.Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
        model.add(kl.Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))

    # if dataset == 2:
    #     model.add(kl.MaxPooling2D(pool_size=(2, 2)))
    model.add(kl.Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
    model.add(kl.Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
    if vgg == 2:
        model.add(kl.Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
    elif vgg == 3:
        model.add(kl.Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
        model.add(kl.Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))

    # if dataset == 2:
    #     model.add(kl.MaxPooling2D(pool_size=(2, 2)))
    model.add(kl.Flatten())
    model.add(kl.Dense(100, activation='relu'))  # 4096
    # model.add(kl.Dropout(0.5))
    model.add(kl.Dense(100, activation='relu'))  # 4096
    # model.add(kl.Dropout(0.5))
    model.add(kl.Dense(param['classnum'], activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    # predict
    acc = []
    if setting == 1:
        sdata, slabel, tdata, tlabel = _dataloader(dataset, setting, param, loaddata)
        history = model.fit(sdata, slabel, validation_data=[tdata, tlabel], batch_size=60, epochs=100, verbose=0)
        acc.append(history.history['val_acc'][-1])
    else:
        for tbatch in range(2, param['batchnum'] + 1):
            sdata, slabel, tdata, tlabel = _dataloader(dataset, setting, param, loaddata, tbatch)
            history = model.fit(sdata, slabel, validation_data=[tdata, tlabel], batch_size=60, epochs=100, verbose=0)
            acc.append(history.history['val_acc'][-1])
    print("[INFO] ({}) Complete dataset {} setting {} VGG {}, Average accuracy: {}.".
          format(inspect.stack()[0][3], dataset, setting, vgg, np.mean(acc)))
    # K.clear_session()


def network_AlexNet(dataset=None, setting=None):
    dataset, setting, param, loaddata = _param_input(dataset, setting)
    print("[INFO] ({}) Using dataset {}, setting {}.".format(inspect.stack()[0][3], dataset, setting))
    # create network
    model = Sequential()
    model.add(kl.Conv2D(96, (11, 11), strides=(1, 1), input_shape=(param['insize'], param['sensornum'], 1),
                        padding='same', activation='relu', kernel_initializer='uniform'))  # 'valid' strides=(4,4)
    if dataset == 2:
        model.add(kl.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(kl.Conv2D(256, (5, 5), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
    model.add(kl.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(kl.Conv2D(384, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
    model.add(kl.Conv2D(384, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
    model.add(kl.Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
    model.add(kl.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(kl.Flatten())
    model.add(kl.Dense(4096, activation='relu'))  # 4096
    model.add(kl.Dropout(0.5))
    model.add(kl.Dense(4096, activation='relu'))  # 4096
    model.add(kl.Dropout(0.5))
    model.add(kl.Dense(param['classnum'], activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # predict
    acc = []
    if setting == 1:
        sdata, slabel, tdata, tlabel = _dataloader(dataset, setting, param, loaddata)
        history = model.fit(sdata, slabel, validation_data=[tdata, tlabel], batch_size=60, epochs=100, verbose=0)
        acc.append(history.history['val_acc'][-1])
    else:
        for tbatch in range(2, param['batchnum'] + 1):
            sdata, slabel, tdata, tlabel = _dataloader(dataset, setting, param, loaddata, tbatch)
            history = model.fit(sdata, slabel, validation_data=[tdata, tlabel], batch_size=60, epochs=100, verbose=0)
            acc.append(history.history['val_acc'][-1])
    print("[INFO] ({}) Complete dataset {} setting {}, Average accuracy: {}.".format(inspect.stack()[0][3], dataset, setting, np.mean(acc)))
    # K.clear_session()


def network_LeNet(dataset=None, setting=None):
    dataset, setting, param, loaddata = _param_input(dataset, setting)
    print("[INFO] ({}) Using dataset {}, setting {}.".format(inspect.stack()[0][3], dataset, setting))
    # create network
    model = Sequential()
    model.add(kl.Conv2D(32, (5, 5), strides=(1, 1), input_shape=(param['insize'], param['sensornum'], 1),
                        padding='same', activation='relu', kernel_initializer='uniform'))  # padding='valid'
    model.add(kl.MaxPooling2D(pool_size=(2, 2)))
    model.add(kl.Conv2D(64, (5, 5), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))  # padding='valid'
    model.add(kl.MaxPooling2D(pool_size=(2, 2)))
    model.add(kl.Flatten())
    model.add(kl.Dense(100, activation='relu'))
    model.add(kl.Dense(param['classnum'], activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    # predict
    acc = []
    if setting == 1:
        sdata, slabel, tdata, tlabel = _dataloader(dataset, setting, param, loaddata)
        history = model.fit(sdata, slabel, validation_data=[tdata, tlabel], batch_size=80, epochs=100, verbose=0)
        acc.append(history.history['val_acc'][-1])
    else:
        for tbatch in range(2, param['batchnum'] + 1):
            sdata, slabel, tdata, tlabel = _dataloader(dataset, setting, param, loaddata, tbatch)
            history = model.fit(sdata, slabel, validation_data=[tdata, tlabel], batch_size=80, epochs=100, verbose=0)
            acc.append(history.history['val_acc'][-1])
    print("[INFO] ({}) Complete dataset {} setting {}, Average accuracy: {}.".format(inspect.stack()[0][3], dataset, setting, np.mean(acc)))
    # K.clear_session()


def network_SniffConv(dataset=None, setting=None):
    dataset, setting, param, loaddata = _param_input(dataset, setting)
    print("[INFO] ({}) Using dataset {}, setting {}.".format(inspect.stack()[0][3], dataset, setting))
    # create network
    inputs = kl.Input(shape=(param['insize'], param['sensornum'], 1))
    model = kl.Conv2D(filters=8, kernel_size=(3, 3), padding='same', strides=(1, 1), activation='relu')(inputs)
    model = kl.BatchNormalization(1)(model)
    model = kl.Conv2D(filters=8, kernel_size=(3, 3), padding='same', strides=(1, 1), activation='relu')(model)
    model = kl.BatchNormalization(1)(model)
    model = kl.MaxPool2D(pool_size=(3, 3), strides=(3, 3), padding='same')(model)
    model = kl.Flatten()(model)
    model = kl.Dense(units=200, activation='relu')(model)
    model = kl.BatchNormalization(1)(model)
    model = kl.Dense(units=200, activation='relu')(model)
    output = kl.Dense(units=param['classnum'], activation='softmax')(model)
    model = km.Model(inputs=inputs, outputs=output)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    # predict
    acc = []
    if setting == 1:
        sdata, slabel, tdata, tlabel = _dataloader(dataset, setting, param, loaddata)
        history = model.fit(sdata, slabel, validation_data=[tdata, tlabel], batch_size=80, epochs=100, verbose=0)
        acc.append(history.history['val_acc'][-1])
    else:
        for tbatch in range(2, param['batchnum'] + 1):
            sdata, slabel, tdata, tlabel = _dataloader(dataset, setting, param, loaddata, tbatch)
            history = model.fit(sdata, slabel, validation_data=[tdata, tlabel], batch_size=80, epochs=100, verbose=0)
            acc.append(history.history['val_acc'][-1])
    print("[INFO] ({}) Complete dataset {} setting {}, Average accuracy: {}.".format(inspect.stack()[0][3], dataset, setting, np.mean(acc)))
    # K.clear_session()


def network_SniffMultinose(dataset=None, setting=None):
    dataset, setting, param, loaddata = _param_input(dataset, setting)
    print("[INFO] ({}) Using dataset {}, setting {}.".format(inspect.stack()[0][3], dataset, setting))
    # create network
    fcdict = {}
    for sensor in range(param['sensornum']):
        scope = 'Sensor{}'.format(sensor + 1)
        inputs = kl.Input(shape=(param['insize'], 1, 1), name=scope + '/Input')
        fc = kl.Flatten()(inputs)
        fc = kl.Dense(units=100, activation='relu')(fc)
        fc = kl.Dense(units=100, activation='relu')(fc)
        fc = kl.Dense(units=100, activation='relu')(fc)
        fcdict[scope] = [inputs, fc]
    tree = kl.concatenate([fcdict['Sensor{}'.format(s + 1)][1] for s in range(param['sensornum'])])
    tree = kl.Dense(units=400, activation='relu')(tree)
    tree = kl.Dense(units=400, activation='relu')(tree)
    tree = kl.Dense(units=param['classnum'], activation='softmax')(tree)
    model = km.Model(inputs=[fcdict['Sensor{}'.format(s + 1)][0] for s in range(param['sensornum'])], outputs=[tree])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    # predict
    acc = []
    if setting == 1:
        sdata, slabel = loaddata(1, shuffle=True)
        if dataset == 2:
            tdata = np.ndarray((param['sensornum'], 0, param['insize'], 1, 1))
            tlabel = np.ndarray((0, 7))
            for tbatch in range(2, 4):
                data, label = loaddata(batch=tbatch, shuffle=False)
                tdata = np.concatenate((tdata, data), axis=1)
                tlabel = np.concatenate((tlabel, label), axis=0)
        else:
            tdata = np.ndarray((0, 8, 16, 1))
            tlabel = np.ndarray((0, 6))
            for tbatch in range(2, 9):
                data, label = loaddata(batch=tbatch, shuffle=True)
                tdata = np.concatenate((tdata, data), axis=0)
                tlabel = np.concatenate((tlabel, label), axis=0)
        sdata = sdata.reshape(sdata.shape[0], sdata.shape[1], sdata.shape[2], sdata.shape[3], 1)  # expand dim
        tdata = tdata.reshape(tdata.shape[0], tdata.shape[1], tdata.shape[2], tdata.shape[3], 1)  # expand dim
        if dataset == 2:
            sdata = sdata.swapaxes(0, 1).swapaxes(1, 2)
            tdata = tdata.swapaxes(0, 1).swapaxes(1, 2)
        history = model.fit([sdata[:, :, s] for s in range(param['sensornum'])], slabel,
                            validation_data=[[tdata[:, :, s] for s in range(param['sensornum'])], tlabel],
                            batch_size=80, epochs=100, verbose=0)
        acc.append(history.history['val_acc'][-1])
    else:
        for tbatch in range(2, param['batchnum'] + 1):
            sdata, slabel = loaddata(tbatch - 1, shuffle=True)
            tdata, tlabel = loaddata(tbatch, shuffle=True)
            sdata = sdata.reshape(sdata.shape[0], sdata.shape[1], sdata.shape[2], sdata.shape[3], 1)  # expand dim
            tdata = tdata.reshape(tdata.shape[0], tdata.shape[1], tdata.shape[2], tdata.shape[3], 1)  # expand dim
            if dataset == 2:
                sdata = sdata.swapaxes(0, 1).swapaxes(1, 2)
                tdata = tdata.swapaxes(0, 1).swapaxes(1, 2)
            history = model.fit([sdata[:, :, s] for s in range(param['sensornum'])], slabel,
                                validation_data=[[tdata[:, :, s] for s in range(param['sensornum'])], tlabel],
                                batch_size=80, epochs=100, verbose=0)
            acc.append(history.history['val_acc'][-1])
    print("[INFO] ({}) Complete dataset {} setting {}, Average accuracy: {}.".format(inspect.stack()[0][3], dataset, setting, np.mean(acc)))
    # K.clear_session()


def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    tf.set_random_seed(seed)


if __name__ == '__main__':
    if tf.__version__[0] == '2':
        tf.disable_v2_behavior()
        config = ConfigProto()
        config.gpu_options.allow_growth = True
        session = InteractiveSession(config=config)
    setup_seed(42)

    auto_solver(dataset=2)
