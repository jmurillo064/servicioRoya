def crear_modeloEmbeddings():
    from tensorflow.keras import optimizers
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dropout, Flatten, Dense, Activation
    from tensorflow.keras.layers import BatchNormalization
    from tensorflow.keras.layers import  Convolution2D, MaxPooling2D
    from tensorflow.keras import backend as K
    longitud, altura = 250, 250
    filtrosConv1 = 32
    filtrosConv2 = 64
    tamano_filtro1 = (3, 3)
    tamano_filtro2 = (2, 2)
    tamano_pool = (2, 2)
    clases = 2
    lr = 0.0004

    cnn = Sequential()
    cnn.add(Convolution2D(filtrosConv1, tamano_filtro1, padding ="same", input_shape=(longitud, altura, 3), activation='relu'))
    cnn.add(MaxPooling2D(pool_size=tamano_pool))

    cnn.add(Convolution2D(filtrosConv2, tamano_filtro2, padding ="same"))
    cnn.add(MaxPooling2D(pool_size=tamano_pool))

    cnn.add(Convolution2D(filtrosConv2, tamano_filtro2, padding ="same"))
    cnn.add(MaxPooling2D(pool_size=tamano_pool))

    cnn.add(Flatten())
    cnn.add(Dense(256, activation='relu'))

    cnn.add(Dense(128, activation='relu'))

    cnn.add(Dense(64, activation='relu'))
    cnn.add(BatchNormalization())

    cnn.add(Dense(clases, activation='softmax'))

    cnn.compile(loss='binary_crossentropy',
               optimizer=optimizers.Adam(
                    learning_rate=lr,
                ),
                metrics=['accuracy'])

    return cnn