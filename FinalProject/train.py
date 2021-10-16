from init import *
from checkpoints import *
from models import *


def train_discriminative_model(labeled, unlabeled, input_shape, gpu=1):
    """
    A function that trains and returns a discriminative model on the labeled and unlabaled data.
    """

    # create the binary dataset:
    y_L = np.zeros((labeled.shape[0], 1), dtype='int')
    y_U = np.ones((unlabeled.shape[0], 1), dtype='int')
    X_train = np.vstack((labeled, unlabeled))
    Y_train = np.vstack((y_L, y_U))
    Y_train = to_categorical(Y_train)

    # build the model:
    model = get_discriminative_model(input_shape)

    # train the model:
    batch_size = 1024
    if np.max(input_shape) == 28:
        optimizer = optimizers.Adam(lr=0.0003)
        epochs = 20
    elif np.max(input_shape) == 128:
        # optimizer = optimizers.Adam(lr=0.0003)
        # epochs = 200
        batch_size = 128
        optimizer = optimizers.Adam(lr=0.0001)
        epochs = 20  # TODO: was 200
    elif np.max(input_shape) == 512:
        optimizer = optimizers.Adam(lr=0.0002)
        # optimizer = optimizers.RMSprop()
        epochs = 20
    elif np.max(input_shape) == 32:
        optimizer = optimizers.Adam(lr=0.0003)
        epochs = 20
    else:
        optimizer = optimizers.Adam()
        # optimizer = optimizers.RMSprop()
        epochs = 20
        batch_size = 32

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer, metrics=['accuracy'])
    callbacks = [DiscriminativeEarlyStopping()]
    model.fit(X_train, Y_train,
              epochs=epochs,
              batch_size=batch_size,
              shuffle=True,
              callbacks=callbacks,
              class_weight={0: float(X_train.shape[0]) / Y_train[Y_train == 0].shape[0],
                            1: float(X_train.shape[0]) / Y_train[Y_train == 1].shape[0]},
              verbose=2)

    return model


def train_mnist(X_train, Y_train, X_validation, Y_validation, checkpoint_path, gpu=1):

    if K.image_data_format() == 'channels_last':
        input_shape = (28, 28, 1)
    else:
        input_shape = (1, 28, 28)

    model = LeNet(input_shape=input_shape, labels=10)
    optimizer = optimizers.Adam()
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer, metrics=['accuracy'])
    callbacks = [Checkpoint(
        filepath=checkpoint_path, verbose=1, weights=True)]

    model.fit(X_train, Y_train,
              epochs=150,
              batch_size=32,
              shuffle=True,
              validation_data=(X_validation, Y_validation),
              callbacks=callbacks,
              verbose=2)
    model.load_weights(checkpoint_path)
    return model


def train_cifar10(X_train, Y_train, X_validation, Y_validation, checkpoint_path, gpu=1):
    if K.image_data_format() == 'channels_last':
        input_shape = (32, 32, 3)
    else:
        input_shape = (3, 32, 32)

    model = VGG(input_shape=input_shape, labels=10)
    optimizer = optimizers.Adam()
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer, metrics=['accuracy'])
    callbacks = [Checkpoint(
        filepath=checkpoint_path, verbose=1, weights=True)]

    model.fit(X_train, Y_train,
              epochs=15,
              batch_size=50,
              shuffle=True,
              validation_data=(X_validation, Y_validation),
              callbacks=callbacks,
              verbose=2)

    # model.load_weights(checkpoint_path)
    return model


def train_cifar100(X_train, Y_train, X_validation, Y_validation, checkpoint_path, gpu=1):
    if K.image_data_format() == 'channels_last':
        input_shape = (32, 32, 3)
    else:
        input_shape = (3, 32, 32)

    model = VGG(input_shape=input_shape, labels=100)
    optimizer = optimizers.Adam(lr=0.0001)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer, metrics=['accuracy'])
    callbacks = [Checkpoint(
        filepath=checkpoint_path, verbose=1, weights=True)]

    model.fit(X_train, Y_train,
              epochs=25,
              batch_size=150,
              shuffle=True,
              validation_data=(X_validation, Y_validation),
              callbacks=callbacks,
              verbose=2)

    model.load_weights(checkpoint_path)
    return model


def train_mobilenet(X_train, Y_train, X_validation, Y_validation, checkpoint_path, gpu=1):
    if K.image_data_format() == 'channels_last':
        input_shape = (32, 32, 3)
    else:
        input_shape = (3, 32, 32)

    model = MobileNet_pretrain(input_shape=input_shape, labels=10)
    for layer in model.layers:
        layer.trainable = False
    for layer in model.layers[:20]:
        layer.trainable = False
    for layer in model.layers[20:]:
        layer.trainable = True
    optimizer = optimizers.Adam(lr=0.0001)

    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy', metrics=['accuracy'])

    callbacks = [Checkpoint(
        filepath=checkpoint_path, verbose=1, weights=True)]

    model.fit(X_train, Y_train,
              epochs=350,
              batch_size=50,
              shuffle=True,
              validation_data=(X_validation, Y_validation),
              callbacks=callbacks,
              verbose=2)

    model.load_weights(checkpoint_path)
    return model
