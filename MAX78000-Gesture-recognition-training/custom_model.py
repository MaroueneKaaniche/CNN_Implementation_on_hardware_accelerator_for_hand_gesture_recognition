import tensorflow as tf
import ai8xTF


#model architecture 

model= tf.keras.Sequential([

    tf.keras.Input(shape=(40,1)),
    ai8xTF.FusedMaxPoolConv1DReLU(filters=25,kernel_size=3),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Flatten(),
    ai8xTF.FusedDenseReLU(128),
    ai8xTF.FusedDenseReLU(64),
    ai8xTF.FusedDense(10),
])
# ai8xTF.MaxPool1D(pool_size=3,pool_strides=2,padding="same"),
# tf.keras.layers.Dense(10,activation='softmax'),
    # tf.keras.layers.BatchNormalization(),
lr_schedule = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_accuracy',
    mode='max',
    factor=0.2,
    patience=3,
    verbose=1,
    min_lr=1e-2)


# sig_shape = (X_fmg_train.shape[1], 1)
    # Signal_Inputs = Input(shape=sig_shape, name='Signal_Inputs')
    # conv1 = Conv1D(filters=8, kernel_size= 3, activation='relu', input_shape=sig_shape)(Signal_Inputs)
    # conv1 = BatchNormalization()(conv1)
    # pool1 = MaxPool1D(pool_size=(3), strides=(2), padding="same")(conv1)
    # D=Dropout(0.2)(pool1)

    # flat1 = Flatten()(D)
    # hidden1 = Dense(128,activation='relu')(flat1)
    # hidden2 = Dense(64,activation='relu')(hidden1)
    # output = Dense(10, activation='softmax')(hidden2)


# python train.py --epochs 250 --optimizer Adam --lr 0.001  --model custom_model --dataset customdataloader_tf





