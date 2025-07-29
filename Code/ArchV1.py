import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv2D, BatchNormalization, ReLU, GlobalAveragePooling2D, Dense, Dropout,
    Add, DepthwiseConv2D, Multiply, Lambda
)
from tensorflow.keras.models import Model

# ------------------------------
# Multi-Head Feature Channel Attention (MHFCA) Block
# ------------------------------
def MHFCA_Block(x, reduction=16):
    channels = x.shape[-1]
    gap = GlobalAveragePooling2D()(x)
    fc1 = Dense(channels // reduction, activation='relu')(gap)
    fc2 = Dense(channels, activation='sigmoid')(fc1)
    scale = Lambda(lambda z: tf.expand_dims(tf.expand_dims(z, axis=1), axis=1))(fc2)
    return Multiply()([x, scale])

# ------------------------------
# TDCA-fused MBConv Block
# ------------------------------
def TDCA_MBConv(x, filters, stride):
    shortcut = x

    x = Conv2D(filters, kernel_size=3, strides=stride, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = MHFCA_Block(x)  # Integrating MHFCA inside TDCA block
    x = Conv2D(filters, kernel_size=1, strides=1, padding='same')(x)
    x = BatchNormalization()(x)

    if shortcut.shape[-1] != filters or stride != 1:
        shortcut = Conv2D(filters, kernel_size=1, strides=stride, padding='same')(shortcut)
        shortcut = BatchNormalization()(shortcut)

    return ReLU()(Add()([x, shortcut]))

# ------------------------------
# MHFCA-fused MBConv Block
# ------------------------------
def MHFCA_MBConv(x, filters, stride):
    shortcut = x

    x = Conv2D(filters, kernel_size=1, padding='same')(x)
    x = DepthwiseConv2D(kernel_size=3, strides=stride, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = MHFCA_Block(x)
    x = Conv2D(filters, kernel_size=1, padding='same')(x)
    x = BatchNormalization()(x)

    if shortcut.shape[-1] != filters or stride != 1:
        shortcut = Conv2D(filters, kernel_size=1, strides=stride, padding='same')(shortcut)
        shortcut = BatchNormalization()(shortcut)

    return ReLU()(Add()([x, shortcut]))

# ------------------------------
# CHA-LungCancer Architecture with TDCA and MHFCA Modules
# ------------------------------
def create_convnext_xt_tdca_mhfca_model(input_shape=(224, 224, 3), num_classes=4):
    inputs = Input(shape=input_shape)

    # Stage 0 - Initial Conv
    x = Conv2D(32, kernel_size=3, strides=2, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # Stage 1 - TDCA Block
    x = TDCA_MBConv(x, filters=64, stride=1)

    # Stage 2 - TDCA Block
    x = TDCA_MBConv(x, filters=96, stride=2)

    # Stage 3 - TDCA Block
    x = TDCA_MBConv(x, filters=128, stride=2)

    # Stage 4 - MHFCA Block
    x = MHFCA_MBConv(x, filters=160, stride=2)

    # Stage 5 - MHFCA Block
    x = MHFCA_MBConv(x, filters=192, stride=1)

    # Stage 6 - MHFCA Block
    x = MHFCA_MBConv(x, filters=256, stride=2)

    # Stage 7 - Final Convolution + Pooling
    x = Conv2D(256, kernel_size=1, padding='same')(x)
    x = GlobalAveragePooling2D()(x)

    # Fully Connected Classifier
    x = Dropout(0.4)(Dense(128, activation='relu')(x))
    x = Dropout(0.2)(Dense(64, activation='relu')(x))
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model
