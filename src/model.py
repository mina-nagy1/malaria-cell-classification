# src/model.py

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Rescaling, Conv2D, MaxPool2D, Dropout, BatchNormalization, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import TruePositives, TrueNegatives, FalsePositives, FalseNegatives, BinaryAccuracy, Precision, Recall, AUC
from .config import IMAGE_SIZE, LEARNING_RATE


def build_model():
    model = Sequential([
        InputLayer(shape=(IMAGE_SIZE, IMAGE_SIZE, 3)),
        Rescaling(1.0 / 255),

        Conv2D(filters=6, kernel_size=3, strides=1, padding='valid', activation='relu'),
        MaxPool2D(pool_size=2, strides=2),
        Dropout(0.05),

        Conv2D(filters=16, kernel_size=3, strides=1, padding='valid', activation='relu'),
        BatchNormalization(),
        MaxPool2D(pool_size=2, strides=2),

        Flatten(),
        Dense(100, activation='relu'),
        BatchNormalization(),
        Dropout(0.05),

        Dense(10, activation='relu'),
        BatchNormalization(),

        Dense(1, activation='sigmoid')
    ])

    metrics = [
        TruePositives(name='tp'),
        TrueNegatives(name='tn'),
        FalsePositives(name='fp'),
        FalseNegatives(name='fn'),
        BinaryAccuracy(name='accuracy'),
        Precision(name='precision'),
        Recall(name='recall'),
        AUC(name='auc')
    ]

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss=BinaryCrossentropy(),
        metrics=metrics
    )

    return model
