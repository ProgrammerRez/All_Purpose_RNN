import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import *
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import *
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

def preprocess_text(data):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts([data])
    total_words = len(tokenizer.word_index)+1

    input_seqs = []
    for line in data.split('\n'):
        tokens = tokenizer.texts_to_sequences([line])[0]
        for i in range(1,len(tokens)):
            n_gram_seq = tokens[:i+1]
            input_seqs.append(n_gram_seq)

    max_len = max([len(seq) for seq in input_seqs])
    input_seqs = pad_sequences(input_seqs,maxlen=max_len,padding='pre')

    X = input_seqs[:,:-1]
    y = to_categorical(input_seqs[:,-1:],num_classes=total_words)

    return X,y

def train_model(X,y):
    model = Sequential()
    model.add(Embedding(total_words, 120, input_length=max_len-1))
    model.add(GRU(150))
    model.add(Dense(total_words, activation='softmax'))

    # -------------------------
    # Checkpoints
    # -------------------------
    checkpoint_acc = ModelCheckpoint(
        "models/best_model_acc.keras",
        monitor="accuracy",
        save_best_only=True,
        mode="max",
        verbose=1,
    )


    # -------------------------
    # Early Stopping
    # -------------------------
    early_stop = EarlyStopping(
        monitor="loss",  # usually better to stop on loss
        patience=5,
        restore_best_weights=True,
        verbose=1,
    )

    # -------------------------
    # Reduce LR on Plateau
    # -------------------------
    reduce_lr = ReduceLROnPlateau(
        monitor="accuracy",   # reduce LR if accuracy plateaus
        factor=0.5,
        patience=2,
        min_lr=1e-5,
        verbose=1,
        mode="max",
    )

    # -------------------------
    # Optimizer
    # -------------------------
    optimizer = Adam(learning_rate=0.001)

    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    history = model.fit(
        X,y,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        callbacks=[checkpoint_acc, early_stop],
        verbose=1,)
    
    return model

