import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GRU, LSTM, SimpleRNN, Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from sklearn.model_selection import train_test_split

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"


# ---------------------------------
# Data Preprocessing
# ---------------------------------
def preprocess_text(
    data: str,
    lower: bool = True,               # convert to lowercase
    tokenizer_num_words: int = None,  # limit vocab size
    oov_token: str = None,            # handle out-of-vocab
    padding: str = "pre"              # "pre" or "post"
):
    """
    Returns:
        X: (n_samples, seq_len-1) integer sequences
        y: (n_samples, vocab_size) one-hot labels
        tokenizer: fitted Tokenizer
        max_len: int (sequence length)
        total_words: vocab size (int)
    """
    if not data or not isinstance(data, str):
        raise ValueError("`data` must be a non-empty string")

    if lower:
        data = data.lower()

    # Fit tokenizer on the whole text (you could pass list of lines too)
    tokenizer = Tokenizer(num_words=tokenizer_num_words, oov_token=oov_token)
    tokenizer.fit_on_texts([data])
    total_words = len(tokenizer.word_index) + 1

    input_seqs = []
    for line in data.split("\n"):
        line = line.strip()
        if not line:
            continue
        tokens = tokenizer.texts_to_sequences([line])[0]
        if len(tokens) < 2:
            continue
        for i in range(1, len(tokens)):
            n_gram_seq = tokens[: i + 1]
            input_seqs.append(n_gram_seq)

    if len(input_seqs) == 0:
        raise ValueError("No valid input sequences were generated from the data")

    max_len = max(len(seq) for seq in input_seqs)
    input_seqs = pad_sequences(input_seqs, maxlen=max_len, padding=padding)

    X = input_seqs[:, :-1]                      # features
    y_int = input_seqs[:, -1]                   # integer labels
    y = to_categorical(y_int, num_classes=total_words)  # one-hot

    return X, y, tokenizer, max_len, total_words

# ---------------------------------
# Model Training
# ---------------------------------
def train_model(
    X,
    y,
    total_words,
    max_len,
    save_path="models/best_model_acc.keras",

    # architecture
    mode="GRU",                 # "GRU", "LSTM", "SimpleRNN"
    RNN_hidden_layers=1,        # number of stacked RNN layers
    RNN_units=150,
    Embedding_units=100,
    dense_activation="softmax",

    # optimizer / training params
    opt="adam",                 # "adam", "rmsprop", "sgd", or keras optimizer instance
    lr=0.001,
    epochs=100,
    batch_size=32,

    # dataset splitting
    validation_split=0.2,
    random_state=42,

    # callbacks (tunable)
    model_checkpoint=True,
    checkpoint_monitor="val_accuracy",
    checkpoint_mode="max",
    earlystop=True,
    earlystop_monitor="val_loss",
    earlystop_patience=5,
    reducelr=True,
    reducelr_monitor="val_accuracy",
    reduce_factor=0.5,
    reduce_patience=2,
    min_lr=1e-5,

    # retry mechanism
    retries=3,                  # number of retry attempts
    retry_acc_threshold=0.5,    # minimum accuracy required
    retry_loss_threshold=2.0,   # maximum loss allowed
    verbose=1,
    show_summary=True
):
    """
    Build, compile, and train the model with retries.
    Returns: model, history, loss, acc
    """

    # -------------------------
    # Ensure save directory exists
    # -------------------------
    save_dir = os.path.dirname(save_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    # -------------------------X
    # Function to build model
    # -------------------------
    def build_model():
        model = Sequential()
        model.add(Embedding(total_words, Embedding_units, input_length=max_len - 1))

        for layer_idx in range(RNN_hidden_layers):
            return_seq = layer_idx < (RNN_hidden_layers - 1)
            if mode.upper() == "GRU":
                model.add(GRU(RNN_units, return_sequences=return_seq))
            elif mode.upper() == "LSTM":
                model.add(LSTM(RNN_units, return_sequences=return_seq))
            elif mode.upper() == "SIMPLERNN":
                model.add(SimpleRNN(RNN_units, return_sequences=return_seq))
            else:
                raise ValueError("Unsupported RNN mode. Choose 'GRU', 'LSTM', or 'SimpleRNN'.")

        model.add(Dense(total_words, activation=dense_activation))

        # Optimizer selection
        if isinstance(opt, str):
            opt_lower = opt.lower()
            if opt_lower == "adam":
                optimizer = Adam(learning_rate=lr)
            elif opt_lower == "rmsprop":
                optimizer = RMSprop(learning_rate=lr)
            elif opt_lower == "sgd":
                optimizer = SGD(learning_rate=lr)
            else:
                raise ValueError(f"Unknown optimizer name: {opt}")
        else:
            optimizer = opt

        model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
        return model

    # -------------------------
    # Prepare callbacks
    # -------------------------
    def get_callbacks():
        cbs = []
        if model_checkpoint:
            cbs.append(
                ModelCheckpoint(
                    filepath=save_path,
                    monitor=checkpoint_monitor,
                    save_best_only=False,   # always overwrite latest
                    mode=checkpoint_mode,
                    verbose=1
                )
            )
        if earlystop:
            cbs.append(
                EarlyStopping(
                    monitor=earlystop_monitor,
                    patience=earlystop_patience,
                    restore_best_weights=True,
                    verbose=1
                )
            )
        if reducelr:
            mode_arg = "max" if "acc" in reducelr_monitor.lower() else "min"
            cbs.append(
                ReduceLROnPlateau(
                    monitor=reducelr_monitor,
                    factor=reduce_factor,
                    patience=reduce_patience,
                    min_lr=min_lr,
                    mode=mode_arg,
                    verbose=1
                )
            )
        return cbs

    # -------------------------
    # Train/test split
    # -------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=validation_split, random_state=random_state
    )

    # -------------------------
    # Initial training
    # -------------------------
    model = build_model()
    if show_summary:
        model.summary()

    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        callbacks=get_callbacks(),
        verbose=verbose
    )

    loss, acc = model.evaluate(X_test, y_test, verbose=0)

    # -------------------------
    # Retry loop if poor performance
    # -------------------------
    tries = 0
    while (acc < retry_acc_threshold or loss > retry_loss_threshold) and tries < retries:
        tries += 1
        print(f"[INFO] Retry {tries}/{retries} — acc: {acc:.4f}, loss: {loss:.4f}")

        model = build_model()
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_test, y_test),
            callbacks=get_callbacks(),
            verbose=verbose
        )
        loss, acc = model.evaluate(X_test, y_test, verbose=0)

    return model, history, loss, acc


# ---------------------------------
# Text Generation
# ---------------------------------
def generate_text(
    model, tokenizer, seed_text, max_len,
    predict_next_words: int = 2,
    temp: float = 1.0,
    topK: int = 0,
    topP: float = 1.0,
    padding: str = "pre"
):
    """
    Sampling-based next-word generation supporting temperature, top-k, top-p.
    """

    def respond(preds, temp=1.0, topK=0, topP=1.0):
        preds = np.asarray(preds).astype("float64")
        preds = np.log(preds + 1e-8) / temp
        exp_preds = np.exp(preds - np.max(preds))
        preds = exp_preds / np.sum(exp_preds)

        # Top-K filter
        if topK > 0:
            indices = np.argpartition(preds, -topK)[-topK:]
            probs = np.zeros_like(preds)
            probs[indices] = preds[indices]
            preds = probs / np.sum(probs) if np.sum(probs) > 0 else preds

        # Top-P (nucleus) filter
        if topP < 1.0:
            sorted_indices = np.argsort(preds)[::-1]
            sorted_probs = np.sort(preds)[::-1]
            cum_probs = np.cumsum(sorted_probs)
            cutoff = cum_probs <= topP
            cutoff_indices = sorted_indices[cutoff]
            probs = np.zeros_like(preds)
            probs[cutoff_indices] = preds[cutoff_indices]
            preds = probs / np.sum(probs) if np.sum(probs) > 0 else preds

        preds = np.nan_to_num(preds)
        if np.sum(preds) == 0:
            preds = np.ones_like(preds) / len(preds)
        else:
            preds = preds / np.sum(preds)
        return np.random.choice(len(preds), p=preds)

    input_text = seed_text.strip()
    for _ in range(predict_next_words):
        token_list = tokenizer.texts_to_sequences([input_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_len - 1, padding=padding)
        preds = model.predict(token_list, verbose=0)[0]
        next_index = respond(preds, temp=temp, topK=topK, topP=topP)

        # map index -> token (inefficient loop but OK for small vocabs)
        for word, index in tokenizer.word_index.items():
            if index == next_index:
                input_text += " " + word
                break

    return input_text


# ---------------------------------
# Plot helpers
# ---------------------------------
def plot_history(history):
    if history is None:
        return
    h = history.history
    if "accuracy" in h:
        plt.figure(figsize=(8, 4))
        plt.plot(h["accuracy"], label="train_acc")
        if "val_accuracy" in h:
            plt.plot(h["val_accuracy"], label="val_acc")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.tight_layout()
        plt.show()
    if "loss" in h:
        plt.figure(figsize=(8, 4))
        plt.plot(h["loss"], label="train_loss")
        if "val_loss" in h:
            plt.plot(h["val_loss"], label="val_loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.tight_layout()
        plt.show()


# ---------------------------------
# Test Function / Entry
# ---------------------------------
def test_pipeline(
    data_path: str = "data.txt",
    rnn_mode: str = "LSTM",
    rnn_hidden_layers: int = 5,
    rnn_units: int = 150,
    embedding_dim: int = 200,
    optimizer: str = "adam",
    learning_rate: float = 0.01,
    epochs: int = 50,
    batch_size: int = 16,
    predict_words: int = 2,
    temp: float = 0.8,
    topK: int = 5,
    topP: float = 0.8,
    plot_history_flag: bool = True
):
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"{data_path} not found. Put training text in that file.")

    with open(data_path, "r", encoding="utf-8") as f:
        data = f.read()

    print("\n[INFO] Preprocessing...")
    X, y, tokenizer, max_len, total_words = preprocess_text(data)

    print("[INFO] Training...")
    model, history, loss, acc = train_model(
        X, y, total_words, max_len,
        save_path="models/best_model.keras",
        mode=rnn_mode,
        RNN_hidden_layers=rnn_hidden_layers,
        RNN_units=rnn_units,
        Embedding_units=embedding_dim,
        opt=optimizer,
        lr=learning_rate,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.4,
        earlystop=False,
        earlystop_patience=20,
        reducelr=True,
        reducelr_monitor="val_loss",
        reduce_factor=0.5,
        retries=10,
        retry_acc_threshold=0.5,
        retry_loss_threshold=2.0,
        verbose=1,
        show_summary=False
    )

    print(f"[INFO] Finished training. Eval -> loss: {loss:.4f}, acc: {acc:.4f}")

    if plot_history_flag:
        plot_history(history)

    print("[INFO] Generating sample text...")
    sample = generate_text(model, tokenizer, seed_text="to be", max_len=max_len,
                           predict_next_words=predict_words, temp=temp, topK=topK, topP=topP)
    print("\nGenerated Text:\n", sample)
    return model, history, tokenizer


# ---------------------------------
# Run if called directly
# ---------------------------------
if __name__ == "__main__":
    test_pipeline()
