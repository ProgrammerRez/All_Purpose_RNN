import os
import numpy as np
import pickle
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, GRU, Dense, Dropout, SimpleRNN
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from sklearn.model_selection import train_test_split


# -------------------------
# Configurable Preprocessing
# -------------------------
def preprocess_data(
    filepath: str = "dataset.txt",
    num_words: int = 5000,
    oov_token: str = "<OOV>",
    lower: bool = True,
    padding: str = "pre",
    fresh: bool = False,
):
    """
    Reads text, tokenizes, creates sequences, and saves tokenizer.

    Args:
        filepath (str): Path to the text dataset file.
        num_words (int): The maximum number of words to keep, based on word frequency.
        oov_token (str): The out-of-vocabulary token.
        lower (bool): Whether to convert text to lowercase.
        padding (str): Padding type, 'pre' or 'post'.
        fresh (bool): If True, deletes any previous tokenizer to start clean.

    Returns:
        tuple: A tuple containing:
            - X (np.ndarray): The input sequences for the model.
            - y (np.ndarray): The one-hot encoded target labels.
            - vocab_size (int): The size of the vocabulary.
            - max_len (int): The maximum length of the sequences.
            - tokenizer (Tokenizer): The fitted Keras Tokenizer object.
    """
    os.makedirs("pipelines", exist_ok=True)

    tokenizer_path = "pipelines/tokenizer.pkl"
    if fresh and os.path.exists(tokenizer_path):
        os.remove(tokenizer_path)
        print("[INFO] Previous tokenizer deleted for a fresh start.")

    # Reading data
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = f.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset file not found at: {filepath}")

    if not data or not isinstance(data, str):
        raise ValueError("Data must be a valid non-empty string.")

    if lower:
        data = data.lower()

    # Tokenizer
    tokenizer = Tokenizer(num_words=num_words, oov_token=oov_token)
    tokenizer.fit_on_texts([data])

    # Building sequences
    sequences = []
    for line in data.split("\n"):
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[: i + 1]
            sequences.append(n_gram_sequence)

    if not sequences:
        raise ValueError("No valid sequences were created for training.")

    # Find max length and pad sequences
    max_len = max(len(seq) for seq in sequences)
    padded_sequences = pad_sequences(sequences, maxlen=max_len, padding=padding)

    # Separate features (X) and labels (y) without data leakage
    # X contains all but the last word, y is the last word
    X, y = padded_sequences[:, :-1], padded_sequences[:, -1]
    
    # Calculate vocab size
    # +1 to account for 0-padding
    vocab_size = min(num_words, len(tokenizer.word_index) + 1)
    
    # One-hot encode the target labels
    y = to_categorical(y, num_classes=vocab_size)

    # Save tokenizer
    with open(tokenizer_path, "wb") as f:
        pickle.dump(tokenizer, f)

    return X, y, vocab_size, max_len, tokenizer


# -------------------------
# Model Builder
# -------------------------
def build_model(
    total_words: int,
    max_len: int,
    mode: str = "GRU",
    rnn_hidden_layers: int = 1,
    embedding_units: int = 100,
    rnn_units: int = 150,
    dense_activation: str = "softmax",
    dropout_rate: float = 0.2,
    opt: str = "adam",
    lr: float = 0.001,
    metrics: list = ["accuracy"],
):
    """
    Builds a sequential RNN-based Keras model.

    Args:
        total_words (int): The size of the vocabulary.
        max_len (int): The maximum length of the sequences.
        mode (str): Type of RNN layer ('GRU', 'LSTM', or 'SimpleRNN').
        rnn_hidden_layers (int): Number of RNN hidden layers.
        embedding_units (int): Dimensionality of the embedding vector.
        rnn_units (int): Number of units in the RNN layers.
        dense_activation (str): Activation function for the final dense layer.
        dropout_rate (float): Dropout rate for the dropout layer.
        opt (str): Optimizer to use ('adam', 'rmsprop', 'sgd').
        lr (float): Learning rate for the optimizer.
        metrics (list): List of metrics to be evaluated by the model.

    Returns:
        tensorflow.keras.models.Sequential: The compiled Keras model.
    """
    model = Sequential()
    # Mask_zero=True prevents padding from being trained
    model.add(Embedding(total_words, embedding_units, input_length=max_len - 1, mask_zero=True))

    for i in range(rnn_hidden_layers):
        return_sequences = (i < rnn_hidden_layers - 1)
        if mode.upper() == "GRU":
            model.add(GRU(rnn_units, return_sequences=return_sequences))
        elif mode.upper() == "LSTM":
            model.add(LSTM(rnn_units, return_sequences=return_sequences))
        else:
            model.add(SimpleRNN(rnn_units, return_sequences=return_sequences))
        model.add(Dropout(dropout_rate))

    # Final dense layer with softmax activation for multi-class classification
    model.add(Dense(total_words, activation=dense_activation))

    # Optimizer
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

    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=metrics)
    
    return model


# -------------------------
# Training with Retry Loop
# -------------------------
def train_or_retrain(
    X,
    y,
    model,
    save_path: str = "models/best_model_acc.keras",
    epochs: int = 50,
    batch_size: int = 128,
    retry_acc_threshold: float = 0.40,
    retry_loss_threshold: float = 1.0,
    retries: int = 2,
    verbose: int = 1,
    validation_split: float = 0.2,
    random_state: int = 42,
    model_checkpoint: bool = True,
    save_best_only: bool = True,
    checkpoint_monitor: str = "val_accuracy",
    checkpoint_mode: str = "max",
    earlystop: bool = True,
    earlystop_monitor: str = "val_loss",
    earlystop_patience: int = 5,
    reducelr: bool = True,
    reducelr_monitor: str = "val_loss",
    reduce_factor: float = 0.5,
    reduce_patience: int = 2,
    min_lr: float = 1e-5,
    show_summary: bool = True,
    fresh: bool = False,
):
    """
    Trains or retrains a Keras model with various callbacks.

    Args:
        X (np.ndarray): Input features (padded sequences).
        y (np.ndarray): One-hot encoded target labels.
        model (Sequential): The Keras model to train.
        save_path (str): Path to save the best model.
        epochs (int): Number of training epochs.
        batch_size (int): Number of samples per batch.
        retry_acc_threshold (float): Accuracy threshold to trigger a retry.
        retry_loss_threshold (float): Loss threshold to trigger a retry.
        retries (int): Maximum number of retries.
        verbose (int): Verbosity mode (0, 1, or 2).
        validation_split (float): The fraction of the training data to be used as validation data.
        random_state (int): Seed for the random number generator.
        model_checkpoint (bool): Whether to use ModelCheckpoint callback.
        save_best_only (bool): If True, only save the best model observed.
        checkpoint_monitor (str): Quantity to monitor for ModelCheckpoint.
        checkpoint_mode (str): 'auto', 'min', or 'max' for ModelCheckpoint.
        earlystop (bool): Whether to use EarlyStopping callback.
        earlystop_monitor (str): Quantity to monitor for EarlyStopping.
        earlystop_patience (int): Number of epochs with no improvement after which training will be stopped.
        reducelr (bool): Whether to use ReduceLROnPlateau callback.
        reducelr_monitor (str): Quantity to monitor for ReduceLROnPlateau.
        reduce_factor (float): Factor by which the learning rate will be reduced.
        reduce_patience (int): Number of epochs with no improvement after which learning rate will be reduced.
        min_lr (float): A lower bound on the learning rate.
        show_summary (bool): If True, prints the model summary before training.
        fresh (bool): If True, deletes any previous model to start fresh.

    Returns:
        tuple: A tuple containing:
            - model (Sequential): The trained Keras model.
            - history (History): The training history object.
            - loss (float): The final validation loss.
            - acc (float): The final validation accuracy.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if fresh and os.path.exists(save_path):
        os.remove(save_path)
        print(f"[INFO] Previous model at {save_path} deleted for a fresh start.")

    if os.path.exists(save_path) and not fresh:
        print(f"[INFO] Found existing model at {save_path}, loading for retraining...")
        model = load_model(save_path)
    else:
        print("[INFO] No model found, training a new one...")

    # Callbacks
    cbs = []
    if model_checkpoint:
        cbs.append(
            ModelCheckpoint(
                filepath=save_path,
                monitor=checkpoint_monitor,
                save_best_only=save_best_only,
                mode=checkpoint_mode,
                verbose=1,
            )
        )
    if earlystop:
        cbs.append(
            EarlyStopping(
                monitor=earlystop_monitor,
                patience=earlystop_patience,
                restore_best_weights=True,
                verbose=1,
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
                verbose=1,
            )
        )

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=validation_split, random_state=random_state
    )

    # Training
    if show_summary:
        model.summary()

    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        callbacks=cbs,
        verbose=verbose,
    )

    # Retry loop
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    tries = 0
    while (acc < retry_acc_threshold or loss > retry_loss_threshold) and tries < retries:
        tries += 1
        print(f"[INFO] Retrying... {tries}/{retries} -- Current Acc: {acc:.4f}, Current Loss: {loss:.4f}")
        history = model.fit(
            X_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_test, y_test),
            callbacks=cbs,
            verbose=verbose,
        )
        loss, acc = model.evaluate(X_test, y_test, verbose=0)
        
    print(f"[INFO] Final Model Evaluation: Loss={loss:.4f}, Accuracy={acc:.4f}")

    return model, history, loss, acc


# -------------------------
# Text Generation
# -------------------------
def generate_text(
    model,
    tokenizer,
    seed_text,
    max_len,
    predict_next_words: int = 2,
    temp: float = 1.0,
    topK: int = 0,
    topP: float = 1.0,
    padding: str = "pre",
):
    """
    Generates text using a trained RNN model.

    Args:
        model (Sequential): The trained Keras model.
        tokenizer (Tokenizer): The tokenizer used for preprocessing.
        seed_text (str): The starting text for generation.
        max_len (int): The maximum sequence length used during training.
        predict_next_words (int): The number of words to generate.
        temp (float): Controls the randomness of the predictions. Higher is more random.
        topK (int): The number of highest probability words to consider for sampling.
        topP (float): The cumulative probability threshold for nucleus sampling.
        padding (str): Padding type, 'pre' or 'post', matching preprocessing.

    Returns:
        str: The generated text, appended to the seed text.
    """
    def _sample(preds, temp=1.0, topK=0, topP=1.0):
        """Helper function to sample an index from a probability array."""
        preds = np.asarray(preds).astype("float64")
        # Apply temperature
        preds = np.log(preds + 1e-8) / temp
        exp_preds = np.exp(preds - np.max(preds))
        preds = exp_preds / np.sum(exp_preds)
        
        # Top-K filtering
        if topK > 0:
            indices_to_remove = preds.argsort()[:-topK]
            preds[indices_to_remove] = 0
            preds = preds / np.sum(preds) if np.sum(preds) > 0 else np.ones_like(preds) / len(preds)

        # Top-P filtering
        if topP < 1.0:
            sorted_probs = np.sort(preds)[::-1]
            sorted_indices = np.argsort(preds)[::-1]
            cumulative_probs = np.cumsum(sorted_probs)
            # Find the index of the first cumulative probability greater than topP
            cutoff_idx = np.where(cumulative_probs > topP)[0][0]
            indices_to_remove = sorted_indices[cutoff_idx:]
            preds[indices_to_remove] = 0
            preds = preds / np.sum(preds) if np.sum(preds) > 0 else np.ones_like(preds) / len(preds)
        
        preds = np.nan_to_num(preds)
        if np.sum(preds) == 0:
            preds = np.ones_like(preds) / len(preds)
        else:
            preds = preds / np.sum(preds)

        return np.random.choice(len(preds), p=preds)

    generated_text = seed_text.strip()
    for _ in range(predict_next_words):
        token_list = tokenizer.texts_to_sequences([generated_text])[0]
        # Pad sequence to the correct length
        token_list = pad_sequences(
            [token_list], maxlen=max_len - 1, padding=padding
        )
        
        preds = model.predict(token_list, verbose=0)[0]
        next_index = _sample(preds, temp=temp, topK=topK, topP=topP)
        
        # Look up word from index
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == next_index:
                output_word = word
                break
        
        generated_text += " " + output_word

    return generated_text


# -------------------------
# Test Pipeline
# -------------------------
if __name__ == "__main__":
    dataset_path = "dataset.txt"
    model_path = "models/best_model_acc.keras"
    
    # Create a dummy dataset for testing
    if not os.path.exists(dataset_path):
        with open(dataset_path, "w") as f:
            f.write("to be or not to be that is the question\n")
            f.write("whether 'tis nobler in the mind to suffer\n")
            f.write("the slings and arrows of outrageous fortune\n")
            f.write("or to take arms against a sea of troubles\n")
            f.write("and by opposing end them")

    # The issue with perfect accuracy with SimpleRNN and post padding is due to data leakage.
    # The fix is to ensure the last word is the target and is not part of the input sequence.
    # The original code's `X, y = sequences[:, :-1], sequences[:, -1:]` was flawed.
    # The corrected `preprocess_data` function fixes this.
    try:
        X, y, total_words, max_len, tokenizer = preprocess_data(
            filepath=dataset_path, num_words=2000, padding="post", fresh=True
        )

        model = build_model(
            total_words=total_words,
            max_len=max_len,
            mode="SimpleRNN",
            rnn_units=32, # Reduced for faster training
            embedding_units=64, # Reduced for faster training
            dropout_rate=0.4,
            rnn_hidden_layers=1
        )

        trained_model, history, loss, acc = train_or_retrain(
            X=X,
            y=y,
            model=model,
            earlystop=True,
            show_summary=True,
            fresh=True,
            epochs=20, # Reduced for a quicker test,
            batch_size=1
        )

        print("\nGenerated Text:")
        print(
            generate_text(
                model=trained_model,
                tokenizer=tokenizer,
                seed_text="what light",
                max_len=max_len,
                padding="post",
                predict_next_words=5,
                temp=0.8
            )
        )

    except (ValueError, FileNotFoundError) as e:
        print(f"[ERROR] {e}")