"""
The RNN Kitchen 🍔
------------------
A Streamlit application for experimenting with Recurrent Neural Networks (RNNs).

Workflow:
    1. Preprocess text data
    2. Build & train an RNN model
    3. Generate text using the trained model

Dependencies:
    - streamlit
    - numpy
    - tensorflow.keras
    - backend.py (contains preprocess_data, build_model, train_or_retrain, generate_text)
"""

import streamlit as st
import numpy as np
from backend import preprocess_data, build_model, train_or_retrain, generate_text


# -------------------------
# Streamlit Config
# -------------------------
st.set_page_config(page_title='The RNN Kitchen',
                   page_icon='🍔',
                   layout='wide',
                   initial_sidebar_state='expanded')

st.title("🍔 The RNN Kitchen")
st.caption("Experiment with text preprocessing, RNN training, and text generation in one place.")

# Tabs
tabs = st.tabs(['Preprocess Data', 'Train Model', 'Generate Text'])


# -------------------------
# Tab 1: Data Preprocessing
# -------------------------
with tabs[0]:
    st.header("Step 1: Data Preprocessing")
    st.markdown("Upload your text dataset and prepare it for model training.")

    st.info("""
    💡 **Hint: preprocess_data()**
    --------------------
    Args:
        filepath (str): Path to the text dataset file.
        num_words (int): Max number of words in vocab.
        oov_token (str): Token for unknown words.
        lower (bool): Convert text to lowercase.
        padding (str): Sequence padding type ('pre' or 'post').
        fresh (bool): Reset tokenizer?

    Returns:
        tuple: (X, y, vocab_size, max_len, tokenizer)
    """)

    col1, col2, col3 = st.columns(3, gap="large")

    with col1:
        data_file = st.file_uploader("📂 Upload a .txt file", type=["txt"])
        data = ''
        if data_file is not None:
            data = data_file.read().decode("utf-8")
        max_words = st.number_input('Maximum Vocabulary', 2000, 50000, 5000)

    with col2:
        padding = st.selectbox('Padding Type', ['pre', 'post'])
        lower = st.checkbox('Lowercase Data', True)

    with col3:
        fresh = st.checkbox('Fresh Tokenizer', False)
        oov_token = st.text_input('OOV Token', "<OOV>")

    if st.button('🚀 Preprocess'):
        with st.spinner('Preprocessing...'):
            try:
                X, y, total_words, max_len, tokenizer = preprocess_data(
                    data=data,
                    num_words=max_words,
                    oov_token=oov_token,
                    lower=lower,
                    padding=padding,
                    fresh=fresh,
                )
                st.success(f"✅ Done! Vocabulary Size: {total_words}, Max Length: {max_len}")
                st.session_state["data"] = (X, y, total_words, max_len, tokenizer)
            except ValueError:
                st.error("⚠️ Invalid input values. Please check your parameters.")
            except Exception as e:
                st.error(f"❌ Error: {e}")


# -------------------------
# Tab 2: Build & Train Model
# -------------------------
with tabs[1]:
    st.header("Step 2: Build & Train Model")
    st.markdown("⚠️ **Warning:** Training may take a long time.")

    st.info("""
    💡 **Hint: Callbacks you can configure**
    - **ModelCheckpoint**: Save the best model based on validation accuracy/loss  
    - **EarlyStopping**: Stop training when no improvement  
    - **ReduceLROnPlateau**: Reduce learning rate if stuck  
    """)

    if "data" not in st.session_state:
        st.warning("⚠️ Please preprocess data first in **Tab 1**.")
    else:
        X, y, total_words, max_len, tokenizer = st.session_state["data"]

        col1, col2, col3 = st.columns(3)
        with col1:
            rnn_type = st.selectbox("RNN Type", ["GRU", "LSTM", "SimpleRNN"])
            rnn_units = st.number_input("RNN Units", 1, 512, 180)
            embedding_units = st.number_input("Embedding Units", 1, 300, 100)
        with col2:
            dropout_rate = st.slider("Dropout Rate", 0.0, 0.7, 0.3, 0.05)
            rnn_layers = st.number_input("Hidden Layers", 1, 3, 1)
            activation = st.selectbox("Dense Activation", ["softmax", "sigmoid"])
        with col3:
            optimizer = st.selectbox("Optimizer", ["adam", "rmsprop", "sgd"])
            lr = st.number_input("Learning Rate", 0.0001, 0.01, 0.001, format="%.4f")

        col4, col5 = st.columns(2)
        with col4:
            epochs = st.number_input("Epochs", 1, 500, 50)
            batch_size = st.number_input("Batch Size", 8, 4096, 128)
        with col5:
            validation_split = st.slider("Validation Split", 0.1, 0.5, 0.2)
            fresh_model = st.checkbox("Fresh Model", False)

        st.subheader("⚙️ Training Callbacks")
        col6, col7, col8 = st.columns(3)
        with col6:
            use_checkpoint = st.checkbox("ModelCheckpoint", True)
            save_best_only = st.checkbox("Save Best Only", True)
            monitor_ckpt = st.selectbox("Checkpoint Monitor", ["val_accuracy", "val_loss"], 0)
        with col7:
            use_earlystop = st.checkbox("EarlyStopping", True)
            patience_early = st.number_input("EarlyStopping Patience", 1, 20, 5)
        with col8:
            use_reducelr = st.checkbox("ReduceLROnPlateau", True)
            patience_lr = st.number_input("ReduceLR Patience", 1, 10, 2)
            factor_lr = st.slider("ReduceLR Factor", 0.1, 0.9, 0.5)

        # Strong warning before training
        st.error("""
        🚨 Training Warning:
        - Training can be heavy and may freeze/crash Streamlit.  
        - Do **NOT** press the button multiple times.  
        - Monitor training progress in Streamlit logs.  
        """)

        if st.button("🔥 Train Model"):
            with st.spinner("Training in progress... This may take a while."):
                try:
                    model = build_model(
                        total_words=total_words,
                        max_len=max_len,
                        mode=rnn_type,
                        rnn_units=rnn_units,
                        embedding_units=embedding_units,
                        dropout_rate=dropout_rate,
                        rnn_hidden_layers=rnn_layers,
                        dense_activation=activation,
                        opt=optimizer,
                        lr=lr,
                    )

                    trained_model, history, loss, acc = train_or_retrain(
                        X=X,
                        y=y,
                        model=model,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_split=validation_split,
                        fresh=fresh_model,
                        # CALLBACK CONFIG
                        model_checkpoint=use_checkpoint,
                        save_best_only=save_best_only,
                        checkpoint_monitor=monitor_ckpt,
                        earlystop=use_earlystop,
                        earlystop_patience=patience_early,
                        earlystop_monitor="val_loss",
                        reducelr=use_reducelr,
                        reduce_patience=patience_lr,
                        reduce_factor=factor_lr,
                    )

                    st.success(f"✅ Training Complete | Val Loss: {loss:.4f}, Val Accuracy: {acc:.4f}")
                    st.session_state["model"] = trained_model
                    st.session_state["max_len"] = max_len
                    st.session_state["tokenizer"] = tokenizer

                except Exception as e:
                    st.error(f"❌ Training failed: {e}")


# -------------------------
# Tab 3: Generate Text
# -------------------------
with tabs[2]:
    st.header("Step 3: Generate Text")
    st.markdown("Use your trained RNN model to generate text sequences.")

    st.info("""
    💡 **Hint: generate_text()**
    --------------------
    Args:
        seed_text (str): Starting phrase for generation  
        predict_next_words (int): Number of words to generate  
        temp (float): Sampling temperature  
        topK (int): Top-K sampling filter  
        topP (float): Top-P nucleus sampling filter  

    Returns:
        str: Generated continuation  
    """)

    if "model" not in st.session_state:
        st.warning("⚠️ Please train a model first in **Tab 2**.")
    else:
        seed_text = st.text_input("Seed Text", "Once upon a time")
        predict_words = st.number_input("Predict Next Words", 1, 50, 5)

        col1, col2, col3 = st.columns(3)
        with col1:
            temp = st.slider("Temperature", 0.1, 2.0, 1.0, 0.1)
        with col2:
            topK = st.number_input("Top-K", 0, 50, 0)
        with col3:
            topP = st.slider("Top-P (Nucleus)", 0.1, 1.0, 1.0, 0.05)

        if st.button("✨ Generate"):
            try:
                text = generate_text(
                    model=st.session_state["model"],
                    tokenizer=st.session_state["tokenizer"],
                    seed_text=seed_text,
                    max_len=st.session_state["max_len"],
                    predict_next_words=predict_words,
                    temp=temp,
                    topK=topK,
                    topP=topP,
                )
                st.success(f"📝 Generated Text:\n\n{text}")
            except Exception as e:
                st.error(f"❌ Generation failed: {e}")
