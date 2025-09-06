"""
The RNN Kitchen 🍔
------------------
A Streamlit application for experimenting with Recurrent Neural Networks (RNNs).

This app provides a simple 3-step workflow:
    1. Preprocess text data
    2. Build and train an RNN model
    3. Generate text using the trained model

Dependencies:
    - streamlit
    - numpy
    - tensorflow.keras
    - backend.py (contains core ML logic: preprocess_data, build_model, train_or_retrain, generate_text)
"""

import streamlit as st
import os
import pickle
import numpy as np
# from tensorflow.keras.models import load_model

from backend import preprocess_data, build_model, train_or_retrain, generate_text


# -------------------------
# Streamlit Config
# -------------------------
st.set_page_config(page_title='The RNN Kitchen',
                   page_icon='🍔',
                   layout='wide', 
                   initial_sidebar_state='expanded')

# App tabs
tabs = st.tabs(['Preprocess Data','Train Model','Generate Text'])


# -------------------------
# Tab 1: Data Preprocessing
# -------------------------
with tabs[0]:
    st.markdown("""
    ## Tab 1: Data Preprocessing
    Allows users to upload a text file and preprocess it for RNN training.

    Steps:
        - Upload .txt dataset
        - Configure preprocessing parameters (vocab size, padding, lowercase, etc.)
        - Run preprocessing to generate tokenized input sequences
    """)
    
    
    
    with st.sidebar:
        st.info("Can't understand anything?\nHere's a small hint:")
        
        st.info("""
        **preprocess_data()**
        --------------------
        Args:
            filepath (str): Path to the text dataset file.
            num_words (int): The maximum number of words to keep.
            oov_token (str): Token for unknown words.
            lower (bool): Convert text to lowercase.
            padding (str): Sequence padding type ('pre' or 'post').
            fresh (bool): Whether to reset tokenizer.

        Returns:
            tuple: (X, y, vocab_size, max_len, tokenizer)
        """)
    
    col1,col2,col3 = st.columns(3,vertical_alignment='center',gap='large')
    with col1:
        data_file = st.file_uploader(label="""
Upload the .txt File Here.
But remember that larger files 
take longer to preprocess and train
                                """, type=["txt"])
        data = ''
        if data_file is not None:
            # Read as bytes then decode to string
            data: str = data_file.read().decode("utf-8")  
        max_words = st.number_input('Maximum Vocabulary',min_value=2000, max_value=50000)
    
    with col2:
        padding = st.selectbox('Padding Type',['pre','post'])
        lower = st.checkbox('Lowercase Data',True)
    
    with col3:
        fresh = st.checkbox('Fresh Tokenizer', False)
        oov_token = st.text_input('OOV Token',"<OOV>")
    
    if st.button('Preprocess'):
        with st.spinner('Preprocessing',show_time=True):
            try:
                X,y,total_words,max_len,tokenizer = preprocess_data(data=data,
                                                                    num_words=max_words,
                                                                    oov_token=oov_token,
                                                                    lower=lower,
                                                                    padding=padding,
                                                                    fresh=fresh,
                                                                    )
                st.success(f"✅ Preprocessing Done | Vocabulary Size: {total_words}, Max Length: {max_len}")
                st.session_state["data"] = (X, y, total_words, max_len, tokenizer)
            except ValueError:
                st.error('Please Input Values with Proper Data Types')
            except Exception as e:
                st.error(f"Error: {e}")


# -------------------------
# Tab 2: Build & Train Model
# -------------------------
with tabs[1]:
    st.info('Note that this may take some time depending on your data and model architecture')
    with st.sidebar:
        st.info("Can't understand anything?\nHere's a small hint:")
        st.info("""
        You can control callbacks here:

        **Callbacks:**
        - ModelCheckpoint: Save the best model based on validation accuracy/loss
        - EarlyStopping: Stop training when no improvement
        - ReduceLROnPlateau: Lower learning rate when stuck
        """)

    st.header("Step 2: Build & Train Model")

    if "data" not in st.session_state:
        st.warning("⚠️ Please preprocess data first.")
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
        st.warning("Please be patient and wait for it to train and monitor it in the manage app section and \n by all means don't press the button twice")
        if st.button("Train Model"):
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

# -------------------------
# Tab 3: Generate Text
# -------------------------
with tabs[2]:
    st.markdown("""
    ## Tab 3: Generate Text
    Uses the trained RNN model to generate text sequences.

    Function used:
        - generate_text(): takes a seed string and predicts the next N words.

    Args (via UI):
        seed_text (str): Starting phrase for generation.
        predict_next_words (int): Number of words to generate.
        temp (float): Sampling temperature.
        topK (int): Top-K sampling filter.
        topP (float): Top-P (nucleus) sampling filter.

    Returns:
        str: Generated text continuation.
    """)
    st.header("Step 3: Generate Text")

    if "model" not in st.session_state:
        st.warning("⚠️ Please train a model first.")
    else:
        seed_text = st.text_input("Seed Text", "Brandon")
        predict_words = st.number_input("Predict Next Words", 1, 50, 5)

        col1, col2, col3 = st.columns(3)
        with col1:
            temp = st.slider("Temperature", 0.1, 2.0, 1.0, 0.1)
        with col2:
            topK = st.number_input("Top-K", 0, 50, 0)
        with col3:
            topP = st.slider("Top-P (Nucleus)", 0.1, 1.0, 1.0, 0.05)

        if st.button("Generate"):
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
            st.success(text)
