import streamlit as st
import os
import pickle
import numpy as np
from tensorflow.keras.models import load_model

from backend import preprocess_data, build_model, train_or_retrain, generate_text

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="RNN Text Generator", layout="wide")
st.title("📖 RNN Text Generator")

# Tabs
tabs = st.tabs(["Preprocess Data", "Build & Train Model", "Generate Text"])

# -------------------------
# Tab 1: Preprocess Data
# -------------------------
with tabs[0]:
    st.header("Step 1: Preprocess Data")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        dataset_path = st.text_input("Dataset Path", "data.txt")
        num_words = st.number_input("Max Words (Vocabulary)", 1000, 50000, 10000)
    with col2:
        padding = st.selectbox("Padding Type", ["pre", "post"])
        lower = st.checkbox("Lowercase Data", True)
    with col3:
        fresh = st.checkbox("Fresh Tokenizer", False)
        oov_token = st.text_input("OOV Token", "<OOV>")

    if st.button("Preprocess"):
        try:
            X, y, total_words, max_len, tokenizer = preprocess_data(
                filepath=dataset_path,
                num_words=num_words,
                oov_token=oov_token,
                lower=lower,
                padding=padding,
                fresh=fresh,
            )
            st.success(f"✅ Preprocessing Done | Vocabulary Size: {total_words}, Max Length: {max_len}")
            st.session_state["data"] = (X, y, total_words, max_len, tokenizer)
        except Exception as e:
            st.error(f"Error: {e}")

# -------------------------
# Tab 2: Build & Train Model
# -------------------------
with tabs[1]:
    st.header("Step 2: Build & Train Model")

    if "data" not in st.session_state:
        st.warning("⚠️ Please preprocess data first.")
    else:
        X, y, total_words, max_len, tokenizer = st.session_state["data"]

        col1, col2, col3 = st.columns(3)
        with col1:
            rnn_type = st.selectbox("RNN Type", ["GRU", "LSTM", "SimpleRNN"])
            rnn_units = st.number_input("RNN Units", 50, 512, 180)
            embedding_units = st.number_input("Embedding Units", 50, 300, 100)
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
            batch_size = st.number_input("Batch Size", 32, 4096, 128)
        with col5:
            validation_split = st.slider("Validation Split", 0.1, 0.5, 0.2)
            fresh_model = st.checkbox("Fresh Model", False)

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
            )

            st.success(f"✅ Training Complete | Val Loss: {loss:.4f}, Val Accuracy: {acc:.4f}")
            st.session_state["model"] = trained_model
            st.session_state["max_len"] = max_len
            st.session_state["tokenizer"] = tokenizer

# -------------------------
# Tab 3: Generate Text
# -------------------------
with tabs[2]:
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
