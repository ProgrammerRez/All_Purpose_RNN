# <a href='https://rnn-kitchen-project.streamlit.app' target="_blank">🍔 The RNN Kitchen </a>

**The RNN Kitchen** is an interactive **Streamlit app** for experimenting with Recurrent Neural Networks (RNNs) on text datasets.  
It lets you preprocess data, build/train models, and generate text — all from a clean browser-based UI.  

---

## 🚀 Features  

1. **Preprocess Data**  
   - Upload any `.txt` dataset.  
   - Configure preprocessing options (vocab size, padding type, lowercase, OOV tokens, etc.).  
   - Tokenizes and prepares sequences for model training.  

2. **Build & Train Models**  
   - Choose RNN type: `LSTM`, `GRU`, or `SimpleRNN`.  
   - Adjust architecture (embedding units, hidden layers, dropout).  
   - Pick optimizers and learning rate.  
   - Use callbacks like:  
     - **ModelCheckpoint** – Save the best model.  
     - **EarlyStopping** – Stop training when validation stops improving.  
     - **ReduceLROnPlateau** – Adjust learning rate dynamically.  
   - View **real-time logs** of training inside the app.  

3. **Generate Text**  
   - Provide a **seed phrase**.  
   - Configure **temperature**, **Top-K**, or **Top-P** sampling.  
   - Generate realistic text based on your trained RNN.  

---

## 🛠️ Installation  

Clone the repo and install dependencies:  

```bash
git clone https://github.com/your-username/rnn-kitchen.git
cd rnn-kitchen

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows

# Install requirements
pip install -r requirements.txt
  ````

---

## ▶️ Usage

Run the app with:

```bash
streamlit run app.py
```

This will start the server, and you can access the app at:

```
http://localhost:8501
```

---

## 📂 Project Structure

```
rnn-kitchen/
│── app.py          # Streamlit frontend
│── backend.py      # Core backend (data prep, training, text generation)
│── requirements.txt
│── models/         # Saved trained models
│── data/           # Uploaded datasets
```

---

## ⚙️ Configuration Parameters

### 🔹 Preprocessing

* **Vocab Size** – Limit size of vocabulary.
* **OOV Token** – Token for unseen words.
* **Padding Type** – `pre` or `post`.
* **Lowercase** – Convert text to lowercase.

### 🔹 Model Building

* **RNN Type** – `LSTM`, `GRU`, or `SimpleRNN`.
* **Embedding Units** – Size of embedding layer.
* **Hidden Layers** – Number and size of recurrent layers.
* **Dropout** – Prevents overfitting.
* **Optimizer** – `Adam`, `RMSprop`, etc.
* **Learning Rate** – Customizable.

### 🔹 Callbacks

* `ModelCheckpoint` – Save the best model automatically.
* `EarlyStopping` – Stop if validation loss doesn’t improve.
* `ReduceLROnPlateau` – Reduce learning rate on stagnation.

### 🔹 Text Generation

* **Seed Text** – Starting phrase for generation.
* **Temperature** – Controls creativity (higher = more random).
* **Top-K Sampling** – Keep only top-K words at each step.
* **Top-P Sampling** – Nucleus sampling with cumulative probability.

---

## 📊 Example: Generate Text

Once your model is trained, head to **Tab 3** (Generate Text) and try:

```text
Seed: "Once upon a time"
Temperature: 0.8
Top-K: 50
Output: "Once upon a time the king sat quietly in the garden, watching the sun fall behind the hills..."
```

---

## 🧑‍💻 Contributing

Pull requests are welcome!
If you’d like to add new features (e.g., Transformer support), please fork and submit a PR.

---

## 📜 License

MIT License.
Feel free to use, modify, and share!


