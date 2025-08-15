# Next-Word-Prediction-Using-LSTM
# 📜 Shakespeare Next-K Word Prediction with LSTM

This project trains an **LSTM (Long Short-Term Memory)** neural network to predict the **next `K` words** in a sequence using the **Shakespeare dataset**.  
It also provides an interactive **FastAPI web interface** where users can type a phrase and get:

- **The next single word prediction**
- **The next `K` words prediction** (default: 3)

---
## 🧠 Dataset

We use the **Shakespeare text dataset**, cleaned and tokenized into word sequences.  
The dataset is split into training sequences of fixed length, and each sequence is fed to the LSTM to learn word dependencies.

---

## ⚙️ Model Architecture

- **Embedding Layer** — Converts words (indices) into dense vector embeddings.
- **LSTM Layer(s)** — Captures sequential dependencies in text.
- **Fully Connected Layer** — Maps LSTM outputs to vocabulary probabilities.
- **Softmax Activation** — Produces probability distribution over the next possible words.

**Loss Function:** CrossEntropyLoss  
**Optimizer:** Adam  
**Framework:** PyTorch

---

## 🌐 Web Interface (FastAPI)

The project includes a **FastAPI app** to serve predictions via a browser-based UI.

### Features:
- Type in a phrase and submit.
- View the **next word** and **next 3 words** predicted.
- Clean and responsive HTML/CSS design.

---

