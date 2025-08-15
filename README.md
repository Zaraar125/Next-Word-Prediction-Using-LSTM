# Next-Word-Prediction-Using-LSTM
# 📜 Shakespeare Next-K Word Prediction with LSTM

This project trains an **LSTM (Long Short-Term Memory)** neural network to predict the **next `K` words** in a sequence using the **Shakespeare dataset**.  
It also provides an interactive **FastAPI web interface** where users can type a phrase and get:

- **The next single word prediction**
- **The next `K` words prediction** (default: 3)

---

## 📂 Project Structure

.
├── website.py # FastAPI backend
├── lstm_model.pth # Trained LSTM model (PyTorch)
├── vector_dict.json # Word → index mapping
├── inverse_dict.json # Index → word mapping
├── requirements.txt # Python dependencies
├── templates/
│ └── index.html # Web interface template
└── README.md # This file

yaml
Copy
Edit

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

## 📦 Installation & Setup

### 1️⃣ Clone Repository
```bash
git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name
2️⃣ Create Virtual Environment (Optional but Recommended)
bash
Copy
Edit
python -m venv .venv
source .venv/bin/activate      # On macOS/Linux
.venv\Scripts\activate         # On Windows
3️⃣ Install Dependencies
bash
Copy
Edit
pip install -r requirements.txt
requirements.txt should contain:

nginx
Copy
Edit
fastapi
uvicorn
torch
jinja2
🚀 Running the Web App
From your project directory:

bash
Copy
Edit
uvicorn website:app --reload
Open your browser at:

cpp
Copy
Edit
http://127.0.0.1:8000
🔮 How Predictions Work
Text Input is split into words.

Each word is converted into its index using vector_dict.json.

The last N words (e.g., last 5 tokens) are fed into the LSTM.

The model outputs a probability distribution for the next word.

The top predicted index is converted back to a word using inverse_dict.json.

For multi-word prediction, the predicted word is appended to the sequence and the process repeats.

🛠 Error Handling
If a word is not found in vector_dict.json, it is replaced with an <UNK> token index (default: 0).

The system gracefully handles unknown inputs without crashing.

📜 Example
Input:

css
Copy
Edit
to be or not
Output:

vbnet
Copy
Edit
Next Word: to
Next 3 Words: to be or
📈 Training Notes
Sequence length: 5

Vocabulary built from Shakespeare dataset

Trained using Adam optimizer

Saved using:

python
Copy
Edit
torch.save(model, "lstm_model.pth")
🧑‍💻 Author
Your Name
📧 your.email@example.com
🔗 GitHub Profile

📄 License
This project is licensed under the MIT License.

yaml
Copy
Edit

---

You can literally **copy-paste** this into your `README.md` in GitHub and it’s ready.  

Do you want me to also add a **"How to Retrain the Model"** section so people can fine-tune it on new datasets? That could ma