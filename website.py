from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import torch
import json
import pickle
# ====== CONFIG ======
# MODEL_PATH = "lstm_model.pth"  # If you used torch.save(model)
DICT_PATH = "vector_dict.json"
INV_DICT_PATH = "inverse_dict.json"

# ====== FASTAPI INIT ======
app = FastAPI()
templates = Jinja2Templates(directory="templates")

# ====== LOAD MODEL ======
# If you saved with torch.save(model, path)
# model = torch.load(MODEL_PATH, map_location="cpu")
with open('lstm_model.pkl', 'rb') as f:
    model = pickle.load(f)
    model.to('cpu')

model.eval()

# Load dictionaries
DICT = json.load(open(DICT_PATH))
INVERSE_DICT = json.load(open(INV_DICT_PATH))

# ====== PREDICTION FUNCTION ======
def predict_next_words(model, list_of_words):
    gg = [DICT[str(i)] for i in list_of_words]  # Ensure keys are strings
    while len(gg) < 5:
        gg.append(0)
    gg = gg[-5:]

    gg_tensor = torch.tensor(gg).unsqueeze(0)
    output = model(gg_tensor).squeeze(0)

    # Next word
    answer_idx = torch.argmax(output).item()
    answer = INVERSE_DICT[str(answer_idx)]

    # Next few words
    hold = list_of_words + [answer]
    for _ in range(3):
        gg = [DICT[str(i)] for i in hold]
        while len(gg) < 5:
            gg.append(0)
        gg = gg[-5:]
        gg_tensor = torch.tensor(gg).unsqueeze(0)
        output = model(gg_tensor).squeeze(0)
        next_idx = torch.argmax(output).item()
        hold.append(INVERSE_DICT[str(next_idx)])

    return [answer], hold[-3:]

# ====== ROUTES ======
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "prediction_list": [],
        "next_predictions_list": []
    })

@app.post("/predict/", response_class=HTMLResponse)
async def predict(request: Request, input_text: str = Form(...)):
    prediction, next_predictions = predict_next_words(model, input_text.split())
    return templates.TemplateResponse("index.html", {
        "request": request,
        "prediction_list": prediction,
        "next_predictions_list": next_predictions
    })
