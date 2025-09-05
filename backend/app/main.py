from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from app.utils.predict import predict_image
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Allow requests from React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    label = predict_image(contents)
    return JSONResponse(content={"prediction": label})
