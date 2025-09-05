from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Lambda
from tensorflow_hub import KerasLayer
from PIL import Image
from io import BytesIO
import numpy as np

# -------------------------
# Step 1: Define the custom Lambda function
# -------------------------
def extract_features(x):
    # This replicates what you did in the notebook
    hub_layer = KerasLayer(
        "https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/5",
        trainable=False
    )
    return hub_layer(x)

# -------------------------
# Step 2: Load the model with custom_objects
# -------------------------
model = load_model(
    "app/models/dog_cat_model.keras",
    custom_objects={"extract_features": extract_features},
    safe_mode=False
)

# -------------------------
# Step 3: Prediction function
# -------------------------
def predict_image(file_bytes):
    # Open and preprocess image
    img = Image.open(BytesIO(file_bytes)).convert("RGB")
    img = img.resize((224,224))
    x = np.array(img) / 255.0
    x = np.expand_dims(x, axis=0)

    # Make prediction
    pred = model.predict(x)
    return "Dog" if pred[0][0] > 0.5 else "Cat"
