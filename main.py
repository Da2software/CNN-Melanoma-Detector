from core.processes import TensorflowModel, ImageTransform, ImageTransformLocal
from typing import Annotated
from fastapi import FastAPI, File

app = FastAPI()

MODEL = TensorflowModel('tf_model.keras')


@app.get("/accuracy")
def get_accuracy():
    loss, acc = MODEL.model_evaluate()
    return {"loss": loss, "accuracy": acc}


@app.post("/predict")
def request_prediction(image: Annotated[bytes, File()]):
    image_transform = ImageTransform(image)
    img_tf = image_transform.convert()
    predict = MODEL.predict_image(img_tf)
    return {"prediction": predict.name}
