from .interfaces import *
import tensorflow as tf
import numpy as np
from pathlib import Path


class TensorflowModel(ModelManager):
    def __init__(self, path: Path | str):
        self.model = tf.keras.models.load_model(path)

    def predict_image(self, image) -> MelanomaType:
        predictions = self.model.predict(image)
        print(np.argmax(predictions, axis=1)[0])
        return MelanomaType(np.argmax(predictions, axis=1)[0])

    def model_evaluate(self) -> tuple:
        test_dataset_path = 'melanoma_cancer_dataset/test'
        test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
            test_dataset_path, image_size=(224, 224), batch_size=32,
            label_mode='int', )
        return self.model.evaluate(test_dataset, verbose=2)


class ImageTransform(ImageToPreict):
    def __init__(self, image: bytes):
        super().__init__(image)

    def convert(self, resize_shape=(224, 224)):
        img = tf.image.decode_image(self.image, channels=3)
        img = tf.image.resize(images=img, size=resize_shape)
        img = tf.cast(img, tf.float32) / 255  # convert color values to be 0-1
        img = tf.expand_dims(img, axis=0)
        return img
