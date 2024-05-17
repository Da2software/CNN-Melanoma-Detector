from .interfaces import *
import tensorflow as tf
import numpy as np
from pathlib import Path


class TensorflowModel(ModelManager):
    def __init__(self, path: Path | str):
        self.model = tf.keras.models.load_model(path)
        test_dataset_path = 'melanoma_cancer_dataset/test'
        self.test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
            test_dataset_path, image_size=(224, 224), batch_size=32,
            label_mode='int', )

    def predict_image(self, image) -> MelanomaType:
        predictions = self.model.predict(image)
        print(predictions)
        return MelanomaType(np.argmax(predictions, axis=1)[0])

    def model_evaluate(self) -> tuple:
        return self.model.evaluate(self.test_dataset, verbose=2)


class ImageTransform(ImageToPreict):
    def __init__(self, image: bytes):
        super().__init__(image)

    def convert(self, resize_shape=(224, 224)):
        img = tf.image.decode_image(self.image, channels=3)
        img = tf.image.resize(images=img, size=resize_shape)
        img = tf.expand_dims(img, axis=0)
        img = tf.cast(img, tf.float32)
        return img


class ImageTransformLocal(ImageToPreict):
    def __init__(self, image: str):
        super().__init__(image)

    def convert(self, resize_shape=(224, 224)):
        img = tf.keras.utils.load_img(self.image, color_mode='rgb',
                                      target_size=resize_shape, )
        img_array = tf.keras.utils.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        # img_array /= 255.0  # Normalize the image if required by your model
        return img_array
