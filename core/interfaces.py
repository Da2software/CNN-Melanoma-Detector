from abc import ABC, abstractmethod
from enum import Enum


class MelanomaType(Enum):
    BENIGN = 0
    MALIGNANT = 1


class ModelManager(ABC):
    @abstractmethod
    def predict_image(self, image) -> MelanomaType:
        """
        takes an image and process it to be use in the model to predict
        :param image: image object
        :return: MelanomaType (BENIGN, MALIGNANT)
        """
        pass

    @abstractmethod
    def model_evaluate(self) -> tuple:
        """
        takes the model evaluation information
        :return: tuple(loss, accuracy)
        """
        pass


class ImageToPreict(ABC):
    def __init__(self, image):
        """
        takes an image object saves it to be converted later
        :param image: image object
        """
        self.image = image

    @abstractmethod
    def convert(self, resize_shape=(224, 244)):
        """
        Converts the object to be used in the model
        :param: shape we need to resize
        :return: matrix or object used in the model for predictions
        """
        pass
