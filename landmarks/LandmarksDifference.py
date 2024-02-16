import joblib
import numpy as np

from .ImageLandmarks import ImageLandmarks


class LandmarksDifference:
    def __init__(self, image_landmarks: ImageLandmarks):
        self.__image_landmarks = image_landmarks
        self.__differences = {}

    def get_differences(self):
        return self.__differences

    def process(self):
        for image_landmarks_name, landmarks in self.__image_landmarks.landmarks.items():
            subject_id, emotion = image_landmarks_name.split("-")

            if emotion == "neutral":
                continue
            else:
                # "neutral", "anger", "contempt", "disgust", "fear", "happiness", "sadness", "surprise"
                neutral_landmarks = self.__image_landmarks.landmarks[f"{subject_id}-neutral"]

                # neutral_nose = neutral_landmarks[51]
                # self.__image_landmarks.landmarks[image_landmarks_name] = np.array(neutral_landmarks) - np.array(landmarks)
                self.__differences[image_landmarks_name] = np.array(neutral_landmarks) - np.array(landmarks)

    def save(self):
        joblib.dump(self.__differences, "../differences.joblib")
