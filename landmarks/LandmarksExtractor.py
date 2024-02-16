import os
import joblib
import cv2
import numpy as np

from landmarks.ImageLandmarks import ImageLandmarks
from spiga.inference.config import ModelConfig
from spiga.inference.framework import SPIGAFramework


class LandmarksExtractor:
    def __init__(self, dataset_path, dataset_name, image_type):
        self.__dataset_path = dataset_path
        self.__dataset_name = dataset_name
        self.__image_type = image_type
        self.__processor = SPIGAFramework(ModelConfig("wflw"))
        self.__landmarks_dictionary = ImageLandmarks()

    def get_landmarks_dictionary(self):
        return self.__landmarks_dictionary

    def __get_landmarks(self, img_path, face_boundaries):
        # we need to change the structure of the bounding box to be compatible with the SPIGA framework
        # instead of[x1, y1, x2, y2] we need [x1, y1, w, h]

        img = cv2.imread(img_path)

        x1, y1, x2, y2 = face_boundaries
        bounding_box = [x1, y1, x2 - x1, y2 - y1]

        features = self.__processor.inference(img, [bounding_box])
        print(features)
        landmarks = features['landmarks'][0]

        return landmarks

    def __populate_landmarks_dictionary_emotion(self, emotion: str, face_boundaries_dict, id: int):
        emotion_for_id_path = os.path.join(self.__dataset_path, f"{id}-{emotion}.{self.__image_type}")
        if emotion == "neutral":
            if not os.path.exists(emotion_for_id_path):
                raise ValueError("Neutral image does not exist!")
        else:
            if not os.path.exists(emotion_for_id_path):
                return None

        emotion_boundaries = face_boundaries_dict[f"{id}-{emotion}"]

        emotion_landmarks = self.__get_landmarks(emotion_for_id_path, emotion_boundaries)

        self.__landmarks_dictionary.landmarks[f"{id}-{emotion}"] = emotion_landmarks
        print(f"Done for subject {id} and emotion {emotion}...")

    def __align_landmarks(self, emotion: str, subject_id: int):
        if emotion == "neutral":
            return None

        neutral_landmarks = self.__landmarks_dictionary.landmarks[f"{subject_id}-neutral"]
        # check if key exists in dictionary
        if f"{subject_id}-{emotion}" not in self.__landmarks_dictionary.landmarks:
            return None
        to_align_landmarks = self.__landmarks_dictionary.landmarks[f"{subject_id}-{emotion}"]

        nasal_root = neutral_landmarks[51]
        to_align_nose = to_align_landmarks[51]
        to_align_nose_difference = np.array(nasal_root) - np.array(to_align_nose)
        aligned_landmarks = (np.array(to_align_landmarks) + to_align_nose_difference).tolist()
        self.__landmarks_dictionary.landmarks[f"{subject_id}-{emotion}"] = aligned_landmarks

    def populate_landmarks_dictionary(self, emotions: list):
        id_list = []
        for filename in os.listdir(self.__dataset_path):
            id, emotion = filename[:-4].split("-")
            id_list.append(id) if id not in id_list else None

        print(id_list)
        face_boundaries_dict = joblib.load(f"../face_boundaries_dict_{self.__dataset_name}.pkl")
        for id in id_list:

            for emotion in emotions:
                self.__populate_landmarks_dictionary_emotion(emotion, face_boundaries_dict, id)

            neutral_landmarks = self.__landmarks_dictionary.landmarks[f"{id}-neutral"]

            for emotion in emotions:
                self.__align_landmarks(emotion, id)

            print(f"Done for subject {id}")

    def read(self):
        self.__landmarks_dictionary = joblib.load(f"../landmarks_{self.__dataset_name}.joblib")

    def save(self):
        joblib.dump(self.__landmarks_dictionary, f"../landmarks_{self.__dataset_name}.joblib")

