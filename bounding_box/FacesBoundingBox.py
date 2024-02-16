import os
import joblib

from retinaface import RetinaFace


class FacesBoundingBox:
    def __init__(self, dataset_path, dataset_name, image_format):
        self.__dataset_path = dataset_path
        self.__dataset_name = dataset_name
        self.__image_format = image_format
        self.__face_boundaries_dict = {}

    def process(self):
        for file in os.listdir(self.__dataset_path):
            if not file.endswith(self.__image_format):
                raise ValueError("Wrong image format found in directory!")

            img_path = os.path.join(self.__dataset_path, file)
            img_name = file[:-4]

            # img = cv2.imread(img_path)

            coordinates = RetinaFace.detect_faces(img_path=img_path)
            if len(coordinates) != 1:
                raise ValueError("More than one face detected!")

            face_info = coordinates["face_1"]
            self.__face_boundaries_dict[img_name] = face_info["facial_area"]

            # print("detected face: " + str(img_name))

    def save(self):
        joblib.dump(self.__face_boundaries_dict, f"../../face_boundaries_dict_{self.__dataset_name}.pkl")

    def __str__(self):
        return str(self.__face_boundaries_dict)
