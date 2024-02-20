import os

import cv2
import joblib

if __name__ == '__main__':
    dataset_path = "../../dataset_tfeid"
    dataset_name = "tfeid"
    image_format = ".jpg"

    face_boundaries_dict = {}
    for file in os.listdir(dataset_path):
        if not file.endswith(image_format):
            raise ValueError("Wrong image format found in directory!")

        img_path = os.path.join(dataset_path, file)
        img_name = file[:-4]

        image = cv2.imread(img_path)
        height, width = image.shape[:2]

        face_boundaries_dict[img_name] = [0, 0, width-1, height-1]

    joblib.dump(face_boundaries_dict, f"../../face_boundaries_dict_{dataset_name}.pkl")

    print(face_boundaries_dict)
