import cv2
import os
import joblib


class Visualizer:
    def __init__(self, dataset_name, dataset_path, image_format):
        self.__dataset_name = dataset_name
        self.__dataset_path = dataset_path
        self.__image_format = image_format
        self.__landmarks_dictionary = joblib.load(f"../landmarks_{self.__dataset_name}.joblib")

    def compare(self, pose1, pose2):
        img = cv2.imread(os.path.join(self.__dataset_path, f"{pose1}.{self.__image_format}"))

        landmarks_pose1 = self.__landmarks_dictionary.landmarks[pose1]
        # draw landmarks
        for index, (x, y) in enumerate(landmarks_pose1):
            # we take the top of the nose and right corner of the left eye as positional references
            if index == 51:
                cv2.circle(img, (int(x), int(y)), 2, (255, 0, 0), -1)
            else:
                cv2.circle(img, (int(x), int(y)), 2, (0, 0, 255), -1)

        landmarks_pose2 = self.__landmarks_dictionary.landmarks[pose2]
        # draw pose2 landmarks
        for index, (x, y) in enumerate(landmarks_pose2):
            # we take the top of the nose and right corner of the left eye as positional references
            if index == 51:
                cv2.circle(img, (int(x), int(y)), 2, (255, 0, 255), -1)
            else:
                cv2.circle(img, (int(x), int(y)), 2, (0, 255, 0), -1)

        # draw lines between pose1 and pose2 landmarks
        for index, (x, y) in enumerate(landmarks_pose1):
            if index == 51:
                continue
            else:
                cv2.line(img, (int(x), int(y)), (int(landmarks_pose2[index][0]),
                                                 int(landmarks_pose2[index][1])), (255, 0, 0), 1)

        print("Showing image")
        cv2.imshow("image", img)
        cv2.waitKey(0)
