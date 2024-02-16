import os
import cv2

from files_preprocessor.FilesPreprocessor import FilesPreprocessor


class CKPlusFilesPreprocessor(FilesPreprocessor):
    def __init__(self, source_path, destination_path):
        super().__init__(source_path, destination_path)
        self.__image_format = ".png"
        self.__max_for_subject = {}

    def preprocess_files(self, emotion):
        emotion_path = os.path.join(self._source_path, emotion)
        if emotion == "neutral":
            for file in os.listdir(emotion_path):
                if not file.endswith(self.__image_format):
                    raise ValueError("Wrong image format found in directory!")

                img_name = file[:-4]
                subject_id, emotion_th, _ = img_name.split("_")
                subject_id = int(subject_id[1:])
                if subject_id not in self.__max_for_subject:
                    self.__max_for_subject[subject_id] = int(emotion_th)
                else:
                    if int(emotion_th) > self.__max_for_subject[subject_id]:
                        self.__max_for_subject[subject_id] = int(emotion_th)

        for file in os.listdir(emotion_path):
            if not file.endswith(self.__image_format):
                raise ValueError("Wrong image format found in directory!")

            img_path = os.path.join(emotion_path, file)
            img = cv2.imread(img_path)

            img_name = file[:-4]
            subject_id, emotion_th, _ = img_name.split("_")
            subject_id = int(subject_id[1:])
            if emotion == "neutral":
                if int(emotion_th) != self.__max_for_subject[subject_id]:
                    continue

            landmark_key = f"{subject_id}-{emotion}"
            cv2.imwrite(self._destination_path + "//" + landmark_key + self.__image_format, img)

    def preprocess(self):
        emotions = ["happiness", "anger", "surprise", "fear", "disgust", "sadness", "contempt", "neutral"]
        for emotion in emotions:
            self.preprocess_files(emotion)
