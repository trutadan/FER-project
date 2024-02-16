import os
import cv2

from files_preprocessor.FilesPreprocessor import FilesPreprocessor


class KDEFFilesPreprocessor(FilesPreprocessor):
    def __init__(self, source_path, destination_path):
        super().__init__(source_path, destination_path)
        self.__image_format = ".jpg"

    def preprocess(self):
        for folder in os.listdir(self._source_path):
            for file in os.listdir(os.path.join(self._source_path, folder)):
                if not file.endswith(self.__image_format):
                    raise ValueError("Wrong image format found in directory!")
                image = cv2.imread(os.path.join(self._source_path, folder, file))

                image_name = file[:-4]
                if not image_name.endswith("S"):
                    continue

                session = image_name[0]
                gender = image_name[1]
                subject_id = image_name[2:4]
                emotion = image_name[4:6]
                emotions = {
                            "AF": "afraid",
                            "AN": "angry",
                            "DI": "disgusted",
                            "HA": "happy",
                            "NE": "neutral",
                            "SA": "sad",
                            "SU": "surprised"
                            }

                if gender == "M":
                    subject_id = "1" + subject_id
                elif gender == "F":
                    subject_id = "2" + subject_id
                else:
                    raise ValueError(f"Unknown gender: {gender}")

                if session == "A":
                    subject_id = subject_id + "1"
                elif session == "B":
                    subject_id = subject_id + "2"
                else:
                    raise ValueError(f"Unknown session: {session}")

                emotion = emotions[emotion]
                image_name = subject_id + "-" + emotion

                cv2.imwrite(os.path.join(self._destination_path, image_name + self.__image_format), image)
