import os
import cv2

from files_preprocessor.FilesPreprocessor import FilesPreprocessor


class TFEIDFilesPreprocessor(FilesPreprocessor):
    def __init__(self, source_path, destination_path):
        super().__init__(source_path, destination_path)
        self.__image_format = ".jpg"

    def preprocess(self):
        # Dictionary to map folder names to emotions
        emotions = {
            "dfh_anger_x": "anger",
            "dfh_contempt_x": "contempt",
            "dfh_disguest_x": "disgust",
            "dfh_fear_x": "fear",
            "dfh_happiness_x": "happiness",
            "dfh_neutral_x": "neutral",
            "dfh_sadness_x": "sadness",
            "dfh_surprise_x": "surprise"
        }

        # Iterate through each emotion directory
        for emotion_dir in os.listdir(self._source_path):
            emotion_path = os.path.join(self._source_path, emotion_dir)
            # Skip if not a directory
            if not os.path.isdir(emotion_path):
                continue

            # Extract emotion from the directory name
            emotion = emotions.get(emotion_dir, None)
            if emotion is None:
                continue

            # Process each image in the directory
            for file in os.listdir(emotion_path):
                # Validate the file format
                if not file.lower().endswith(self.__image_format):
                    raise ValueError(f"Wrong image format found in directory: {file}")

                # Extract ID and gender from the file name
                # Assuming file format is "{SEX}{ID}_dfh_{...}.jpg"
                parts = file.split('_')
                if len(parts) < 2:
                    continue  # Skip if file name format is unexpected

                gender_id = parts[0]
                gender = gender_id[0]
                id = gender_id[1:]

                subject_id = ""
                if gender == "m":
                    subject_id = "1" + subject_id
                elif gender == "f":
                    subject_id = "2" + subject_id
                else:
                    raise ValueError(f"Unknown gender: {gender}")

                subject_id += id

                # Load and save the image with the new name format
                image = self.load_image(os.path.join(emotion_path, file))
                new_image_name = f"{subject_id}-{emotion}{self.__image_format}"
                cv2.imwrite(os.path.join(self._destination_path, new_image_name), image)
