import os
import cv2


class FilesPreprocessor:
    def __init__(self, source_path, destination_path):
        self._source_path = source_path
        self._destination_path = destination_path

    def load_image(self, path):
        return cv2.imread(path)

    def save_image(self, image, name):
        cv2.imwrite(os.path.join(self._destination_path, name), image)

    def validate_file(self, file, valid_extensions):
        if not any(file.endswith(ext) for ext in valid_extensions):
            raise ValueError("Invalid file format")

    def preprocess(self):
        raise NotImplementedError("Subclasses must implement the preprocess method")
