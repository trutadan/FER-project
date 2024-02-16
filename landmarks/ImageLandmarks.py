class ImageLandmarks:
    def __init__(self):
        self.landmarks = {}

    def add_landmarks(self, image_name, landmarks):
        """
        :param image_name: key
        :param landmarks: value
        :return:
        """
        self.landmarks[image_name] = landmarks
