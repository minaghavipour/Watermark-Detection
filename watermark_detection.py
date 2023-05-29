import numpy as np
import cv2


class WatermarkDetector:
    IMAGE_SIZE = 640
    rW = 0
    rH = 0

    def __int__(self, file_name):
        self.image = cv2.imread(file_name, cv2.IMREAD_COLOR)
        self.orig = self.image.copy()

    @staticmethod
    def _image_padding(image):
        row, col, _ = image.shape
        _max = max(col, row)
        result = np.zeros((_max, _max, 3), np.uint8)
        result[0:row, 0:col] = image
        return result

    def text_detection(self):
        image = self._image_padding(self.image)
        (H, W) = image.shape[:2]
        (newW, newH) = (self.IMAGE_SIZE, self.IMAGE_SIZE)
        self.rW = W / float(newW)
        self.rH = H / float(newH)
        image = cv2.resize(image, (newW, newH))
        (H, W) = image.shape[:2]

        layer_names = [
            "feature_fusion/Conv_7/Sigmoid",
            "feature_fusion/concat_3"]
        net = cv2.dnn.readNet("frozen_east_text_detection.pb")
        blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
                                     (123.68, 116.78, 103.94), swapRB=True, crop=False)
        net.setInput(blob)
        (scores, geometry) = net.forward(layer_names)
        return scores, geometry




