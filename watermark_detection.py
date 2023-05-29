from imutils.object_detection import non_max_suppression
from typing import Dict, List
from easyocr import Reader
import pytesseract
import itertools
import numpy as np
import cv2


class WatermarkDetector:
    IMAGE_SIZE = 640
    CUSTOM_OEM_PSM_CONFIG = r'--oem 3 --psm 6'

    def __init__(self, file_name: str):
        self.image = cv2.imread(file_name, cv2.IMREAD_COLOR)
        self.original_image = self.image.copy()
        self.boxes = None

    @staticmethod
    def _image_padding(image: np.array) -> np.array:
        row, col, _ = image.shape
        _max = max(col, row)
        result = np.zeros((_max, _max, 3), np.uint8)
        result[0:row, 0:col] = image
        return result

    @staticmethod
    def _jaccard_similarity(string1: str, string2: str) -> float:
        intersection = len(list(set(string1).intersection(string2)))
        union = (len(set(string1)) + len(set(string2))) - intersection
        return float(intersection) / union

    def detect_text(self, min_confidence: float = 0.5) -> List[tuple]:
        image = self._image_padding(self.image)
        (H, W) = image.shape[:2]
        (newW, newH) = (self.IMAGE_SIZE, self.IMAGE_SIZE)
        rW = W / float(newW)
        rH = H / float(newH)
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

        (numRows, numCols) = scores.shape[2:4]
        boxes = []
        confidences = []
        for y in range(0, numRows):
            scoresData = scores[0, 0, y]
            xData0 = geometry[0, 0, y]
            xData1 = geometry[0, 1, y]
            xData2 = geometry[0, 2, y]
            xData3 = geometry[0, 3, y]
            anglesData = geometry[0, 4, y]

            for x in range(0, numCols):
                if scoresData[x] < min_confidence:
                    continue
                (offsetX, offsetY) = (x * 4.0, y * 4.0)

                angle = anglesData[x]
                cos = np.cos(angle)
                sin = np.sin(angle)

                h = xData0[x] + xData2[x]
                w = xData1[x] + xData3[x]
                endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
                endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
                startX = int(endX - w)
                startY = int(endY - h)

                boxes.append((startX, startY, endX, endY))
                confidences.append(scoresData[x])

        boxes = non_max_suppression(np.array(boxes), probs=confidences)
        self.boxes = list(np.multiply(boxes, [rW, rH, rW, rH]).astype(int))
        return self.boxes

    def find_watermark(self, watermark_list: List[str], similarity_thr: float = 0.5) -> np.array:
        # pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"
        reader = Reader(['en'])
        watermark_boxes = []
        for (startX, startY, endX, endY) in self.boxes:
            txt_image = self.original_image[startY:endY, startX:endX]

            try:
                # txt = pytesseract.image_to_string(txt_img, lang="eng", config=CUSTOM_OEM_PSM_CONFIG)
                extracted_txt = reader.readtext(txt_image, detail=0)[0]
            except:
                continue

            extracted_txt = extracted_txt.strip().lower()
            if max(list(map(self._jaccard_similarity, itertools.repeat(extracted_txt, len(watermark_list)),
                            watermark_list))) >= similarity_thr:
                cv2.rectangle(self.original_image, (startX, startY), (endX, endY), (0, 255, 0), 1)
        return self.original_image
