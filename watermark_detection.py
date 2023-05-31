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

    def __init__(self, image: np.array):
        self.image = image
        self.original_image = self.image.copy()
        self.txt_boxes = None
        self.watermark_boxes = None

    @staticmethod
    def _image_padding(image: np.array) -> np.array:
        row, col, _ = image.shape
        _max = max(col, row)
        result = np.zeros((_max, _max, 3), np.uint8)
        result[0:row, 0:col] = image
        return result

    @staticmethod
    def _jaccard_similarity(extracted_txt: str, watermark: str) -> tuple:
        key_list = watermark.split(' ')
        similarities = []
        for i in range(1, len(key_list) + 1):
            for j in range(len(key_list) - i + 1):
                key_phrase = ' '.join(key_list[j:j + i])
                intersection = len(list(set(extracted_txt).intersection(key_phrase)))
                union = (len(set(extracted_txt)) + len(set(key_phrase))) - intersection
                similarities.append((key_phrase, float(intersection) / union))
        sim_key, max_sim = similarities[np.argmax(list(zip(*similarities))[1])]
        return sim_key, max_sim

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
        self.txt_boxes = list(np.multiply(boxes, [rW, rH, rW, rH]).astype(int))
        return self.txt_boxes

    def find_watermark(self, watermark_list: List[str], similarity_thr: float = 0.5) -> List[tuple]:
        # pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"
        reader = Reader(['en'])
        watermark_boxes = []
        for (startX, startY, endX, endY) in self.txt_boxes:
            txt_image = self.original_image[startY:endY, startX:endX]

            try:
                # cv2.imshow("Text Detection", txt_image)
                # cv2.waitKey(0)
                # # txt = pytesseract.image_to_string(txt_img, lang="eng", config=CUSTOM_OEM_PSM_CONFIG)
                extracted_txt = reader.readtext(txt_image, detail=0)
                if not extracted_txt:
                    thr_image = cv2.threshold(src=cv2.cvtColor(txt_image, cv2.COLOR_RGB2GRAY), thresh=0, maxval=255,
                                              type=cv2.THRESH_OTSU + cv2.THRESH_BINARY)[1]
                    extracted_txt = reader.readtext(thr_image, detail=0)
                extracted_txt = extracted_txt[0]
            except:
                continue

            extracted_txt = extracted_txt.strip().lower()
            for watermark_index in range(len(watermark_list)):
                key, similarity = self._jaccard_similarity(extracted_txt, watermark_list[watermark_index])
                if similarity >= similarity_thr and (startY < self.IMAGE_SIZE * 0.5 or watermark_index):
                    key_index = watermark_list[watermark_index].find(key)
                    remaining_len = len(watermark_list[watermark_index]) - key_index - len(extracted_txt)
                    watermark_boxes.append(((startX, startY, endX, endY), watermark_index, remaining_len))
                    # cv2.rectangle(self.original_image, (startX, startY), (endX, endY), (0, 255, 0), 1)
                    break
        self.watermark_boxes = watermark_boxes
        return self.watermark_boxes

    def replace_watermark(self, replacement_list: List[tuple]) -> np.array:
        new_watermarks = {}
        for box, watermark_index, remaining_len in self.watermark_boxes:
            startX, startY, endX, endY = box
            if watermark_index > 0:
                replacement_list[watermark_index][:] = self.original_image[max(0, startY - 1), max(0, startX - 1)]
            endX, endY = max(endX + remaining_len * 5 + 3, replacement_list[watermark_index].shape[1]), max(endY,
                                                                     replacement_list[watermark_index].shape[0]) + 3
            H, W = max(endY - startY, replacement_list[watermark_index].shape[0]), max(endX - startX,
                                                                     replacement_list[watermark_index].shape[1])
            if watermark_index not in new_watermarks:
                new_watermarks[watermark_index] = []
            new_watermarks[watermark_index].append((W, H, endX, endY))
            # cv2.rectangle(self.original_image, (endX - W, endY - H), (endX, endY), (0, 255, 0), 1)
            # cv2.imshow("Text Detection", self.original_image)
            # cv2.waitKey(0)

        for watermark_index in new_watermarks.keys():
            box = np.max(new_watermarks[watermark_index], axis=0)
            W, H, endX, endY = box
            self.original_image[endY - H:endY, endX - W:endX] = cv2.resize(replacement_list[watermark_index], (W, H))
        return self.original_image
