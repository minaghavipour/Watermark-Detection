import cv2
import time
import numpy as np
from watermark_detection import WatermarkDetector

MIN_CONFIDENCE = 0.5
SIMILARITY_THRESHOLD = 0.5

if __name__ == '__main__':
    image_file = r"Images/2815064.jpg"
    image = cv2.imread(image_file, cv2.IMREAD_COLOR)
    watermark_file = r"New_Watermarks/technolife_logo_small.png"
    watermark = cv2.imread(watermark_file, cv2.IMREAD_COLOR)
    cover = np.zeros((15, 180, 3))

    watermark_list = ["digi kala", "copyright photo by digikala com"]
    # replacement_list = [(watermark, 0), (watermark, 0), (cover, 150), (cover, 100), (cover, 20), (cover, 0)]
    replacement_list = [watermark, cover]

    detector = WatermarkDetector(image)
    print("[INFO] loading text detection model ...")
    start = time.time()
    boxes = detector.detect_text(MIN_CONFIDENCE)
    end = time.time()
    print("[INFO] text detection took {:.6f} seconds".format(end - start))

    print("[INFO] loading OCR model ...")
    start = time.time()
    watermark_boxes = detector.find_watermark(watermark_list, SIMILARITY_THRESHOLD)
    end = time.time()
    print("[INFO] OCR took {:.6f} seconds".format(end - start))

    # cv2.imshow("Text Detection", detector.original_image)
    # cv2.waitKey(0)

    image = detector.replace_watermark(replacement_list)
    cv2.imshow("Text Detection", image)
    cv2.waitKey(0)
