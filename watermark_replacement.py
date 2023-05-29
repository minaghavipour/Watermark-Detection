import cv2
import time
from watermark_detection import WatermarkDetector


MIN_CONFIDENCE = 0.5
SIMILARITY_THRESHOLD = 0.5


if __name__ == '__main__':
    image_file = r"C:\Users\Software\Desktop\MyData\Images\1662393.jpg"
    watermark_list = ["kala", "digikala", "photo by digikala.com"]

    detector = WatermarkDetector(image_file)
    print("[INFO] loading text detection model ...")
    start = time.time()
    boxes = detector.detect_text(MIN_CONFIDENCE)
    end = time.time()
    print("[INFO] text detection took {:.6f} seconds".format(end - start))

    print("[INFO] loading OCR model ...")
    start = time.time()
    image = detector.find_watermark(watermark_list, SIMILARITY_THRESHOLD)
    end = time.time()
    print("[INFO] OCR took {:.6f} seconds".format(end - start))

    cv2.imshow("Text Detection", image)
    cv2.waitKey(0)

