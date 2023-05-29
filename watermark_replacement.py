import cv2
import time
from watermark_detection import WatermarkDetector

MIN_CONFIDENCE = 0.5
SIMILARITY_THRESHOLD = 0.5

if __name__ == '__main__':
    image_file = r"Images/1249571.jpg"
    watermark_list = ["kala", "digikala", "copyright", "photo by", "digikala.com", "com"]
    replacement_list = ["New_Watermarks/technolife_logo_small.png", "New_Watermarks/technolife_logo_small.png", "", "",
                        "", ""]

    detector = WatermarkDetector(image_file)
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
