import time
from watermark_detection import WatermarkDetector

if __name__ == '__main__':
    watermark_file = "./technolife_logo_small.png"
    image_file = r"C:\Users\Software\Desktop\MyData\Images\114633982.jpg"

    detector = WatermarkDetector(image_file)
    print("[INFO] loading text detection model ...")
    start = time.time()
    scores, geometry = detector.text_detection()
    end = time.time()
    print("[INFO] text detection took {:.6f} seconds".format(end - start))