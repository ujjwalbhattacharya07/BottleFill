import cv2
from matplotlib import pyplot as plt
import imutils
import glob

def calculate_fill_percentage(aspect_ratio):
    # Assuming aspect ratio represents the fill percentage linearly
    return (1 - aspect_ratio) * 100

def ImageProcess(path):
    bottle_3_channel = cv2.imread(path)
    img_gray = cv2.cvtColor(bottle_3_channel, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (3,3), 0)

    edges = cv2.Canny(image=img_blur, threshold1=100, threshold2=600) 
    (T, bottle_threshold) = cv2.threshold(edges, 0, 255, cv2.THRESH_BINARY_INV)

    contours = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    bottle_clone = bottle_3_channel.copy()

    cv2.drawContours(bottle_clone, contours, -1, (255, 0, 0), 2)

    areas = [cv2.contourArea(contour) for contour in contours]
    (contours, areas) = zip(*sorted(zip(contours, areas), key=lambda a:a[1]))
    bottle_clone = bottle_3_channel.copy()
    (x, y, w, h) = cv2.boundingRect(contours[-1])
    aspectRatio = w / float(h)
    fill_percentage = calculate_fill_percentage(aspectRatio)
    print("Fill Percentage:", fill_percentage)

    if aspectRatio < 0.4:
        cv2.rectangle(bottle_clone, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(bottle_clone, f"Full ({fill_percentage:.2f}%)", (x + 10, y + 20), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)
    elif aspectRatio > 0.75:
        cv2.rectangle(bottle_clone, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(bottle_clone, f"Low ({fill_percentage:.2f}%)", (x + 10, y + 20), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
    else:
        cv2.rectangle(bottle_clone, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(bottle_clone, f"medium ({fill_percentage:.2f}%)", (x + 10, y + 20), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)
        
        
    cv2.imshow("Decision", bottle_clone)
    cv2.waitKey(0)  
    
for img in glob.glob("TestImages/*.jpg"):
    ImageProcess(img)
