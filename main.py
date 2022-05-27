import detect
import cv2

# 在此输入图片名称
img_name = '8.jpg'
img = cv2.imread(img_name)

if img is None:
    print('Open failed please check the filename!')
else:
    detect.platenum_detect(img)

