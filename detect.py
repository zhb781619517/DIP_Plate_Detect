import cv2
import numpy as np


def point_limit(point):
    if point[0] < 0:
        point[0] = 0
    if point[1] < 0:
        point[1] = 0
    return point


def show_pic(pic, name='img'):
    cv2.imshow(name, pic)


# Roberts算子
kernelx = np.array([[-1, 0], [0, 1]], dtype=int)
kernely = np.array([[0, -1], [1, 0]], dtype=int)


# Roberts边缘检测
def Roberts(img):
    gray = img
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    x = cv2.filter2D(gray, cv2.CV_16S, kernelx)
    y = cv2.filter2D(gray, cv2.CV_16S, kernely)
    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)
    return cv2.addWeighted(absX, 0.5, absY, 0.5, 0)


def platenum_detect(img):
    # 读取图片
    img = cv2.resize(img, (600, int(600 * img.shape[0] / img.shape[1])))  # 改变尺寸使其合适

    # 锐化核
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]
                       ])
    img = cv2.filter2D(img, -1, kernel)  # 锐化

    # 高斯滤波
    img = cv2.GaussianBlur(img, (5, 5), 0)
    oldimg = img  # 保存原始图像

    # 展示原始图像
    show_pic(oldimg, 'originpic')
    # 将图像转为灰度图
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Roberts边缘检测
    ret, img_thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # 二值化
    img_edge = Roberts(img_thresh)
    # show_pic(img_edge, 'back_edge') # 展示效果

    # 3次闭运算2次开运算
    # kernel = np.ones((20, 20), np.uint8)
    kernel = np.ones((5, 19), np.uint8)
    img_edge1 = cv2.morphologyEx(img_edge, cv2.MORPH_CLOSE, kernel, iterations=2)
    img_edge1 = cv2.morphologyEx(img_edge1, cv2.MORPH_OPEN, kernel, iterations=3)
    # 腐蚀膨胀保持一致 3次腐蚀3次膨胀
    img_edge1 = cv2.dilate(img_edge1, kernel, iterations=3)
    img_edge1 = cv2.erode(img_edge1, kernel, iterations=3)
    # show_pic(img_edge1, 'edge')

    # 得到候选区
    contours, hierarchy = cv2.findContours(img_edge1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    print(len(contours))
    contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 1200]  # 去除像素点数小于1200的区域

    print('contours = ', end='')
    print(len(contours))

    # 去除长宽比小于2或大于5.5的
    new_contours = []
    for cnt in contours:
        rect = cv2.minAreaRect(cnt)
        area_width, area_height = rect[1]
        if area_width < area_height:
            area_width, area_height = area_height, area_width
        wh_ratio = area_width / area_height
        if wh_ratio > 2 and wh_ratio < 5.5:
            new_contours.append(cnt)

    print('new_Contours = ', end='')
    print(len(new_contours))

    # 将目前待选区域以红色线为边缘框入图像中
    # for cnt in new_contours:
    #     cv2.drawContours(oldimg, [cnt], 0, (0, 0, 255), 2)
    # show_pic(oldimg, 'ooooo')

    card_imgs = []  # 存储可疑图像
    # 将可疑区域分割
    for cnt in new_contours:
        # 生成最小外接矩形
        rect = cv2.minAreaRect(cnt)
        rect = (rect[0], (rect[1][0] + 5, rect[1][1] + 5), rect[2])  # 将图像边缘扩大，防止车牌边缘未被纳入
        box = cv2.boxPoints(rect)  # 得到四个角的坐标
        pic_width, pic_hight = img.shape[:2]  # 图像长和宽
        min_x = pic_width
        min_y = pic_hight
        max_x = 0
        max_y = 0
        # 算出该矩形所对应的区域
        for point in box:
            if point[1] > max_y:
                max_y = point[1]
            if point[1] < min_y:
                min_y = point[1]
            if point[0] > max_x:
                max_x = point[0]
            if point[0] < min_x:
                min_x = point[0]
        card_img = oldimg[int(min_y): int(max_y), int(min_x): int(max_x)]  # 截取区域，在之后做筛选
        card_imgs.append(card_img)
    result_img = []

    print('cards_imgs = ', end='')
    print(len(card_imgs))

    # for i, pic in enumerate(card_imgs):
    #     try:
    #         show_pic(pic, 'i' + str(i))
    #     except:
    #         continue

    result_img = []
    resultpic = card_imgs[0]
    max_blue = 0  # 记录当前车牌蓝最大比值
    for card_img in card_imgs:
        # try:
        #     show_pic(card_img)
        # except:
        #     continue
        blue = 0
        try:
            card_img_hsv = cv2.cvtColor(card_img, cv2.COLOR_BGR2HSV)  # 将BGR图转HSV图
        except:
            card_img_hsv = None
        if card_img_hsv is None:
            continue

        row_num, col_num = card_img_hsv.shape[:2]  # 通过长宽来遍历该图像各个像素点做统计
        card_img_count = row_num * col_num

        # 确定车牌颜色
        for i in range(row_num):
            for j in range(col_num):
                H = card_img_hsv.item(i, j, 0)
                S = card_img_hsv.item(i, j, 1)
                V = card_img_hsv.item(i, j, 2)
                if 99 < H <= 124 and S > 34:
                    blue += 1  # 如果满足蓝色像素条件，则计数
        print('blue = {}'.format(blue), end='\t')
        print('col * row = {}'.format(col_num * row_num))
        if (blue / (col_num * row_num)) > max_blue:  # 记录当前蓝色像素最大比值图片
            max_blue = (blue / (col_num * row_num))
            resultpic = card_img
        if 4 * blue > col_num * row_num:
            result_img.append(card_img)

    # print('result_img = ', end='')
    # print(len(result_img))
    if len(result_img) == 0:  # 如果没有一个待选图像素满足要求，则选出最大的图片输出
        t_img = cv2.resize(resultpic, (200, int(200 * resultpic.shape[0] / resultpic.shape[1])))
        show_pic(t_img, str('tag'))
    else:
        for i, card_img in enumerate(result_img):  # 如果有多张满足图片，则全部输出
            t_img = cv2.resize(card_img, (200, int(200 * card_img.shape[0] / card_img.shape[1])))
            show_pic(t_img, 'tag' + str(i))
    cv2.waitKey(0)
