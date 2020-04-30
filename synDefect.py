import cv2
import numpy as np
import random
import os


def PaintImage(src, rect):
    x1, y1, x2, y2 = rect
    if x1 > x2:
        tmp = x2
        x2 = x1
        x1 = tmp
    if y1 > y2:
        tmp = y2
        y2 = y1
        y1 = tmp
    
    img = src[y1:y2, x1:x2]
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    h,w=img.shape
    if h!=0 and w!=0:
        dilated = cv2.dilate(img, element)
        src[y1:y2, x1:x2] = dilated
    else:
        src = cv2.dilate(src, element)
    return src


def PaintImage2(src, rect):
    x1, y1, x2, y2 = rect
    if x1 > x2:
        tmp = x2
        x2 = x1
        x1 = tmp
    if y1 > y2:
        tmp = y2
        y2 = y1
        y1 = tmp
    img = src[y1:y2, x1:x2]
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate(img, element)
    src[y1:y2, x1:x2] = dilated
    return src


def PaintImage3(src, rect):
    x1, y1, x2, y2 = rect
    if x1 > x2:
        tmp = x2
        x2 = x1
        x1 = tmp
    if y1 > y2:
        tmp = y2
        y2 = y1
        y1 = tmp
    img = src[y1:y2, x1:x2]
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 1))
    dilated = cv2.morphologyEx(img, cv2.MORPH_OPEN, element)
    src[y1:y2, x1:x2] = dilated
    return src


def PaintImage4(src, rect):
    x1, y1, x2, y2 = rect
    if x1 > x2:
        tmp = x2
        x2 = x1
        x1 = tmp
    if y1 > y2:
        tmp = y2
        y2 = y1
        y1 = tmp
    img = src[y1:y2, x1:x2]
    std = 5
    h_, w_ = img.shape
    if y1 + h_ > 218:
        std = -5
    img2 = src[y1 + std:y1 + h_ + std, x1:x1 + w_]
    img3 = np.zeros((h_, w_), dtype='uint8')
    for h in range(h_):
        for w in range(w_):
            if img[h, w] <= img2[h, w]:
                img3[h, w] = img[h, w] - 0 if img[h, w] - 0 > 0 else 0
            elif img[h, w] > img2[h, w]:
                img3[h, w] = img2[h, w] - 0 if img2[h, w] - 0 > 0 else 0
    src[y1:y2, x1:x2] = img3
    return src


def PaintImage5(src, rect):
    x1, y1, x2, y2 = rect
    if x1 > x2:
        tmp = x2
        x2 = x1
        x1 = tmp
    if y1 > y2:
        tmp = y2
        y2 = y1
        y1 = tmp
    img = src[y1:y2, x1:x2]
    std = 5
    h_, w_ = img.shape
    if x1 + w_ > 218:
        std = -5
    img2 = src[y1:y1 + h_, x1 + std:x1 + w_ + std]
    img3 = np.zeros((h_, w_), dtype='uint8')
    for h in range(h_):
        for w in range(w_):
            if img[h, w] <= img2[h, w]:
                img3[h, w] = img[h, w] - 0 if img[h, w] - 0 > 0 else 0
            elif img[h, w] > img2[h, w]:
                img3[h, w] = img2[h, w] - 0 if img2[h, w] - 0 > 0 else 0
    src[y1:y2, x1:x2] = img3
    return src

       
rect1 = [32, 42, 204, 210]         
savedir='dataset/train/defect/normSyn_'
imgl=os.listdir('normSamples')
for imgn in imgl:
    print(imgn)
    img = cv2.imread('normSamples/' + imgn, 0)
    
    img2=img.copy()
    prob = random.choice(range(0, 5))
    if prob==0:
        imgout=PaintImage(img2, rect1)
        cv2.imwrite(savedir+'A_'+imgn,imgout)
    elif prob==1:
        imgout=PaintImage2(img2, rect1)
        cv2.imwrite(savedir+'C_'+imgn,imgout)
    elif prob==2:
        imgout=PaintImage3(img2, rect1)
        cv2.imwrite(savedir+'D_'+imgn,imgout)
    elif prob==3:
        imgout=PaintImage4(img2, rect1)
        cv2.imwrite(savedir+'E_'+imgn,imgout)
    elif prob==4:
        imgout=PaintImage5(img2, rect1)
        cv2.imwrite(savedir+'F_'+imgn,imgout)
    