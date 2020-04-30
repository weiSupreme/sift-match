import cv2
import numpy as np
import random
import os
import shutil


def sift_kp(image):
    gray_image = image  #cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    kp, des = sift.detectAndCompute(image, None)
    kp_image = cv2.drawKeypoints(gray_image, kp, None)
    return kp_image, kp, des


def get_good_match(des1, des2):
    '''
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2) #des1为模板图，des2为匹配图
    matches = sorted(matches,key=lambda x:x[0].distance/x[1].distance)
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)
    return good
    '''
    indexParams = dict(algorithm=0, trees=5)
    searchParams = dict(checks=50)
    flann = cv2.FlannBasedMatcher(indexParams, searchParams)
    matches = flann.knnMatch(des1, des2, k=2)
    matches = sorted(matches, key=lambda x: x[0].distance / x[1].distance)
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)
    return good


def GetInvalidMatches(goodMatches):
    X1 = 233
    Y1 = 306
    X2 = 288
    Y2 = 341
    goodM = []
    cnt = 0
    for m in goodMatch:
        pt_ = kp1[m.queryIdx].pt
        if (pt_[0] > X1 and pt_[0] < X2) and (pt_[1] > Y1 and pt_[1] < Y2):
            goodM.append(m)
            cnt = cnt + 1
        #print(pt_)
        if cnt == 3:
            break
    return goodM


def TransformMat(kp1, kp2, match, flag):
    H = []
    if len(match) == 3:
        ptsA = np.float32([kp1[m.queryIdx].pt for m in match]).reshape(
            -1, 1, 2)
        ptsB = np.float32([kp2[m.trainIdx].pt for m in match]).reshape(
            -1, 1, 2)
        ransacReprojThreshold = 4
        if flag == 0:
            H = cv2.getAffineTransform(ptsB, ptsA)
            #H, status = cv2.findHomography(ptsA, ptsB, cv2.RANSAC,
            #                              ransacReprojThreshold)
        else:
            H = cv2.getAffineTransform(ptsA, ptsB)
            #H, status = cv2.findHomography(ptsB, ptsA, cv2.RANSAC,
            #                              ransacReprojThreshold)
    return H


def GetTransformedPoint(rect, M):
    m = np.array((rect[0], rect[1], 1), dtype='float')
    m1 = M.dot(m)
    m = np.array((rect[2], rect[3], 1), dtype='float')
    m2 = M.dot(m)
    pt1 = (min(int(m1[0]),0), min(int(m1[1]),0))
    pt2 = (max(int(m2[0]),655), max(int(m2[1]),489))
    return pt1, pt2


def GetTransformedPoint2(rect, M):
    m = np.array((rect[0], rect[1], 1), dtype='float')
    m1 = M.dot(m)
    m = np.array((rect[2], rect[3], 1), dtype='float')
    m2 = M.dot(m)
    pt1 = (int(m1[0]), int(m1[1]))
    pt2 = (int(m2[0]), int(m2[1]))
    return pt1, pt2


def PaintImage(src, pt1, pt2):
    x1, y1, x2, y2 = pt1[0], pt1[1], pt2[0], pt2[1]
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
    h, w = img.shape
    if h != 0 and w != 0:
        dilated = cv2.dilate(img, element)
        src[y1:y2, x1:x2] = dilated
    else:
        src = cv2.dilate(src, element)
    return src


def PaintImage2(src, pt1, pt2):
    x1, y1, x2, y2 = pt1[0], pt1[1], pt2[0], pt2[1]
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


def PaintImage3(src, pt1, pt2):
    x1, y1, x2, y2 = pt1[0], pt1[1], pt2[0], pt2[1]
    if x1 > x2:
        tmp = x2
        x2 = x1
        x1 = tmp
    if y1 > y2:
        tmp = y2
        y2 = y1
        y1 = tmp
    img = src[y1:y2, x1:x2]
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
    dilated = cv2.morphologyEx(img, cv2.MORPH_OPEN, element)
    src[y1:y2, x1:x2] = dilated
    return src


def PaintImage4(src, pt1, pt2):
    x1, y1, x2, y2 = pt1[0], pt1[1], pt2[0], pt2[1]
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


def PaintImage5(src, pt1, pt2):
    x1, y1, x2, y2 = pt1[0], pt1[1], pt2[0], pt2[1]
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


rects = [[234, 305, 384, 318],[231, 325, 385, 340],[235, 346, 382, 359],[234, 366, 386, 382]]
rect1 = [32, 42, 204, 210] 
img1 = cv2.imread(r'template.bmp', 0)
imgdir = ('train/')
imglist = os.listdir(imgdir)
imgc = len(imglist)
dir1 = 'train_nn_from_n/'
dir2 = 'train_n_from_n/'
dir3 = 'train_d_from_n/normMatch_'
dir4 = 'train_d_from_n_/normMatch_'
for imgn in imglist:
    print(imgn, ' ', imgc)
    imgc -= 1
    img2 = cv2.imread(imgdir + imgn, 0)

    _, kp1, des1 = sift_kp(img1)
    _, kp2, des2 = sift_kp(img2)
    goodMatch = get_good_match(des1, des2)
    good = GetInvalidMatches(goodMatch)
    if len(good) < 3:
        shutil.move(imgdir + imgn, dir1 + imgn)
        print(imgn, ' cant match')
        continue

    #img3 = cv2.drawMatches(img1, kp1, img2, kp2, good, None, flags=2)
    #cv2.imshow('img3', img3)
    #cv2.waitKey(0)

    M = TransformMat(kp1, kp2, good, 0)
    if M is None:
        shutil.move(imgdir + imgn, dir1 + imgn)
        print(imgn, ' cant match')
        continue
    img4 = cv2.warpAffine(img2, M, (656, 490))
    flag = 0
    while (1):
        cv2.imshow('imgout', img4)
        keyin = cv2.waitKey(50)
        if keyin & 0xFF == 27:
            flag = 1
            shutil.move(imgdir + imgn, dir1 + imgn)
            print(imgn, ' cant match')
            break
        elif keyin & 0xFF == 13:
            break
    if flag:
        continue
    #continue
    cnt = 0
    sps = random.sample(range(0, 4), 1)
    for sp in sps:
        img2 = cv2.imread(imgdir + imgn, 0)
        pt1, pt2 = GetTransformedPoint2(rects[sp],
                                        cv2.invertAffineTransform(M))
        imgout = PaintImage(img2, pt1, pt2)

        imgnn = dir3 +str(sp) + '_' + imgn
        cv2.imwrite(imgnn, imgout)

        #cv2.imshow('imgout', imgout)
        #cv2.waitKey(0)
        cnt = cnt + 1
    prob = random.choice(range(0, 2))
    if prob==17:
        pt1 = (rect1[0],rect1[1])
        pt2 = (rect1[2],rect1[3])

        img2 = cv2.imread(imgdir + imgn, 0)
        imgout = PaintImage(img2, pt1, pt2)
        cv2.imwrite(dir4+'01_'+imgn, imgout)

        img2 = cv2.imread(imgdir + imgn, 0)
        imgout = PaintImage2(img2, pt1, pt2)
        cv2.imwrite(dir4+'02_'+imgn, imgout)
        
        img2 = cv2.imread(imgdir + imgn, 0)
        imgout = PaintImage3(img2, pt1, pt2)
        cv2.imwrite(dir4+'03_'+imgn, imgout)
        
        img2 = cv2.imread(imgdir + imgn, 0)
        imgout = PaintImage4(img2, pt1, pt2)
        cv2.imwrite(dir4+'04_'+imgn, imgout)
        
        img2 = cv2.imread(imgdir + imgn, 0)
        imgout = PaintImage5(img2, pt1, pt2)
        cv2.imwrite(dir4+'05_'+imgn, imgout)
    shutil.move(imgdir + imgn, dir2 + imgn)
cv2.destroyAllWindows()