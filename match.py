import cv2
import numpy as np
import random
import os


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


def siftImageAlignment(img1, img2):
    _, kp1, des1 = sift_kp(img1)
    _, kp2, des2 = sift_kp(img2)
    goodMatch = get_good_match(des1, des2)
    if len(goodMatch) > 4:
        ptsA = np.float32([kp1[m.queryIdx].pt for m in goodMatch]).reshape(
            -1, 1, 2)
        ptsB = np.float32([kp2[m.trainIdx].pt for m in goodMatch]).reshape(
            -1, 1, 2)
        ransacReprojThreshold = 4
        H, status = cv2.findHomography(ptsA, ptsB, cv2.RANSAC,
                                       ransacReprojThreshold)
        imgOut = cv2.warpPerspective(
            img2,
            H, (img1.shape[1], img1.shape[0]),
            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    return imgOut, H, status


def GetInvalidMatches(goodMatches):
    X1 = 246
    Y1 = 234
    X2 = 515
    Y2 = 324
    goodM = []
    cnt=0
    for m in goodMatch[:10]:
        pt_ = kp1[m.queryIdx].pt
        if (pt_[0] > X1 and pt_[0] < X2) and (pt_[1] > Y1 and pt_[1] < Y2):
            goodM.append(m)
            cnt=cnt+1
        #print(pt_)
        if cnt==10:
            break
    return goodM


def TransformMat(kp1, kp2, match, flag):
    H = []
    if len(match) > 3:
        ptsA = np.float32([kp1[m.queryIdx].pt for m in match]).reshape(
            -1, 1, 2)
        ptsB = np.float32([kp2[m.trainIdx].pt for m in match]).reshape(
            -1, 1, 2)
        ransacReprojThreshold = 4
        if flag == 0:
            #H=cv2.getAffineTransform(ptsB, ptsA)
            H, status = cv2.findHomography(ptsA, ptsB, cv2.RANSAC,
                                           ransacReprojThreshold)
        else:
            #H=cv2.getAffineTransform(ptsA, ptsB)
            H, status = cv2.findHomography(ptsB, ptsA, cv2.RANSAC,
                                           ransacReprojThreshold)
    return H


def GetColor(src, rec):
    pixel = 0
    for i in range(0, 3):
        x = random.choice(range(rec[0], rec[2]))
        y = random.choice(range(rec[1], rec[3]))
        pixel = int(src[y, x])
    pixel = int(pixel / 3)
    return pixel


rect = [421, 286, 515, 323]
img1 = cv2.imread(r'base.bmp', 0)
imgdir = ('images/')
imglist = os.listdir(imgdir)
for imgn in imglist:
    print(imgn)
    img2 = cv2.imread(imgdir+imgn, 0)
    img2_ = np.zeros((980, 1312), dtype='uint8')
    img2_[245:735, 328:984] = img2
    img2 = img2_
    #cv2.imshow('img',img2)
    #cv2.waitKey(0)

    _, kp1, des1 = sift_kp(img1)
    _, kp2, des2 = sift_kp(img2)
    goodMatch = get_good_match(des1, des2)
    good = GetInvalidMatches(goodMatch)

    img3 = cv2.drawMatches(img1, kp1, img2, kp2, good, None, flags=2)
    #cv2.imshow('img3', img3)

    M = TransformMat(kp1, kp2, good, 0)
    #img4 = cv2.warpAffine(img2,M,(img2.shape[1], img2.shape[0]))
    img4=cv2.warpPerspective(
        img2,
        M, (img2.shape[1], img2.shape[0]),
        flags=cv2.INTER_CUBIC + cv2.WARP_INVERSE_MAP)
    #cv2.imwrite('1t.bmp',img4)

    img41 = img4[rect[1]:rect[3], rect[0]:rect[2]]
    blur41 = cv2.blur(img41, (17, 17))
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))
    dilated41 = cv2.dilate(blur41, element)
    img4[rect[1]:rect[3], rect[0]:rect[2]] = dilated41

    M_ = TransformMat(kp1, kp2, good, 1)
    #img5 = cv2.warpAffine(img4,M_,(img2.shape[1], img2.shape[0]))[245:735, 328:984]
    img5 = cv2.warpPerspective(
        img4,
        M_, (img2.shape[1], img2.shape[0]),
        flags=cv2.INTER_CUBIC + cv2.WARP_INVERSE_MAP)[245:735, 328:984]

    cv2.imshow('img4', img4[:489, :655])
    cv2.imshow('img5', img5)
    cv2.waitKey(0)
cv2.destroyAllWindows()