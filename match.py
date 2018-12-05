import cv2
import numpy as np


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
    for m in goodMatch[:20]:
        pt_ = kp1[m.queryIdx].pt
        if (pt_[0] > X1 and pt_[0] < X2) and (pt_[1] > Y1 and pt_[1] < Y2):
            goodM.append(m)
        #print(pt_)
    return goodM


def TransformImg(src, kp1, kp2, match):
    imgout=[]
    if len(match) > 3:
        ptsA = np.float32([kp1[m.queryIdx].pt for m in match]).reshape(
            -1, 1, 2)
        ptsB = np.float32([kp2[m.trainIdx].pt for m in match]).reshape(
            -1, 1, 2)
        ransacReprojThreshold = 4
        H, status = cv2.findHomography(ptsA, ptsB, cv2.RANSAC,
                                       ransacReprojThreshold)
        imgout = cv2.warpPerspective(
            src,
            H, (src.shape[1], src.shape[0]),
            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    return imgout


img1 = cv2.imread(r'base.bmp')
img2 = cv2.imread(r'images/4.bmp')

_, kp1, des1 = sift_kp(img1)
_, kp2, des2 = sift_kp(img2)
goodMatch = get_good_match(des1, des2)
good = GetInvalidMatches(goodMatch)

#img3 = cv2.drawMatches(img1, kp1, img2, kp2, good, None, flags=2)
#cv2.imshow('img3', img3)

img4 = TransformImg(img2,kp1,kp2,good)

cv2.imshow('img4', img4)
cv2.waitKey(0)
cv2.destroyAllWindows()