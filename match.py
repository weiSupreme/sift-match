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
    cnt = 0
    for m in goodMatch[:10]:
        pt_ = kp1[m.queryIdx].pt
        if (pt_[0] > X1 and pt_[0] < X2) and (pt_[1] > Y1 and pt_[1] < Y2):
            goodM.append(m)
            cnt = cnt + 1
        #print(pt_)
        if cnt == 10:
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


def GetTransformedPoint(rect, M):
    pts = np.float32([[rect[0], rect[1]], [rect[2], rect[3]]]).reshape(
        -1, 1, 2)
    ptsT = cv2.perspectiveTransform(pts, M)
    
    pyPts = [tuple(npt[0]) for npt in ptsT.astype(int).tolist()]
    return pyPts


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
    blur = cv2.blur(img, (17, 17))
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))
    dilated = cv2.dilate(blur, element)
    src[y1:y2, x1:x2] = dilated
    return cv2.blur(src, (3, 3))


rects = [[248, 239, 278, 281], [277, 242, 303, 279], [306, 241, 329, 283],
         [334, 244, 365, 283], [364, 242, 388, 282], [394, 243, 416, 285],
         [423, 242, 450, 281], [455, 238, 483, 277], [246, 286, 270, 322],
         [272, 285, 296, 325], [332, 288, 363, 324], [362, 287, 392, 325],
         [421, 285, 453, 323], [451, 285, 477, 321], [480, 290, 512, 321],
         [248, 239, 367, 285], [364, 237, 486, 283], [245, 285, 297, 324],
         [330, 285, 392, 326], [419, 285, 512, 326], [246, 238, 328, 323],
         [331, 242, 417, 326], [424, 238, 513, 324], [248, 237, 484, 281],
         [244, 283, 511, 329], [247, 237, 513, 328]]

img1 = cv2.imread(r'base.bmp', 0)
imgdir = ('images/')
imglist = os.listdir(imgdir)
for imgn in imglist:
    print(imgn)
    img2 = cv2.imread(imgdir + imgn, 0)

    _, kp1, des1 = sift_kp(img1)
    _, kp2, des2 = sift_kp(img2)
    goodMatch = get_good_match(des1, des2)
    good = GetInvalidMatches(goodMatch)
    if len(good) < 5:
        print(imgn, ' cant match')
        continue

    img3 = cv2.drawMatches(img1, kp1, img2, kp2, good, None, flags=2)
    #cv2.imshow('img3', img3)
    #cv2.waitKey(0)

    M = TransformMat(kp1, kp2, good, 0)
    img4=cv2.warpPerspective(
        img2,
        M, (img2.shape[1], img2.shape[0]),
        flags=cv2.INTER_CUBIC + cv2.WARP_INVERSE_MAP)
    flag=0
    while(1):
        cv2.imshow('imgout',img4)
        keyin=cv2.waitKey(50)
        if keyin & 0xFF ==27:
            flag=1
            break
        elif keyin & 0xFF ==9:
            break
    if flag:
        continue

    cnt = 0
    for rect in rects:
        img2 = cv2.imread(imgdir + imgn, 0)
        pts = GetTransformedPoint(rect, M)
        pt1, pt2 = pts[0], pts[1]
        imgout = PaintImage(img2, pt1, pt2)

        imgnn = str(cnt) + '_' + imgn
        #cv2.imwrite(imgnn,imgout)

        cv2.imshow('imgout', imgout)
        cv2.waitKey(0)
        cnt = cnt + 1
cv2.destroyAllWindows()