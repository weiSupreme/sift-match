import os
import cv2 as cv
flag = False
rect = [0, 0, 0, 0]
img = []
img1 = []
txt=[]


def Draw(events, x, y, flags, param):
    global flag, rect, img, img1,txt
    if events == cv.EVENT_LBUTTONDOWN:
        rect[0] = x
        rect[1] = y
        flag = True
    elif events == cv.EVENT_LBUTTONUP:
        flag = False
        txt.write(str(rect)+',')
    if events == cv.EVENT_MOUSEMOVE:
        img3=img1.copy()
        cv.line(img3,(0,y),(655,y),(127,127,127),1)
        cv.line(img3,(x,0),(x,489),(127,127,127),1)
        img=img3.copy()
        if flag:
            rect[2] = x
            rect[3] = y
            img2 = img1.copy()
            cv.rectangle(img2, (rect[0], rect[1]), (rect[2], rect[3]),
                         (0, 0, 0), 1)
            img = img2.copy()


cv.namedWindow('draw', cv.WINDOW_NORMAL)
cv.setMouseCallback('draw', Draw)
txt=open('rect.txt','w')
txt.write('[')
img = cv.imread('template.bmp')
img1 = img.copy()
while (1):
    cv.imshow('draw', img)
    keyin = cv.waitKey(100)
    if keyin & 0xFF == 27:
        break
cv.destroyAllWindows()
txt.write(']')
txt.close()