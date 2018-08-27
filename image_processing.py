import cv2
import numpy as np
import matplotlib.pylab as pl
from PIL import Image
import os

def enhance(img_ndarray):
    list = []
    for i in range(3):
        list.append(cv2.equalizeHist(img_ndarray[:,:,i]))
    cv2.merge(list, img_ndarray)
    return img_ndarray

#特征图生成并显示
def getGabor(img):
    image = cv2.imread(img)
    img_ndarray = np.asarray(image)
    #cv2.imshow('ori',img_ndarray)
    img_ndarray = enhance(img_ndarray) 
    #cv2.imshow('en',img_ndarray)
    #Sobel边缘检测
    sobelX = cv2.Sobel(img_ndarray,cv2.CV_64F,1,0)#x方向的梯度
    sobelY = cv2.Sobel(img_ndarray,cv2.CV_64F,0,1)#y方向的梯度
    
    sobelX = np.uint8(np.absolute(sobelX))#x方向梯度的绝对值
    sobelY = np.uint8(np.absolute(sobelY))#y方向梯度的绝对值
    
    sobelCombined = cv2.bitwise_or(sobelX,sobelY)
    #cv2.imshow("Sobel", sobelCombined)
    img_new = cv2.addWeighted(img_ndarray,0.9,sobelCombined,0.1,0)
    #cv2.imshow("res", img_new)
    #cv2.waitKey()
    return img_new
    

if __name__ == '__main__':  
    dir = 'data/test_b'
    index = 0
    for root,dirs,files in os.walk(dir):
        for file in files:
            filepath = os.path.join(root, file)
            filesuffix = os.path.splitext(filepath)[1][1:]            
            if(filesuffix == 'jpg'):
                img_new = getGabor(filepath)
                cv2.imwrite(root+ "/%s"%(file),img_new)
        print(root)