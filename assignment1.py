import os

import cv2 as cv
import numpy as np
import glob as gb
import skimage.color
import skimage.measure
import math

def start():
    # Set intial Path #
    dirPath="sample_drive"

    for camFolder in os.listdir(dirPath):
        inputPath=os.path.join(dirPath, camFolder)
        outputPath=os.path.join("output",camFolder)
        images = gb.glob(inputPath + "/*.jpg")
        ## Pre Process Images ##
        firstImage=cvReadImage(images[0])
        preProcessedImages=preProcessImages(inputPath, outputPath, 1000, images)
        #postProcessedImage = postProcessImage(preProcessedImages)
        ## Calculate Mean Image ##
        # batchedPreprocessedImages=splitData(preProcessedImages)
        # batchedPostProcess(firstImage,batchedPreprocessedImages)

        meanImage=calculateMeanImage(preProcessedImages)
        cvWriteImage(meanImage, "Mean_Image.jpg", outputPath)
        ## Post Process Mean Image ##
        postProcessedImage = postProcessImage(meanImage)
        cvWriteImage(postProcessedImage,"Masked.jpg",outputPath)


def isBright(image, threshold):
    x,Y,V=cv.split(cv.cvtColor(image, cv.COLOR_BGR2HSV))
    V=V/np.max(V)
    return V.mean() < threshold

def splitData(images):
    # Split images into 500 for memory constraints ##
    batchSize = 500
    size = len(images)
    divisions = int(size / batchSize)
    counter = 0
    imageBatchedList = []
    print("Batching into size of ",batchSize," of ",divisions," divisions")

    for x in range(divisions):
        max = counter + batchSize
        if max < size:
            images_split = images[counter:max]
        else:
            images_split = images[counter:]
        imageBatchedList.append(images_split)
        counter += batchSize

    print("Image batched list of size ",len(imageBatchedList))

    return imageBatchedList

def postProcessImage(input):

    # Post Processed 1 #
    adaptiveThreshold=cvAdaptiveThreshold(input,105,10)
    postProcessedImage=cvBitwiseNot(adaptiveThreshold)
    eroded=cvErode(postProcessedImage)
    finalPostProcessedImage=cvDilate(eroded)
    return finalPostProcessedImage

    # Post Processed 2 #
    # postProcess=extendedFunctions(cvErode, cvDilate, cvMedianBlur, cvApplyThreshold(120), cvDrawContours(5))
    # images=list(input)
    # result=sum([postProcess(img) / len(img) for img in images])

    # Connected-component labeling
    # labeled_image = skimage.measure.label(postProcessedImage, connectivity=2, return_num=True)
    # # Convert the label image to color image
    # colored_label_image = skimage.color.label2rgb(labeled_image[0], bg_label=0)

    # return postProcessedImage

def preProcessImages(inputPath, outputPath, batchSize, images):
    print("Pre Processing ",batchSize," images in ",inputPath)
    preProcessedImages=[]
    counter=0
    for imagePath in images:
        if counter > batchSize:
            break
        image=cvReadImage(imagePath)
        if (isBright(image,0.6)):
            #preProcessedImage=cvCannyEdgeDetection(cvDilate(cvApplyThreshold(cvConvertToGrayscale(image))))

            # Preprocessed 1 #
            grayscale=cvConvertToGrayscale(image)
            equalistHist=cvEqualizeHist(grayscale)
            gaussianBlur=cvGaussianBlur(equalistHist)
            preProcessedImages.append([gaussianBlur])

            # Preprocessed 2 #
            # grayscale=cvConvertToGrayscale(image)
            # threshold=cvApplyThreshold(grayscale,120)
            # dilated=cvDilate(threshold)
            # cannyEdgeDetected=cvCannyEdgeDetection(dilated)
            # preProcessedImages.append([cannyEdgeDetected])

            counter+=1

    return preProcessedImages

def calculateMeanImage(imageList):

    return getMean(imageList)
    # meanImageList=[]
    # for images in imageList:
    #     print("Calculating mean for ",len(images)," images!")
    #     meanImage=getMean(images)
    #     meanImageList.append(meanImage)
    #
    # return getMean(meanImageList)

def getMean(images):
    avgImage = np.mean(images, axis=0)
    avgImage = avgImage.astype(np.uint8)
    avgImage=avgImage.reshape(avgImage.shape[1:])

    return avgImage

    # firstImage=images[0]
    #
    # for i in range(len(images)):
    #     if i==0:
    #         pass
    #     else:
    #         alpha=1.0/(i+1)
    #         beta=1.0-alpha
    #         firstImage=cv.addWeighted(images[i],alpha,firstImage,beta,0.0)
    #
    # return firstImage
    # size=len(images)
    # meanImage=images[0] * 1/size
    # index=0
    # while index < size:
    #     meanImage=cv.add(meanImage, images[index]*1/size)
    #     index+=1
    #
    # return meanImage


# https://docs.opencv.org/master/d7/d4d/tutorial_py_thresholding.html #
# Setting binary threshold value to 125 #
def cvApplyThreshold(image, low):
    ret, thresholdedImage=cv.threshold(image,low,255,cv.THRESH_BINARY)
    return thresholdedImage


def cvCannyEdgeDetection(image):
    return cv.Canny(image,125,150)

def cvCLAHE(image):
    clahe=cv.createCLAHE(2.0,(8,8))
    return clahe.apply(image)

def cvEqualizeHist(image):
    return cv.equalizeHist(image)

def cvGaussianBlur(image):
    return cv.GaussianBlur(image,(3,3),0)

def cvBitwiseNot(image):
    return cv.bitwise_not(image)


def readImages(inputPath):
    print("Reading images in folder "+inputPath)
    images=[]
    for imagePath in os.listdir(inputPath):
        images.append(cvReadImage(os.path.join(inputPath,imagePath)))

    return images

def writeAllImages(images, outputPath):
    if not os.path.isdir(outputPath):
        os.mkdir(outputPath)
    for image in images:
        cvWriteImage(image, outputPath)

def cvWriteImage(image, imageName, outputPath):
    print("Writing image ",imageName," at ", outputPath)
    if not os.path.isdir(outputPath):
        os.mkdir(outputPath)
    return cv.imwrite(outputPath+"/"+imageName, image)

def cvReadImage(image):
    return cv.imread(image)

def cvAdaptiveThreshold(image, block, c):
    return cv.adaptiveThreshold(image,255,cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY, block,c)

def cvConvertToGrayscale(image):
    return cv.cvtColor(image, cv.COLOR_BGR2GRAY)

# Kernel is like matrix of odd size #
# We do this to increase size of white portions #
def cvDilate(image):
    kernel=np.ones((3,3), np.uint8)
    return cv.dilate(image, kernel)

def cvErode(image):
    kernel=np.ones((5,5), np.uint8)
    return cv.erode(image, kernel)

def cvMedianBlur(image):
    return cv.medianBlur(image,7)

def extendedFunctions(*functions):
    def func(input):
        result=input
        for function in list(functions):
            result=function(result)
        return result
    return func


if __name__=="__main__":
    start()
