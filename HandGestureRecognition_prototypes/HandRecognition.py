import cv2
import numpy as np
import os
from Tools import RegionProps
import Tools as tools

global regionProps
regionProps = RegionProps()

'''
- image in YCrCb
- restrict channels
- biggest contour as hand
77 < Cb < 127
133 < Cr < 173
'''
def approach1(img):
    #print img.shape
    #Y, Cb, Cr = cv2.split(img)
    #print Y
    imgColor = img.copy()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    minRange = cv2.cv.Scalar(1.0, 133, 102)
    maxRange = cv2.cv.Scalar(255, 173 , 128)
    img = cv2.inRange(img, minRange, maxRange)
    contours, _ = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    #biggestContour[0] == contour, biggestContour[1] == props
    biggestContour = contours[0]
    biggestContourProps = regionProps.CalcContourProperties(contours[0], ["Area", "Boundingbox", "Centroid", "Extend"])
    for contour in contours:
        #print contour
        props = tools.RegionProps.CalcContourProperties(contour, ["Area", "Boundingbox", "Centroid", "Extend"])
        if(biggestContourProps["Area"] < props["Area"]):        
            biggestContour = contour
            biggestContourProps = props

    center = (int(biggestContourProps["Centroid"][0]), int(biggestContourProps["Centroid"][1]))
    cv2.circle(imgColor, center, 5, (0, 0, 255), (int(biggestContourProps["Area"]*.00005) + 1) ) 
    cv2.drawContours(imgColor, [biggestContour], -1, (0, 255, 0))
    hull =  cv2.convexHull(biggestContour)
    cv2.drawContours(imgColor, [hull], -1, (0, 0, 255))

    return imgColor

'''
-------------------------------
- image in HSV
- restrict channels
- biggest contour as hand
'''
def approach2(img):
    imgColor = img.copy()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    minRange = cv2.cv.Scalar(1.0 , 94.0, 1.0)
    maxRange = cv2.cv.Scalar(29.0, 255.0, 255.0)
    img = cv2.inRange(img, minRange, maxRange)
    #tools.showImages(testing=img)
    contours, _ = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    #biggestContour[0] == contour, biggestContour[1] == props
    biggestContour = contours[0]
    biggestContourProps = regionProps.CalcContourProperties(contours[0], ["Area", "Boundingbox", "Centroid", "Extend"])
    for contour in contours:
        #print contour
        props = tools.RegionProps.CalcContourProperties(contour, ["Area", "Boundingbox", "Centroid", "Extend"])
        if(biggestContourProps["Area"] < props["Area"]):        
            biggestContour = contour
            biggestContourProps = props

    center = (int(biggestContourProps["Centroid"][0]), int(biggestContourProps["Centroid"][1]))
    cv2.circle(imgColor, center, 5, (0, 0, 255), (int(biggestContourProps["Area"]*.00005) + 1) ) 
    cv2.drawContours(imgColor, [biggestContour], -1, (0, 255, 0))
    hull =  cv2.convexHull(biggestContour)
    cv2.drawContours(imgColor, [hull], -1, (0, 0, 255))
    
    return imgColor


'''
--------------------------
#Mean-Shift Pyramid Segmentation
#color segmentation
#pyrMeanShiftFiltering(src, sp, sr)
#sp = spatial window radius
#sr = color window radius
#img = HSV
- biggest contour as hand
'''
def approach3(img):
    print "init approach 3"
    imgColor = img.copy()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    image = cv2.pyrMeanShiftFiltering(img, 40, 30)
    #cv2.imshow("pyr", image)
    minRange = cv2.cv.Scalar(1.0 , 94.0, 1.0)
    maxRange = cv2.cv.Scalar(29.0, 255.0, 255.0)
#    min = cv2.cv.Scalar(1.0 , 91.0, 89.0)
#    max = cv2.cv.Scalar(25.0, 173.0, 229.0)
    image = cv2.inRange(image, minRange, maxRange)
    #cv2.imshow("range", image)
    contours, _ = cv2.findContours(image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    biggestContour = contours[0]
    biggestContourProps = regionProps.CalcContourProperties(contours[0], ["Area", "Boundingbox", "Centroid", "Extend"])
    for contour in contours:
        props = tools.RegionProps.CalcContourProperties(contour, ["Area", "Boundingbox", "Centroid", "Extend"])
        if(biggestContourProps["Area"] < props["Area"]):        
            biggestContour = contour
            biggestContourProps = props

    print biggestContour
    center = (int(biggestContourProps["Centroid"][0]), int(biggestContourProps["Centroid"][1]))
    cv2.circle(imgColor, center, 5, (0, 0, 255), (int(biggestContourProps["Area"]*.00005) + 1) ) 
    cv2.drawContours(imgColor, [biggestContour], -1, (0, 255, 0))
    hull =  cv2.convexHull(biggestContour)
    cv2.drawContours(imgColor, [hull], -1, (0, 0, 255))
    print "done approach 3"
    return imgColor

'''
#--------------------------
#floodFill operation on center of biggest contour 
'''
def approach4(img):
    print "init approach 4"
    imgColor = img.copy()
    imgGray = cv2.cvtColor(imgColor, cv2.COLOR_BGR2GRAY)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    #image = cv2.pyrMeanShiftFiltering(img, 40, 30)
    minRange = cv2.cv.Scalar(1.0 , 94.0, 1.0)
    maxRange = cv2.cv.Scalar(29.0, 255.0, 255.0)
    #cv2.imshow("pyrMean", image)
    image = cv2.inRange(img, minRange, maxRange)    
    #cv2.imshow("range", image)
    #cv2.waitKey(0)
    contours, _ = cv2.findContours(image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    biggestContourProps = regionProps.CalcContourProperties(contours[0], ["Area", "Centroid"])
    for contour in contours:
        props = tools.RegionProps.CalcContourProperties(contour, ["Area", "Centroid"])
        if(biggestContourProps["Area"] < props["Area"]):        
            biggestContourProps = props

    center = (int(biggestContourProps["Centroid"][0]), int(biggestContourProps["Centroid"][1]))
    #cv2.circle(imgColor, center, 5, (0, 0, 255), (int(biggestContourProps["Area"]*.00005) + 1) ) 
    #cv2.drawContours(imgColor, [biggestContour], -1, (0, 255, 0))
    #hull =  cv2.convexHull(biggestContour)
    #cv2.drawContours(imgColor, [hull], -1, (0, 0, 255))
    h, w = imgGray.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    mask[:] = 0
    cv2.floodFill(imgGray, mask, center, (255,255,255), 10, 5)
    cv2.imshow("flooded", imgGray)
    cv2.waitKey(0)
    
    print "done approach 4"
    return imgColor

'''
#----------------
- History equalization
- pyrMeanShiftFiltering on equalized image
- image to HSV and restrict channels
- biggest contour as hand
'''
def approach5(img):
    print "init 5"
    imgColor = img.copy()
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgGray = cv2.equalizeHist(imgGray)

    imageEq = cv2.pyrMeanShiftFiltering(img, 40, 30)
    imageEq = cv2.cvtColor(imageEq, cv2.COLOR_BGR2HSV)
    
    minRange = cv2.cv.Scalar(1.0 , 94.0, 1.0)
    maxRange = cv2.cv.Scalar(29.0, 255.0, 255.0)
    imageEq = cv2.inRange(imageEq, minRange, maxRange)
    contours, _ = cv2.findContours(imageEq, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    biggestContour = contours[0]
    biggestContourProps = regionProps.CalcContourProperties(contours[0], ["Area", "Centroid"])
    for contour in contours:
        props = regionProps.CalcContourProperties(contour, ["Area", "Centroid"])
        #cv2.drawContours(imgColor, contour, -1, (255, 255, 0))
        if(biggestContourProps["Area"] < props["Area"]):        
            biggestContour = contour
            biggestContourProps = props

    center = (int(biggestContourProps["Centroid"][0]), int(biggestContourProps["Centroid"][1]))
    cv2.drawContours(imgColor, [biggestContour], -1, (0, 255, 0), 3)
    cv2.circle(imgColor, center, 5, (0, 0, 255), (int(biggestContourProps["Area"]*.00005) + 1) ) 
    hull =  cv2.convexHull(biggestContour)
    cv2.drawContours(imgColor, [hull], -1, (0, 0, 255), 2)
    print "end 5"
    return imgColor
#------ end of approaches-------- #

'''
--------------------------------------------------------------------------
DENSE SIFT + K-Nearest Neighbor
- get descriptors from picture
- return descriptors
'''
def approach6GetDescriptors(imagename,resultname,size=20,steps=10, force_orientation=False,resize=None):
    """ Process an image with densely sampled SIFT descriptors
    and save the results in a file. Optional input: size of features,
    steps between locations, forcing computation of descriptor orientation (False means all are oriented upwards), tuple for resizing the image."""
    #im = Image.open(imagename).convert('L') #converts image to grayscale
    imgGray = cv2.cvtColor(cv2.imread('HandGestureDB/train/'+str(imagename)), cv2.COLOR_BGR2GRAY)
    if resize!=None:
        imgGray = cv2.resize(imgGray, resize)
    m,n = imgGray.shape
    '''
    will use this when training with different gestures
    if imagename[-3:] != 'pgm': # create a pgm file
        np.save('HandGestureDB/savedData/tmp.pgm', imgGray)
        #im.save('tmp.pgm') 
        imagename = 'tmp.pgm'
    '''
    #
    #change frame to opencv mask
    #
    # create frames and save to temporary file
    #scale = size/3.0
    x,y = np.meshgrid(range(steps,m,steps),range(steps,n,steps))
    mask = zip(x.flatten(),y.flatten())
    keyPoints = [cv2.KeyPoint(x, y, size) for (x, y) in mask]

    sift = cv2.SIFT()
    #shape = number of KeyPoints*128
    #print mask
    keyPoints, descriptors = sift.compute(imgGray, keyPoints)
    np.save('HandGestureDB/descriptors/SIFT/'+resultname, descriptors)
    #imgGray=cv2.drawKeypoints(imgGray,keyPoints,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    #print("Keypoints", keyPoints)
    #print("Descriptors", descriptors)  
    #cv2.imshow('sift_keypoints.jpg',imgGray)
    return descriptors
    #cv2.waitKey(0)

'''
approach 6 training face
- get descriptors from all pictures
- save descriptors in files
- return trained knn with descriptors
'''
def approach6TrainingInit(size, steps, resize):
    #labels
    # A = 0
    # B = 1
    # C = 2
    # V = 3
    # Points = 4
    # Five = 5
    labels_dictionary = { "A":0, "B":1, "C":2, "V":3, "Point":4, "Five":5}
    # process images at fixed size (50,50)
    imlist = []
    descriptors = []
    labels = []
    for (_, _, filenames) in os.walk('HandGestureDB/train/'):
        imlist = filenames
    #print imlist
    for filename in imlist:
        featfile = str(filename[:-4])
        tmp = approach6GetDescriptors(filename,featfile, size, steps, resize=resize)
        descriptors.append(tmp.flatten())
        tmp_name = featfile.split('-')[0]
        labels.append(labels_dictionary[tmp_name])
    descriptors = np.array(descriptors, np.float32)
    #print descriptors
    labels = np.array(labels)
    knn = cv2.KNearest()
    knn.train(descriptors, labels)
    return knn

'''
approach 6 training face
- read descriptors from files
- return trained knn with descriptors
'''
def approach6TrainingFromFiles(size, steps, resize):
    #labels
    # A = 0
    # B = 1
    # C = 2
    # V = 3
    # Points = 4
    # Five = 5
    labels_dictionary = { "A":0, "B":1, "C":2, "V":3, "Point":4, "Five":5}

    # process images at fixed size (50,50)
    imlist = []
    descriptors = []
    labels = []
    for (_, _, filenames) in os.walk('HandGestureDB/descriptors/SIFT/'):
        imlist = filenames
    #print imlist
    for filename in imlist:
        tmp = np.load('HandGestureDB/descriptors/SIFT/'+filename)
        descriptors.append(tmp.flatten())
        featfile = str(filename[:-4])
        #tmp = approach6(filename,featfile, size, steps, resize=resize)
        #descriptors.append(tmp)
        tmp_name = featfile.split('-')[0]
        labels.append(labels_dictionary[tmp_name])
    
    descriptors = np.array(descriptors, np.float32)
    labels = np.array(labels)
    knn = cv2.KNearest()
    knn.train(descriptors, labels)
    return knn

'''
- classify image with knn
- return: ret, results, neighbours, dist
'''
def approach6Classify(knn, imageName, descriptorSize, descriptorSteps, imgSize):
    imgGray = cv2.cvtColor(cv2.imread('HandGestureDB/test/'+str(imageName)), cv2.COLOR_BGR2GRAY)
    imgGray = cv2.resize(imgGray, imgSize)
    m,n = imgGray.shape
    x,y = np.meshgrid(range(descriptorSteps, m, descriptorSteps), range( descriptorSteps, n, descriptorSteps))
    mask = zip(x.flatten(),y.flatten())
    keyPoints = [cv2.KeyPoint(x, y, descriptorSize) for (x, y) in mask]
    sift = cv2.SIFT()
    _, descriptors = sift.compute(imgGray, keyPoints)
    descriptors = np.matrix(descriptors.flatten(), np.float32)
    #print("descriptors", descriptors)
    ret, results, neighbours, dist = knn.find_nearest(descriptors, 1)
    #print("ret:", ret)
    #print("result:", results)
    #print("neighbours:", neighbours)
    #print("distance:", dist)
    return int(ret), results, neighbours, dist
    

'''
--------------------------------------------------------------------------
DENSE SIFT + SVM
- get descriptors from training set
- get labels from training set
- train and save SVM 
- return SVM
'''
def approach7TrainingInit():
    #labels
    # A = 0
    # B = 1
    # C = 2
    # V = 3
    # Points = 4
    # Five = 5
    labels_dictionary = { "A":0, "B":1, "C":2, "V":3, "Point":4, "Five":5}
    imlist = []
    descriptors = []
    labels = []
    for (_, _, filenames) in os.walk('HandGestureDB/descriptors/'):
        imlist = filenames
    #print imlist
    for filename in imlist:
        tmp = np.load('HandGestureDB/descriptors/'+filename)
        descriptors.append(tmp.flatten())
        featfile = str(filename[:-4])
        #tmp = approach6(filename,featfile, size, steps, resize=resize)
        #descriptors.append(tmp)
        tmp_name = featfile.split('-')[0]
        labels.append(labels_dictionary[tmp_name])
    descriptors = np.array(descriptors, np.float32)
    labels = np.array(labels)

    svm_params = dict(kernel_type = cv2.SVM_LINEAR, svm_type = cv2.SVM_C_SVC, C=2.67)
    svm = cv2.SVM()
    svm.train(descriptors, labels, params=svm_params)
    svm.save('savedData/svm_data.dat')
    print("SIFT+SVM: SVM_data saved")
    return svm

'''
- load saved SVM data
- return svm
'''
def approach7TrainingFromFile():
    svm = cv2.SVM()
    svm.load('savedData/svm_data.dat')
    return svm 

'''
- classify image with SVM
- return result
'''
def approach7Classify(svm, imageName, descriptorSize, descriptorSteps, imgSize):
    imgGray = cv2.cvtColor(cv2.imread('HandGestureDB/test/'+str(imageName)), cv2.COLOR_BGR2GRAY)
    imgGray = cv2.resize(imgGray, imgSize)
    m,n = imgGray.shape
    x,y = np.meshgrid(range(descriptorSteps, m, descriptorSteps), range( descriptorSteps, n, descriptorSteps))
    mask = zip(x.flatten(),y.flatten())
    keyPoints = [cv2.KeyPoint(x, y, descriptorSize) for (x, y) in mask]
    sift = cv2.SIFT()
    _, descriptors = sift.compute(imgGray, keyPoints)
    descriptors = np.matrix(descriptors.flatten(), np.float32)

    result = svm.predict(descriptors)
    #print("res", result)
    #result = result.flatten()
    return int(result)

'''
--------------------------------------------------------------------------
DENSE SURF + K-Nearest Neighbor
- get descriptors from picture
- return descriptors
'''
def approach8GetDescriptors(imagename,resultname,size=20,steps=10, force_orientation=False,resize=None):
    """ Process an image with densely sampled SURF descriptors
    and save the results in a file. Optional input: size of features,
    steps between locations, forcing computation of descriptor orientation (False means all are oriented upwards), tuple for resizing the image."""
    #im = Image.open(imagename).convert('L') #converts image to grayscale
    imgGray = cv2.cvtColor(cv2.imread('HandGestureDB/train/'+str(imagename)), cv2.COLOR_BGR2GRAY)
    imgGray = cv2.equalizeHist(imgGray)
    if resize!=None:
        imgGray = cv2.resize(imgGray, resize)
    m,n = imgGray.shape
    '''
    will use this when training with different gestures
    if imagename[-3:] != 'pgm': # create a pgm file
        np.save('HandGestureDB/savedData/tmp.pgm', imgGray)
        #im.save('tmp.pgm') 
        imagename = 'tmp.pgm'
    '''
    #
    #change frame to opencv mask
    #
    # create frames and save to temporary file
    #scale = size/3.0
    x,y = np.meshgrid(range(steps,m,steps),range(steps,n,steps))
    mask = zip(x.flatten(),y.flatten())
    keyPoints = [cv2.KeyPoint(x, y, size) for (x, y) in mask]

    surf = cv2.SURF(300)
    #shape = number of KeyPoints*128
    #print mask
    keyPoints, descriptors = surf.compute(imgGray, keyPoints)
    np.save('HandGestureDB/descriptors/SURF/'+resultname, descriptors)
    #imgGray=cv2.drawKeypoints(imgGray,keyPoints,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    #print("Keypoints", keyPoints)
    #print("Descriptors", descriptors)  
    #cv2.imshow('sift_keypoints.jpg',imgGray)
    return descriptors
    #cv2.waitKey(0)

'''
approach 8 training face
- get descriptors from all pictures
- save descriptors in files
- return trained knn with descriptors
'''
def approach8TrainingInit(size, steps, resize):
    #labels
    # A = 0
    # B = 1
    # C = 2
    # V = 3
    # Points = 4
    # Five = 5
    labels_dictionary = { "A":0, "B":1, "C":2, "V":3, "Point":4, "Five":5}
    # process images at fixed size (50,50)
    imlist = []
    descriptors = []
    labels = []
    for (_, _, filenames) in os.walk('HandGestureDB/train/'):
        imlist = filenames
    #print imlist
    for filename in imlist:
        featfile = str(filename[:-4])
        tmp = approach8GetDescriptors(filename,featfile, size, steps, resize=resize)
        descriptors.append(tmp.flatten())
        tmp_name = featfile.split('-')[0]
        labels.append(labels_dictionary[tmp_name])
    descriptors = np.array(descriptors, np.float32)
    #print descriptors
    labels = np.array(labels)
    knn = cv2.KNearest()
    knn.train(descriptors, labels)
    return knn

'''
approach 8 training face
- read descriptors from files
- return trained knn with descriptors
'''
def approach8TrainingFromFiles(size, steps, resize):
    #labels
    # A = 0
    # B = 1
    # C = 2
    # V = 3
    # Points = 4
    # Five = 5
    labels_dictionary = { "A":0, "B":1, "C":2, "V":3, "Point":4, "Five":5}

    # process images at fixed size (50,50)
    imlist = []
    descriptors = []
    labels = []
    for (_, _, filenames) in os.walk('HandGestureDB/descriptors/SURF/'):
        imlist = filenames
    #print imlist
    for filename in imlist:
        tmp = np.load('HandGestureDB/descriptors/SURF/'+filename)
        descriptors.append(tmp.flatten())
        featfile = str(filename[:-4])
        #tmp = approach6(filename,featfile, size, steps, resize=resize)
        #descriptors.append(tmp)
        tmp_name = featfile.split('-')[0]
        labels.append(labels_dictionary[tmp_name])
    
    descriptors = np.array(descriptors, np.float32)
    labels = np.array(labels)
    knn = cv2.KNearest()
    knn.train(descriptors, labels)
    return knn

'''
- classify image with knn
- return: ret, results, neighbours, dist
'''
def approach8Classify(knn, imageName, descriptorSize, descriptorSteps, imgSize):
    imgGray = cv2.cvtColor(cv2.imread('HandGestureDB/test/'+str(imageName)), cv2.COLOR_BGR2GRAY)
    imgGray = cv2.resize(imgGray, imgSize)
    m,n = imgGray.shape
    x,y = np.meshgrid(range(descriptorSteps, m, descriptorSteps), range( descriptorSteps, n, descriptorSteps))
    mask = zip(x.flatten(),y.flatten())
    keyPoints = [cv2.KeyPoint(x, y, descriptorSize) for (x, y) in mask]
    surf = cv2.SURF(300)
    _, descriptors = surf.compute(imgGray, keyPoints)
    descriptors = np.matrix(descriptors.flatten(), np.float32)
    imgGray=cv2.drawKeypoints(imgGray,keyPoints,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    #print("Keypoints", keyPoints)
    #print("Descriptors", descriptors)  
    cv2.imshow('surf_keypoints.jpg',imgGray)

    #print("descriptors", descriptors)
    ret, results, neighbours, dist = knn.find_nearest(descriptors, 1)
    #print("ret:", ret)
    #print("result:", results)
    #print("neighbours:", neighbours)
    #print("distance:", dist)
    return int(ret), results, neighbours, dist
    
