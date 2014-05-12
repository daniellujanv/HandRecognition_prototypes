import cv2
import HandRecognition as recon
from datetime import datetime, timedelta
import Tools as tools

#------------------------- main method -------------------------------------------------

imgSize = (340,420)
imgSize2 = (620, 440)
imgSize3 = (310, 310)
descriptorSize = 50
descriptorSteps = 35
labels_dictionary = ["A", "B", "C", "V", "Point", "Five"] #for approach 6, 7, 8
##labels
## A = 0
## B = 1
## C = 2
## V = 3
## Points = 4
## Five = 5


'''
hand_11 = cv2.resize(cv2.imread('./images/hand_11.jpg'), imgSize)
hand_resize_1 = cv2.resize(cv2.imread('./images/resize_1.jpg'), imgSize2)
hand_resize_2 = cv2.resize(cv2.imread('./images/resize_2.jpg'), imgSize2)
hand_rotate_1 = cv2.resize(cv2.imread('./images/rotate_1.jpg'), imgSize2)
hand_rotate_2 = cv2.resize(cv2.imread('./images/rotate_2.jpg'), imgSize2)
hand_switch_1 = cv2.resize(cv2.imread('./images/switch_1.jpg'), imgSize2)
hand_switch_1 = cv2.resize(cv2.imread('./images/switch_1.jpg'), imgSize2)
hand_12 = cv2.resize(cv2.imread('./images/hand_12.jpg'), imgSize)
hand_21 = cv2.resize(cv2.imread('./images/hand_21.jpg'), imgSize)
hand_22 = cv2.resize(cv2.imread('./images/hand_22.jpg'), imgSize)
#tools.showImages(
#                test = db1
                #approach5=recon.approach5(hand_resize_1),
                #approach51=recon.approach5(hand_resize_2),
                #approach52=recon.approach5(hand_rotate_1)
#                 )

#cv2.waitKey(0)

'''
######### show trained gestures
gestureA = cv2.resize(cv2.imread('HandGestureDB/train/A-uniform01.ppm'), imgSize3)
gestureB = cv2.resize(cv2.imread('HandGestureDB/train/B-uniform01.ppm'), imgSize3)
gestureC = cv2.resize(cv2.imread('HandGestureDB/train/C-uniform01.ppm'), imgSize3)
gestureFive = cv2.resize(cv2.imread('HandGestureDB/train/Five-uniform01.ppm'), imgSize3)
gesturePoint = cv2.resize(cv2.imread('HandGestureDB/train/Point-uniform01.ppm'), imgSize3)
gestureV = cv2.resize(cv2.imread('HandGestureDB/train/V-uniform01.ppm'), imgSize3)
tools.showImages(gestureA=gestureA, gestureB=gestureB, gestureC=gestureC, gestureFive=gestureFive,
                 gesturePoint=gesturePoint, gestureV=gestureV)


newComerName = 'hand_11.jpg'
newcomer = cv2.resize(cv2.imread('HandGestureDB/test/'+newComerName), imgSize3)

############# --------------------------------- 6 ------------------------------ #################
###knn = K-Nearest Neighbors
initial_SIFT = datetime.now()
#knnSIFT = recon.approach6TrainingInit(descriptorSize, descriptorSteps,resize=imgSize3)
knnSIFT = recon.approach6TrainingFromFiles(descriptorSize, descriptorSteps,resize=imgSize3)
result_SIFT, _, _, _ = recon.approach6Classify(knnSIFT, newComerName, descriptorSize, descriptorSteps, imgSize=imgSize3)
time_SIFT = timedelta( microseconds = (datetime.now() - initial_SIFT).microseconds)
print("sift+knn___seconds: ", time_SIFT.total_seconds())
#print("SIFT+KNN: newComer results:", result_KNN)
#print("SIFT+KNN: Label: ", labels_dictionary[result_KNN])

###########----------------------------------- 7 ---------------------------------####################
##svm = suport vector machine
#svm = recon.approach7TrainingInit()
initial_SVM = datetime.now()
svm = recon.approach7TrainingFromFile()
result_SVM = recon.approach7Classify(svm, newComerName, descriptorSize, descriptorSteps, imgSize=imgSize3)
time_SVM = timedelta( microseconds=(datetime.now() - initial_SVM).microseconds)
print("sift+svm___seconds: ", time_SVM.total_seconds() )

#print("SIFT+SVM: Results classify: ", result_SVM)
#print("SIFT+SVM: Label: ", labels_dictionary[result_SVM])

############# --------------------------------- 8 ------------------------------ #################
###knn = K-Nearest Neighbors
initial_SURF = datetime.now()
#knnSURF = recon.approach8TrainingInit(descriptorSize, descriptorSteps,resize=imgSize3)
knnSURF = recon.approach8TrainingFromFiles(descriptorSize, descriptorSteps,resize=imgSize3)
result_SURF, _, _, _ = recon.approach8Classify(knnSURF, newComerName, descriptorSize, descriptorSteps, imgSize=imgSize3)
time_SURF = timedelta(microseconds=(datetime.now() - initial_SURF).microseconds)
print("surf+knn___seconds: ", time_SURF.total_seconds())

cv2.imshow("sift+knn___Gesture: "+labels_dictionary[result_SIFT], newcomer)
cv2.imshow("surf+knn___Gesture: "+labels_dictionary[result_SURF], newcomer)
cv2.imshow("sift+svm___Gesture: "+labels_dictionary[result_SVM], newcomer)

cv2.waitKey(0)

