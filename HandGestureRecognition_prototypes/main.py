import cv2
import HandRecognition as recon
import Tools as tools

#------------------------- main method -------------------------------------------------

imgSize = (340,420)
imgSize2 = (620, 440)
imgSize3 = (310, 310)
descriptorSize = 50
descriptorSteps = 35
labels_dictionary = ["A", "B", "C", "V", "Point", "Five"] #for approach 6


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

#knn = K-Nearest Neighbors
#knn = recon.approach6TrainingInit(descriptorSize, descriptorSteps,resize=imgSize3)
knn = recon.approach6TrainingFromFiles(descriptorSize, descriptorSteps,resize=imgSize3)
print "done training"

newComerName = 'hand_22.jpg'
newcomer = cv2.resize(cv2.imread('HandGestureDB/test/'+newComerName), imgSize3)

result, _, _, _ = recon.approach6Classify(knn, newComerName, descriptorSize, descriptorSteps, imgSize=imgSize3)
result = int(result)
    #labels
    # A = 0
    # B = 1
    # C = 2
    # V = 3
    # Points = 4
    # Five = 5
print("newComer results:", result)
print("Label: ", labels_dictionary[result])
cv2.imshow("Gesture: "+labels_dictionary[result], newcomer)

cv2.waitKey(0)

