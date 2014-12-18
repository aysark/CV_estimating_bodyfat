import numpy as np
import cv2, os
import drawMatches
from matplotlib import pyplot as plt

### Add other test images when needed: 
## cv2.imread('test2/t4.png',0),cv2.imread('test2/t5.png',0), cv2.imread('test2/t6.png',0)
test_images = [cv2.imread('test2/t9.png',0),cv2.imread('test2/t7.png',0),cv2.imread('test2/t3.png',0)]

for test_image in test_images:
	################ Apply Template Matching ################
	template = cv2.imread('pos/avg.png',0)
	w, h = template.shape[::-1]
	res = cv2.matchTemplate(test_image,template,cv2.TM_CCORR)

	min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
	top_left = max_loc
	bottom_right = (top_left[0] + w, top_left[1] + h)

	cv2.rectangle(test_image,top_left, bottom_right, 255, 2)

	plt.subplot(121)
	plt.imshow(res,cmap = 'gray')
	plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
	plt.subplot(122),plt.imshow(test_image,cmap = 'gray')
	plt.title('Detected Region')
	plt.xticks([]), plt.yticks([])
	plt.suptitle('cv2.TM_CCORR')
	plt.show()

	test_image_part = test_image[top_left[1]:top_left[1]+h, top_left[0]:top_left[0]+w]

	################ Key point detection and matching ################
	# Initiate Scale Invarient Feature Transform
	orb = cv2.ORB()
        
	allfiles=os.listdir(os.getcwd()+'\\pos')
	imlist=[filename for filename in allfiles if  filename[-4:] in [".png",".PNG"]]
	N=len(imlist)
	kpmax = 0
	for im in imlist:
	    t = cv2.imread('pos/'+im,0)
	    # find the keypoints and descriptors with SIFT
	    kp1, des1 = orb.detectAndCompute(t,None)
	    kp2, des2 = orb.detectAndCompute(test_image_part,None)
	    # create BFMatcher object
	    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
	    # Match descriptors.
	    matches = bf.match(des1,des2)
	    # Sort them in the order of their distance.
	    matches = sorted(matches, key = lambda x:x.distance)
	    if (kp1 > kpmax):
	        kpmax = kp1
	        lowbf_matches = len(matches)
	        outmax = drawMatches.drawMatches(t, kp1, test_image_part, kp2, matches[:10])
	lowbf_kp = kpmax
	
	cv2.imshow('Matched Features', outmax)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

	allfiles=os.listdir(os.getcwd()+'\\neg2')
	imlist=[filename for filename in allfiles if  filename[-4:] in [".png",".PNG"]]
	N=len(imlist)
	kpmax = 0
	for im in imlist:
	    t = cv2.imread('neg2/'+im,0)
	    kp1, des1 = orb.detectAndCompute(t,None)
	    kp2, des2 = orb.detectAndCompute(test_image_part,None)
	    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
	    matches = bf.match(des1,des2)
	    matches = sorted(matches, key = lambda x:x.distance)
	    if (kp1 > kpmax):
	        kpmax = kp1
	        hibf_matches = len(matches)
	        outmax = drawMatches.drawMatches(t, kp1, test_image_part, kp2, matches[:10])
	hibf_kp = kpmax
	
	cv2.imshow('Matched Features', outmax)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	matches_r = lowbf_matches / hibf_matches
	if (matches_r > 2):
		resultStr = 'Below 15% bodyfat '+ `lowbf_matches` + ' lm / ' + `hibf_matches` +' hm'
	elif (matches_r >= 1):
		resultStr = 'Around 15% bodyfat '+ `lowbf_matches` + ' lm / ' + `hibf_matches` +' hm'
	else:
		resultStr = 'Above to 15% bodyfat '+ `lowbf_matches` + ' lm / ' + `hibf_matches` +' hm'
	plt.imshow(test_image,cmap = 'gray')
	plt.title(resultStr), plt.xticks([]), plt.yticks([])
	plt.suptitle('Training Image Bodyfat Estimate')
	plt.show()