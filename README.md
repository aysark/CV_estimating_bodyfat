Estimating Body Fat Using Computer Vision
=====================

## Abstract
Using preprocessed images of abdominal and chest areas (cropped and grayed), we were able to derive certain structural and features for muscle detection on the body.  Based on the tests done so far, a ~60% correct estimation rate was achieved.  This was done using template matching and SIFT (scale-invariant feature transform).

## Technical Description
The first step taken was to identify the areas on the body that we want to analyze, this is the pectoral and abdominal regions.  Initially the method used to determine this was using a Cascade Classifier which basically learns given a dataset of positive images and negative images, the trained classifier can learn to classify an image through a series of stages.  It uses all the output from the first stage as additional information for the next stage.  OpenCV2 had a utility, trainclassifier, which was used to train a classifier with various stages.  Various flags were attempted with a positive dataset size of 40 images and a negative dataset of 63 images.  However, the classifier takes atleast 6 hours to be trained, and attempting various other options required more compute power.  Therefore, which these constraints, this method was not successful.
The actual method used was a simplified template matcher, it is a method for searching and finding a template image in a larger image.  Our template image in this case was the average image of the all of our positive images.  Using openCV’s matchTemplate method, this method slides the template over the given image and compares, returning an image where each pixel’s intensity is how much it compares with the given image.

The actual method used was a simplified template matcher, it is a method for searching and finding a template image in a larger image.  Our template image in this case was the average image of the all of our positive images.  Using openCV’s matchTemplate method, this method slides the template over the given image and compares, returning an image where each pixel’s intensity is how much it compares with the given image.

![](http://i.imgur.com/ZeQor0A.png)

![](http://i.imgur.com/KmjscIk.png)

The limitations of this method is the region was not always correct, and this would result in more error in later stages.
Once we have the region of analysis, we then iterate over each positive image and perform a SIFT keypoint matching with our input image.  In this case our positive images are images of abdominal and pectoral regions that are known to be less than 10% bodyfat.  We then find the positive image with the most keypoints matched.  We do this again with our negative images, whereby our negative images are images of abdominal and pectoral regions that are known to be more than 15% bodyfat.  All in all, this will give us the best keypoint matched positive image and negative image.
We then perform a comparison between the positive keypoints matched and negative, and deduce based on which one was larger, that the input image is either less than 15% bodyfat, around 15% or greater than 15% bodyfat.  Below we see our results of 12 test images.

![](http://i.imgur.com/XNAMEiV.png)

Overall, we see that our successful estimation rate is around 50% of the time.  Most of our errors are due to very poor lighting of the input test image or due to an incorrect region of analysis found by our template matcher.  Below we see actualy result for test 1:

![](http://i.imgur.com/HdNUamP.png)
