##Writeup Template
###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[boxes]:  ./output_images/boxes.png "boxes"
[dataset]:  ./output_images/dataset.png "dataset"
[features]:  ./output_images/features.png "features"
[windows]:  ./output_images/windows.png "windows"
[video1]: ./project_video_output.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the cell 6 of the IPython notebook `Vehicle Detection.ipynb`).  

For the dataset I used the images provided by Udacity.
I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][dataset]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

I used this parameters:

```python
color_space = 'GRAY' 
orient = 8
pix_per_cell = 16
cell_per_block = 1
hog_channel = 'ALL'
spatial_size = (16, 16)
hist_bins = 16
spatial_feat = False
hist_feat = False
hog_feat = True
```




![alt text][features]

####2. Explain how you settled on your final choice of HOG parameters.

I used gray color space to avoid  color information since cars have a huge variety of colors. I also I used less number of features so it was faster for the classifier and improve the performance.

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a SVM classifier with default parameters (rbf kernel). The training code is in the cell 8 from the notebook.

I used a dataset of 8792 cars and 8968 not cars, which I used the 20% for testing and 80% for training. I got a final accuracy of 98.7%. I used `StandardScaler` to normalize the feature vector, actually this incremented the accuracy around 3%.

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

To find the cars, I used sliding windows search technique to sweep the images. I used 3 sizes and areas of windows, defined in the cell 19 from the notebook

```python
sw_x_limits = [
    [None, None],
    [32, None],
    [412, 1280]
]

sw_y_limits = [
    [400, 640],
    [400, 600],
    [390, 540]
]

sw_window_size = [
    (128, 128),
    (96, 96),
    (80, 80)
]

sw_overlap = [
    (0.5, 0.5),
    (0.5, 0.5),
    (0.5, 0.5)
]
```
In order to improve the performance, I used an overlapping of 50% with great results and a small windows of 80x80 pixels.

In the next picture is shown the different areas and sizes of the windows. Also the areas are covering only the piece of road where we are looking for cars.


![alt text][windows]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?


After applying the clasiffier on the windowed regions, we have a number of hot boxes. Some of these boxes are false positives, they don't contain a car or they are finding the same car. To avoid this, we need to estimate the real car positions and sizes.

We take the overlapping boxes, boxes whose center points are inside a region, and we join them in only one calculating the average box and we use the heat map to calculate its size

The next picture shows the followed procedure:

![alt text][boxes]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video_output.mp4)
In the video I also used the lane detection (see `utils/lane_finder.py`) from previous project.

The result is the lane and vehicles detected.


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

Already explained earliear




---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Using SVM as classifier for detecting car works impressively well but under very controlled conditions. It is very easy to have false positives, even in the video there are a couple of frames where the tracking is lost. Obviously I could use more images for the training but I found the classiffer to be very slowly.

Actually I think this is the main problem of this method, it can not be used on real time. It took 7 minutes to process the whole video. When I tried to increment the number of features (9 orientation, YCrCb color space, 32 pixels per cell) and smaller windows with more overlapping I could have times of hours to process a 2 minutes video.

It would be interesting to try using a neural network and using the different layers to know the position of the cars.

 

