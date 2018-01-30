
**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Can also apply a color transform and append binned color features, as well as histograms of color, to HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/Car_and_nonCar_Images.png
[image2]: ./output_images/HOG_Visualization.png
[image3]: ./output_images/Sliding_Windows_on_Test_Image.png
[image4]: ./output_images/bboxes_of_Cars_on_Test_Image.png
[image5]: ./output_images/bboxes_and_heatmap_on_Test_Image.png
[image6]: ./output_images/Heatmap_on_Test_Image.png
[image7]: ./output_images/Label_Image.png
[image8]: ./output_images/Vehicle_Detection_on_Test_Images.png
[video1]: ./project_video.mp4


### Histogram of Oriented Gradients (HOG)

#### 1. Eextraction of HOG features from the training images.

The code for this step is contained in the second code cell of the IPython notebook (or in lines # through # of the file called `some_file.py`). Function 'get_hog_features' uses hog funtion from  scikit-image package to generate Histogram of Oriented Gradients (HOG) features.

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]

#### 2. How I settled on my final choice of HOG parameters.

I tried various combinations of parameters and trained a classifier to figure out which parameters gave the best results. Finally, I notices that color_spaces 'HSV' and 'YCrCb' with ALL hog_channels gave highest accuracy. Hence, I chose YCrCb colorspace with ALL hog_channels.

#### 3. How I trained a classifier using your selected HOG features (and color features).

I trained a linear SVM using different combinations of hog_channels, color spaces and orientation bins. Before training the classifier I normalized my features using Sklearn's StandardScaler function. The code for the classier is contained in seventh cell of the Ipython Notebook. 

### Sliding Window Search

#### 1. Implemention a sliding window search.  How I decided what scales to search and how much to overlap windows?

I decided to step across my image in a pattern and extract the same features used by the classifier to predict if it's a car or not.I also limited my search to lower half of my image since the road can be found only in the lower half anf this also eliminates false positive detections on tree tops etc. The code for this is in third and fourth cells of the Ipython Notebook.

![alt text][image3]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]

To optimize the performance of my classifier, I used heatmaps on my results. Heatmaps help in combining overlapping detections and in reducing false positives. An example image of overlapping detections and heatmap can be found below.

![alt text][image5]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a frame of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image6]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image7]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image8]


---

### Discussion

To summarize, I used HOG features plus spatially binned color and histograms of color in the feature vector and used a linear SVM classifier to train.

To improve the process, I could used decision trees important feature extraction  method to reduce the number of features and also use different scale to find the windows. I could also use different window sizes to search. 

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

