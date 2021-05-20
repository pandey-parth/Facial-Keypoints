# Facial-Keypoints
create facial keypoint detector based on CNN regressor

### Load and preprocess data

Script get_data.py unpacks data — images and labelled points. 6000 images are located in images folder and keypoint coordinates are in gt.csv file. Run the cell below to unpack data.

Now you have to read gt.csv file and images from images dir. File gt.csv contains header and ground truth points for every image in images folder. It has 29 columns. First column is a filename and next 28 columns are x and y coordinates for 14 facepoints. We will make following preprocessing:

1. Scale all images to resolution 100×100 pixels.
2. Scale all coordinates to range [−0.5;0.5]

To obtain that, divide all x's by width (or number of columns) of image, and divide all y's by height (or number of rows) of image and subtract 0.5 from all values.
Function load_imgs_and_keypoint should return a tuple of two numpy arrays: imgs of shape (N, 100, 100, 3), where N is the number of images and points of shape (N, 28).

### Simple data augmentation

For better training we will use simple data augmentation — flipping an image and points. Implement function flip_img which flips an image and its' points. Make sure that points are flipped correctly! For instance, points on right eye now should be points on left eye (i.e. you have to mirror coordinates and swap corresponding points on the left and right sides of the face). VIsualize an example of original and flipped image

## Network architecture and training

Now let's define neural network regressor. It will have 28 outputs, 2 numbers per point. The precise architecture is up to you. We recommend to add 2-3 (`Conv2D` + `MaxPooling2D`) pairs, then `Flatten` and 2-3 `Dense layers`. Don't forget about ReLU activations. We also recommend to add `Dropout` to every Dense layer (with p from 0.2 to 0.5) to prevent overfitting.

Time to train! Since we are training a regressor, make sure that you use mean squared error (mse) as loss

![](https://github.com/pandey-parth/Facial-Keypoints/blob/master/example.png)
