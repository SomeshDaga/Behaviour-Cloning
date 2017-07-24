# **Behavioral Cloning Project** 

## Goals of the Project

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

The given materials included image data for a manual run across the track (provided by Udacity) and python scripts to:
* Autonomously drive the car based on the model created in `model.py`
* Generate images and a video during the autonomous run


[//]: # (Image References)

[center]: ./images/center.jpg "Center"
[left]: ./images/left.jpg "Left"
[right]: ./images/right.jpg "Right"
[cropped]: ./images/cropped.jpg "Cropped"
[flipped]: ./images/flipped.jpg "Flipped"

---
## Files in this Repository/Project

My project includes the following files:
* `model.py` containing the script to create and train the model
* `drive.py` for driving the car in autonomous mode
* `model.h5` containing a trained convolution neural network
* `run.mp4` containing the video of the autonomous run for one lap around the track
* A writeup of the features of this project (read below)

The model defined in the `model.py` in trained on the data by running:
```sh
python model.py
```

This generates a model file (created using Keras) in the `.h5` format. The vehicle is then autonomously run by starting the simulator and running the following:

```sh
python drive.py model.h5
```

## Model Architecture and Training Strategy

### The Model Architecture

The model architecture is detailed below:

| Layer         | Description                                                  |
|---------------|--------------------------------------------------------------|
| Cropping2D    | Crops 15px and 6px from top and bottom of image respectively |
| Lambda        | A normalization layer to scale pixel values to [-1.0, 1.0]   |
| Convolution2D | Filters=6, Kernel_Size=2, Strides=2                          |
| Activation    | tanh                                                         |
| Convolution2D | Filters=6, Kernel_Size=2, Strides=2                          |
| Dropout       | 0.2                                                          |
| Flatten       | Flatten the result of the previous layer                     |
| Dense         | Output of size 100 (Fully-Connected Layer)                   |
| Activation    | tanh                                                         |
| Dropout       | 0.2                                                          |
| Dense         | Output of size 50 (Fully-Connected Layer)                    |
| Activation    | tanh                                                         |
| Dropout       | 0.2                                                          |
| Dense         | Output of size 1 (Fully-Connected Layer)                     |
| Activation    | tanh (The resulting steering angle prediction)               |

Listed below are some reasons for the use of this LeNet architecture:
* An NVIDIA architecture was attempted but took too long to train on a 8GB, 1600 MHz, Intel Core i7 Macbook Pro. Likely needs a GPU
* A Cropping2D layer to remove parts of image which serve no purpose in determining steering angle (e.g sky, trees)
* Convolution Layers with small Kernel Sizes and Strides as it was applied to an image scaled down to 30% of its original size
* `tanh` activation functions used as they have a more varied output than the `elu` or `relu` activation functions. Also good because it bounds steering angle between -1 and 1
* Dropouts to prevent overfitting data to the training dataset

Also, a number of image pre-processing steps and data augmentation techniques were used to improve the performance of the autonomous driving:

1. OpenCV reads images in BGR format while the simulator provides images in RGB format to `drive.py`. Converting color spaces drastically improved performance in the regions with dirt on the road borders.
This was achieved using the `cv2.cvtColor(img, cv2.COLOR_BGR2RGB)` utility.

2. Shrinking images to improve training speed. This helped to trim down the training time to ~7 epochs in 10 minutes, while not impact performance in any significant way.
Achieved using the `cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)` function

3. Using images from the left and right camera in the simulator (with the center camera images). Angles were defined with an offset bias of 0.25 from the center steering angle, to correspond to the left and right images.
This also increased the amount of data used in the training by a factor of 3.

4. Flipping images horizontally (and negating steering angles), to prevent the trained model from establishing left-turn bias (as the track dominantly required the car to turn left). 50% of the images were flipped, 
making use of a uniform random variable, created using numpy i.e. `np.random.uniform()`

5. Data was shuffled using the `sklearn.utils.shuffle()` function to prevent rapidly fluctuating weights across batches of data.

Examples of images are shown below:

![Center Camera][center] ![Left Camera][left] ![Right Camera][right] ![Cropped IMage][cropped] ![Flipped Image][flipped]

The validation data was generated by splitting of 20% of the entire dataset, keeping 80% for the training of the model.

#### Training Parameters

The learning rate for the Adam optimizer was reduced from the default of 1e-3 to 5e-4. This was done because it was observed that the losses seemed to start fluctuating a few epochs in, instead of monotonically decreasing. 
This signalled the possibility of the learning rate being too high such that the weights were not being shifted finely.

The loss function used was the `mse` or mean squared error. This seemed appropriate as this was a regression problem (and not a classification task, where the softmax would be more appropriate)

The number of epochs used was 7. The validation loss was found to decrease across all 7 epochs (with not much change after that, seen during previous runs)

### Results

The results of the autonomous driving were successful with the vehicle not leaving the road during the coverage of the entire track.
This can be see through viewing the video file, `run.mp4`
