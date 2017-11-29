[![Udacity - Robotics NanoDegree Program](https://s3-us-west-1.amazonaws.com/udacity-robotics/Extra+Images/RoboND_flag.png)](https://www.udacity.com/robotics)
## Deep Learning Project - Follow Me##
[image_0]: ./docs/misc/sim_screenshot.png
![alt text][image_0] 

### <center> <h1>Network architecture</h1> </center>
learning_rate = 0.001
batch_size = 64
num_epochs = 40
steps_per_epoch = 64
validation_steps = 25
workers = 1

Achieves 0.417639112688 final score

Both models in the weights folder achieve satisfactory results.  
These results for the following architecture are in my_amazing_model.h5 <br>


| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 160x160x3 RGB image   							| 
|   Encoder Block L1  	| Seperable Convolution Layer Batch Normalized 32 filters  kernel 3x3 stride 2 RELU activation, output 80x80x32|
| Encoder Block L2	| Seperable Convolution Layer Batch Normalized  64 filters kernel 3x3 stride 2 RELU activation, output  40x40x64	|
| Encoder Block L3     	| Seperable Convolution Layers Batch Normalized 128 filters kernel 3x3 stride 2 RELU activation, output  20x20x128|
| Convolution 1x1	L4    | 1x1 stride 256 filters   kernel 1x1 stride 1, output  20x20x256 |
| Decoder Block L5	| L4 Bilinear Upsampling - Layer Concatination L4 + L2- 2x Separable Convolution Layers Batch Normalized 3x3 stride 1 RELU activation, output  40x40x128|
| Decoder Block L6 | L5 Bilinear Upsampling - Layer Concatination L5 + L1- 2x Separable Convolution Layers Batch Normalized 3x3 stride 1 RELU activation, output  80x80x64|
| Decoder Block L7  | L6 Bilinear Upsampling - Layer Concatination L6 + Input - 2x Separable Convolution Layers Batch Normalized 3x3 stride 1 RELU activation, output  160x160x32    									|
| Convolution Layer x  |Seperable Convolution Layer Batch Normalized 32 filters  kernel 3x3 stride 2 RELU activation, output 160x 160x32 |
| Output layer, Fully Connected Convolution	|Output   160x160x3  |

### Encoder Block
Each encoder block is made up of a single convolutional layer with kernel of 3x3 and same padding. The layer is not a typical Convolutional layer, but a seperable convolutional layer.  Seperable Convolution layers use far less parameters and are more efficent without losing too much information. This is important for the current project because inputs need to be evaluated in real time as the drone supplies images. These layers also add non-linearity to the network throught the use of RELU activation which helps improve speed up traing because the RELU function is computational;y less expensive thatn other activation functions, and adds non-linearity to the network which helps combat overfitting. I left the kernel size and padding as the default sizes so that the output from each layer would be half the LxW of the input. 
#### Batch Normalization
Each seperable convolution layer in the nework is also batch normalized. Meaning that in adition to all the training data being normalized, each batch chosen created will be normalized to itself. This allows for higher learning rates, faster training and helps regualarize by introducing noise that will help alleviate overfitting. 
### 1x1 Convolution
At this point in a typical classifier convolutional neural newtork would be a fully connected which is useful for classifying objects in a picture, but loses spatial information. So for this newtork a 1x1 convolution is used that will retain per pixel location information. This is done by crating a 1x1 seperabel convolution layer with 1x1 kernel, stride of 1. An advantage of using a 1x1 convolution layer is that it can handle pictures of any size, so using the model with different size input pictures is much easier to do. It can also be used like a pooling layer for the depth of a previous convolution layer, effectively reducing depth. 
### Decoder Block
The first step in the decoder block is blinear upsampling. This is the process of taking a smaller layer and expanding it so that it meets the dimensionality of a larger layer.  All values in the new higher dimension layer are interpolated from neighboring pixels that have known values. It is done though linear interpolation along two axis, thus the name bilinear. This process is often used to make digital images larger. Once both input layers to the decoder block are the same dimensionality, they are concatinated, or put together into one layer. The advantage of doing this is that while the encoder layers prgressively get narrower they are more accurate at classifying minute features, if we add the layers output to the decoder then those fine details are added back into the other layer that would otherwise be missing. The result is a much more clear image coming out of the decoder block. After concatination I add two seperable convoution layers. These extra layers add depth to the network, that will require extra epochs to train but will result in higher accuracy and more resistance to overfitting by adding extra layers of nonlinearity.

### Hyper Parameters
My goal was to maximize the batch size so that the ram on my GPU was fully utilized. For the size of this network I could have choses 128 batch size, I prefer to use powers of two, but it was occasionally unstable at this size so I chose to use the next lower pwer of two, 64. 
There are about 4100 images in the training data so dividing that by 64 gets a number close to 64, so I my steps per epoch to 64.
I reduced the validation steps to 25 because It got the same results as higer numbers but didn't take as much time to run. I chose to only use 1 worker because I was training on my GPU and I have only 1. The learning rate I settled on was .001. This took longer to converge than higher learning rates, but  at higher learning rates the performance of the model was very dependant on the  results or the last epoch. Learning rate was my biggest issue for this project. Higher learning rates yeilded decent reults, though not passing, with fewer epochs and simpler network arctitecture, but the results were inconsistent and  were coming up 0.1-0.2 points away from passing. If I lowered the learning rate and increased the epochs, then training would often get stuck in local minima. This lead me to adding a few extra  seperable layers to the architecture. I was able to keep the learning rate lower, but reached convergence (possibly local minima as well)  at a much better validation loss and achieved a passing result. 

### Model Use
This model architecture could be used to follow just about any object provided that the input data is similiar in nature. Meaning that the object would have to be a specific color that is unique in the scene for which a mask could be produced to isolate it. This specific model that is trained on this specific data would not generalize outside following a human. 

### Improvements
I think there could be some improvements in the architecture of my network. I get the general idea of how it each part works together, but ended up adding layers here and there in order to squeeze out that extra bit of accuracy. Overall though I don't think the effort put into improving the network would produce large results. I think the biggest improvements would come from data augmentation and creating a more dynamic image processing pipeline. The network is only as good as the data it is fed. 