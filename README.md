# Emotion-recognition-on-image-project

1. Libraries to install
- pip install flask
- pip install pybase64
- pip install Pillow
- pip install uuid
- pip install torch
- pip install numpy
- pip install torchvision

2. After running the project to see the result follow this localhost:
-  http://127.0.0.1:5000

## Neural network structure

- Input Data: The CNN takes images or video frames containing human faces as input. Each image is typically represented as a matrix of pixel values.

- Convolutional Layers: The CNN starts with one or more convolutional layers. These layers consist of a set of learnable filters or kernels. Each filter is small and moves across the input image, performing a mathematical operation known as convolution. This operation extracts local features such as edges, corners, and textures.

- Activation Function: After each convolutional operation, an activation function (such as ReLU - Rectified Linear Unit) is applied element-wise to introduce non-linearity and capture complex patterns in the data.

- Pooling Layers: In between convolutional layers, pooling layers are often used to downsample the spatial dimensions of the data. Common pooling techniques include max pooling or average pooling. Pooling helps reduce the spatial size of the feature maps, making subsequent layers more computationally efficient and robust to variations in the input.

- Fully Connected Layers: Once the feature extraction stages are complete, the output is flattened into a vector and fed into one or more fully connected layers. These layers are similar to traditional neural networks, where each neuron is connected to every neuron in the previous layer. Fully connected layers learn global patterns and relationships among the features extracted by the convolutional layers.

- Output Layer: The final fully connected layer is followed by an output layer, typically using softmax activation for multi-class classification. The output layer contains neurons corresponding to different expression classes (e.g., happy, sad, angry). The softmax activation function ensures that the predicted probabilities across all classes sum up to one.





