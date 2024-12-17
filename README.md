**Goal:** Develop a RGB-Depth fusion architecture for semantic segmentation based on a Fully Convolutional Network (FCN).

**Tool:** Keras, Jupyter notebook.

**Data set:** Dataset consists of 366 (per modality) images of forest scenes. It is divided into train (200 images), test (100 images), and validation (60 images) datasets. Change the size of all images into
256*256.
           - Pixel-level annotations (ground truth) are available for 6 semantic classes: Trail, Grass, Vegetation, Obstacle, Sky, and Void.
           - Available modalities: RGB, Depth

![image](https://github.com/user-attachments/assets/54970592-65bf-48cf-b10f-74a508215bf3)

**Tasks:** There is a Jupyter Notebook template (Assignment4_template.ipynb) for the exercise that you need to write your code in the template. In summary, you will do the following tasks in the
template notebook:
- Define a Fully Convolutional Network (FCN) for image segmentation by fusing RGB and
depth images. The network consists of two streams each stream has the following layers:
1. Use the pre-trained ResNet50 on imageNet
2. Two Conv layers with 128 and 256 nodes, respectively. Kernel size (3,3), stride (1,1)
3. On Top of the Conv layers, add a dropout layer with 0.2
4. Concatenate two streams.
5. Add a transposed convolution layer (Conv2DTranspose) with Kernel size (64,64),
stride (32,32)
6. Add a reshape layer (tf.keras.layers.Reshape) to reshape inputs into the given shape.
7. Add a softmax activation layer
- Compile the model with SGD(learning_rate=0.008, decay=1e-6, momentum=0.9) and loss="categorical_crossentropy"
- Train the model on the “train” dataset and “validation” dataset for epochs =10.
- Evaluate the model on the test dataset.
- Print loss and accuracy of the model for the test dataset.
- Predict semantically segmented images on 5 random examples of the test dataset.

**Model Summary:**

![image](https://github.com/user-attachments/assets/7f96d6cc-5d4c-43fd-b2cd-e14be4f7b4b3)
