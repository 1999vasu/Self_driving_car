# Self_driving_car
The problem seems to be quite complex, and that is truth as well. This project shows solution just to a part of this problem. We 
will be feeding sequences of images from dashcam and this model will predict the steering angle in degrees.

# Dataset Link
The link to the dataset is [here.](https://drive.google.com/file/d/0B-KJCaaF7elleG1RbzVPZWV4Tlk/view) The datset contains around
45k images and corresponding steering angles. Its around 2.2 GB and you will be able to train a small model on your pc itself.

# Model insights
1. The steering angle was in degrees and it was converted into radians first.
2. The model was designed with basic CNN, Flatten, Dense and Dropout Layers.
3. The activation used in inner layers was relu
4. The activation used in output layer was tanh. (I tried for Linear Activation as well but tanh produced someway better results)
5. Dropout layers were added for regularization and to prevent overfitting.
6. The model predicted the value of steering angle in radians so later it was converted back to degrees.

# Loss function
Loss function used is general:  mse_loss + (l2_norm_constant * l2_norm_of_weights_and_biases)

l2_norm_constant is hyper_parameter.
I tried for 0.1, 0.01, 0.001 and 0.001. 0.001 produces better results

# Training 
I trained my model on google colab. This is the most critical part. You need to be very patient while training your model.
It took me around 15-16 hours of training while I was trying for different hyper_parameters.

# Testing
run.py containse code for testing the model and its performance. This output video is on testing data (not on training data) which
can be verified from code itself.
