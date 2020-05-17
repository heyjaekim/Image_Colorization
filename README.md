# Colorization project
CS440 Final Project

### Brief Description about this project
1. The purpose of this assignment is to demonstrate and explore some basic techniques in supervised learning and computer vision.

2. Typically, a color image is represented by a matrix of 3-component vectors, where Image[x][y] = (r,g,b) indicates that the pixel at position (x,y) has color (r,g,b) where r represents the level of red, g of green, and b blue respectively, as values between 0 and 255. A classical color to gray conversion formula is given by

(1) Gray(r,g,b) = 0.21r + 0.72g + 0.07b, 

where the resulting value Gray(r,g,b) is between 0 and 255, representing the corresponding shade of gray (from totally black to completely white). Note that converting from color to grayscale is (with some exceptions) losing information. For most shades of gray, there will be many (r,g,b) values that correspond to that same shade. 
However, by training a model on similar images, we can make contextually-informed guesses at what the shades of grey ought to correspond to. In an extreme case, if a program recognized a black and white image as containing a tiger (and had experience with the coloring of tigers), that would give a lot of information about how to color it realistically.

3. For the purpose of this assignment, you are to take a single color image (of reasonable size and interest). By converting this image to black and white, you have useful data capturing the correspondence between color images and black and white images. We will use the left half of the image as training data, and the right half of the image as testing data. You will implement the basic model described below to try to re-color the right half of the black and white image based on the color/grayscale correspondence of the left half, and as usual, try to do something better

### Implemented Models (Basic Agent, Improved Agent)

1. Basic Agent
    - Used k-means clustering and KNN algorithm to find the representing colors from the trained dataset and then match with the testing dataset by using the KNN algorithm.
    - Gray colors will be recolored by comparing the similiarties of the representing colors that are extracted from the trained dataset.

2. Improved Agent
    - Used KNN and 2-layered NN algorithms, and reduced the data loss by using softmax function and back propagation function. 

3. Both Basic Agent and Improved Agent are coded in the main.py

### Performance Comparison

1. The statistical data of loss functions from basic agent and improved agent are implemented, and the data will be indicated in the plot. All the codes are implemented in performance_comparison.py.
