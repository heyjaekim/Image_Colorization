import cv2 as cv
import numpy as np
import sys, os
sys.path.append(os.pardir)
from skimage.measure import compare_ssim
from collections import Counter
import matplotlib.pyplot as plt
import model

def find_similar_patch(target_patch, find_image, similar_patch_num, patch_size):
    height, width = find_image.shape
    best_similar = []
    patcher = patch_size // 2
    patcher_i = float(patch_size * patch_size)
    for y in range(patcher, height-patcher):
        for x in range(patcher, width-patcher):
            #processing simialrity with mean square error                
            err = np.sum((target_patch.astype("float") - find_image[y-patcher:y+patcher+1, x-patcher:x+patcher+1].astype("float")) ** 2)
            err /= patcher_i
            

            #appending best similar patch
            if(len(best_similar) < similar_patch_num):
                best_similar.append([y,x,err])
            
            #sorting best_similar lsit by similarity
            else:
                if(best_similar[similar_patch_num-1][2] > err):
                    best_similar[similar_patch_num-1] = [y,x,err]
                    best_similar.sort(key = lambda x: x[2])        
    
    return best_similar

def color_to_label(input_color, target_color_list, k):
    result = [0] * k
      
    for i in range(len(target_color_list)):        
        if(np.array_equal(input_color, target_color_list[i])):
            result[i] = 1
            return result

    result[0] = 1
    return result

# hyperparameter
K = 5 #the best K representative colors
patch_size = 3   # it must be odd number
patcher = patch_size // 2 
similar_patch_num = 6
skip_knn = False 

# hyperparameter for improved agent
iters_num = 10000
batch_size = 100 
learning_rate = 0.01
hidden_size = 50



#load image
img = cv.imread('./dataset/beach1.jpg')
height, width, channel = img.shape

##conversion process with a classical color to gray conversion formula
gray_img = np.zeros((height,width,1), np.uint8) 
print('Converting to gray scale...')
for y in range(0, height):
    for x in range(0, width):
        gray_img[y,x] = [img[y,x][0]*0.07 + img[y,x][0]*0.72+ img[y,x][0]*0.21]

#basic agent :  Instead of considering the full range of possible colors, 
# run k-means clustering on the colors present in your training data to determine the best 5 representative colors. 
# We will color the test data in terms of these 5 colors. 
print('Running k-means clustering...')
Z = img.reshape((-1,3))
Z = np.float32(Z)
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
ret,label,center=cv.kmeans(Z,K,None,criteria,10,cv.KMEANS_RANDOM_CENTERS)
center = np.uint8(center)

#Re-color the right half of the image by replacing each pixel’s true color with the representative color it was clustered with. 
res = center[label.flatten()]
res2 = res.reshape((img.shape))

#crop image to half
left_train_img = img[:, 0:int(width/2)]
right_test_img = img[:, int(width/2):]

left_train_gray_img = gray_img[:, 0:int(width/2)]
right_test_gray_img = gray_img[:, int(width/2):]

left_train_img_k = res2[:, 0:int(width/2)]
right_test_img_k = res2[:, int(width/2):]


#padding image
#Added border for image edge processing (beacause of patch processing)
test_i = cv.copyMakeBorder(right_test_gray_img,patcher,patcher,patcher,patcher,cv.BORDER_CONSTANT) 
train_i = cv.copyMakeBorder(left_train_gray_img,patcher,patcher,patcher,patcher,cv.BORDER_CONSTANT) 

test_i_height, test_i_width = test_i.shape
train_i_height, train_i_width = train_i.shape


#The Basic Coloring Agent
if(not skip_knn):
    print('Running the basic agent (knn)...')
    colored_right_img = np.zeros((height,int(width/2),3), np.uint8)

    for y in range(patcher, test_i_height-patcher):
        for x in range(patcher, test_i_width-patcher):

            #For every 3x3 grayscale pixel patch in the test data (right half of the image), 
            #ﬁnd the six most similar 3x3 grayscale pixel patches in the training data (left half of the image). 
            #For each of the selected patches in the training data, take the representative color of the middle pixel.
            best_similar = find_similar_patch(test_i[y-patcher:y+patcher+1, x-patcher:x+patcher+1], train_i, similar_patch_num, patch_size)
            
            #If there is a majority representative color, take that representative color to be the color of the middle pixel in the test data patch.
            #If there is no majority representative color or there is a tie, 
            #break ties based on the selected training data patch that is most similar to the test data patch.        
            temp_colors = []

            #Collect the color of best_similar`s coordinate in representative color image
            for e in best_similar:
                temp_colors.append(res2[e[0]-patcher, e[1]-patcher].tolist())

            #Counting representative color in the list     
            for i in range(len(best_similar)):                      
                best_similar[i].append(temp_colors.count(temp_colors[i]))
            
            #get majority representative color (best_similar list is sorted in order of error rate and number of selected color)        
            best_similar.sort(key = lambda x: x[3], reverse = True)

            #In this way, select a color for the middle pixel of each 3x3 grayscale patch in the test data, 
            #and in doing so generate a coloring of the right half of the image.       
            colored_right_img[y-patcher, x-patcher] = res2[best_similar[0][0]-patcher, best_similar[0][1]-patcher]
        
        print(y, '/', test_i_height-patcher)

    cv.imshow('recolored image with basic agent', colored_right_img)
    cv.waitKey(0)

#The Improved Agent
#Create train_data with the left half of the image
print('Running the improved agent')
print('Processing dataset...')
x_train = []
t_train = []
for y in range(patcher, train_i_height-patcher):
    for x in range(patcher, train_i_width-patcher):
        x_train.append(train_i[y-patcher:y+patcher+1, x-patcher:x+patcher+1].reshape(-1))
        t_train.append(color_to_label(left_train_img_k[y-patcher, x-patcher], center, K))

x_train = np.array(x_train) 
t_train = np.array(t_train)

#Create test_data with the right half of the image
x_test = []
t_test = []
for y in range(patcher, test_i_height-patcher):
    for x in range(patcher, test_i_width-patcher):
        x_test.append(test_i[y-patcher:y+patcher+1, x-patcher:x+patcher+1].reshape(-1))
        t_test.append(color_to_label(right_test_img_k[y-patcher, x-patcher], center, K))

x_test = np.array(x_test)
t_test = np.array(t_test)

#Get 2-layerd neural network model
network = model.TwoLayerNet(input_size=patch_size * patch_size, hidden_size=hidden_size, output_size=K)
train_size = x_train.shape[0]
iter_per_epoch = max(train_size / batch_size, 1)

#start learning
print('Learning start...')
train_loss_list = []
train_acc_list = []
test_acc_list = []
for i in range(iters_num):    
    #get mini batch
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    
    #calculate gradient
    #grad = network.numerical_gradient(x_batch, t_batch)
    grad = network.gradient(x_batch, t_batch)
    
    #update parameter
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]
    
    # write learning history
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)
    
    # calculate accuracy in each epoch
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print("train acc, test acc | " + str(train_acc) + ", " + str(test_acc))

#display model loss per epoch
plt.plot(train_loss_list)
plt.title('model_loss')
plt.ylabel('loss')
plt.xlabel('epoch')
#plt.legend(['train', 'test'], loc = 'upper left')
plt.show()

#colorization with the improved agent
print('coloring right image with the improved agent')
improved_colored_right_img = np.zeros((height,int(width/2),3), np.uint8)
for y in range(patcher, test_i_height-patcher):
    for x in range(patcher, test_i_width-patcher):        
        improved_colored_right_img[y-patcher, x-patcher-1] = center[np.argmax(network.predict(test_i[y-patcher:y+patcher+1, x-patcher:x+patcher+1].reshape(-1)))]

cv.imshow('original image', img)
cv.waitKey(0)
cv.imshow('clustured', res2)
cv.waitKey(0)
cv.imshow('recolored image with improved agent', improved_colored_right_img)
cv.waitKey(0)
