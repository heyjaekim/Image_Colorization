import cv2 as cv
import numpy as np
import sys, os
sys.path.append(os.pardir)
from skimage.measure import compare_ssim
from collections import Counter
import matplotlib.pyplot as plt
import model

# hyperparameter
data_list = ['p5','p6','p7','p8','p9']
#data_list = ['p5','p8']
K = 5 #the best K representative colors
patch_size = 3   #it must be odd number
patcher = patch_size // 2 
similar_patch_num = 6
skip_knn = False
skip_display = False
# hyperparameter for improved agent
iters_num = 1000
batch_size = 100
learning_rate = 0.002
hidden_size = 800

#conversion process with a classical color to gray conversion formula
def conversion_grayscale(img, height, width):
    #empty image for grayscaled image
    gray_img = np.zeros((height,width,1), np.uint8) 
    for y in range(0, height):
        for x in range(0, width):
            gray_img[y,x] = [img[y,x][0]*0.07 + img[y,x][0]*0.72+ img[y,x][0]*0.21]
    return gray_img


# run k-means clustering on the colors present in your training data to determine the best K representative colors
def conversion_k_representative_color(img, K):    
    print('Running k-means clustering...')
    Z = img.reshape((-1,3))
    Z = np.float32(Z)
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret,label,center=cv.kmeans(Z,K,None,criteria,10,cv.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))

    return res2, center

def find_similar_patch(target_patch, find_image, similar_patch_num, patch_size):
    height, width = find_image.shape
    best_similar = []
    patcher = patch_size // 2
    patcher_i = float(patch_size * patch_size)
    for y in range(patcher, height-patcher):
        for x in range(patcher, width-patcher):
            #mean square error                
            err = np.sum((target_patch.astype("float") - find_image[y-patcher:y+patcher+1, x-patcher:x+patcher+1].astype("float")) ** 2)
            err /= patcher_i
            if(len(best_similar) < similar_patch_num):
                best_similar.append([y,x,err])
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

#basic agent
def basic_agent_with_knn(patcher, patch_size, similar_patch_num, center, train_i, test_i, res2):
        print("running the basic_agent...")
        res2_height, res2_width, ch =  res2.shape
        output_hegiht, output_width, = test_i.shape
        colored_right_img = np.zeros((res2_height , int(res2_width / 2) ,3), np.uint8)

        for y in range(patcher, output_hegiht-patcher):
            for x in range(patcher, output_width-patcher):
                best_similar = find_similar_patch(test_i[y-patcher:y+patcher+1, x-patcher:x+patcher+1], train_i, similar_patch_num, patch_size)        
                temp_colors = []
                for e in best_similar:
                    temp_colors.append(res2[e[0]-patcher, e[1]-patcher].tolist())       
                for i in range(len(best_similar)):                      
                    best_similar[i].append(temp_colors.count(temp_colors[i]))        
                best_similar.sort(key = lambda x: x[3], reverse = True)       
                colored_right_img[y-patcher-1, x-patcher-1] = res2[best_similar[0][0]-patcher, best_similar[0][1]-patcher]
            
            print(y, '/', output_hegiht-patcher)
        
        cnt = 0
        #get basic agent`s accuracy 
        for y in range(patcher, output_hegiht-patcher):
            for x in range(patcher, output_width-patcher):
                if((colored_right_img[y-patcher-1, x-patcher-1] == res2[y-patcher-1, x-patcher-1]).all()):
                    cnt += 1
        acc = cnt /  (output_hegiht * output_width) * 100
        print('basic agent accuracy :', acc)

        return colored_right_img, acc
    
#The Improved Agent
def improved_agent_with_nn(patcher, patch_size, K, center,
                            iters_num, batch_size, learning_rate, hidden_size,
                            train_i, test_i, left_train_img_k, right_test_img_k, res2):

    test_i_height, test_i_width = test_i.shape
    train_i_height, train_i_width = train_i.shape
    height, width, ch = right_test_img_k.shape

    print("running the improved agent...")

    #Create train_data with the left half of the image
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


    print("data shapes: ", x_train.shape, t_train.shape, x_test.shape, t_test.shape)

    train_size = x_train.shape[0]
    iter_per_epoch = max(train_size / batch_size, 1)

    network = model.TwoLayerNet(input_size=patch_size * patch_size, hidden_size=hidden_size, output_size=K)

    print('Learning start...')

    train_loss_list = []
    train_acc_list = []
    test_acc_list = []

    for i in range(iters_num):
        #print(i,'/', iters_num)
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
        if i % 1000 == 0:
            train_acc = network.accuracy(x_train, t_train)
            test_acc = network.accuracy(x_test, t_test)
            train_acc_list.append(train_acc)
            test_acc_list.append(test_acc)
            print(i, '/', iters_num, '    ', "train acc, test acc | " + str(train_acc) + ", " + str(test_acc))   

    #colorization with improved agent
    improved_colored_right_img = np.zeros((height,width,3), np.uint8)

    for y in range(patcher, test_i_height-patcher):
        for x in range(patcher, test_i_width-patcher):        
            improved_colored_right_img[y-patcher-1, x-patcher-1] = center[np.argmax(network.predict(test_i[y-patcher:y+patcher+1, x-patcher:x+patcher+1].reshape(-1)))]


    #get improved agent's agent
    cnt = 0
    for y in range(patcher, test_i_height-patcher):
            for x in range(patcher, test_i_width-patcher):
                if((improved_colored_right_img[y-patcher-1, x-patcher-1] == res2[y-patcher-1, x-patcher-1]).all()):
                    cnt += 1
    acc = cnt /  (height * width) * 100
    print('improved agent accuracy :', acc)

    return improved_colored_right_img, acc, train_loss_list




if __name__ == '__main__':

    basic_agent_acc = []
    improved_agent_acc = []

    for data_i in range(len(data_list)):
        print('image data :', data_i + 1, '/', len(data_list))
        #load image
        img = cv.imread('performance_dataset\\' + data_list[data_i] + '.jpg')
        height, width, channel = img.shape

        #conversion process with a classical color to gray conversion formula
        gray_img = conversion_grayscale(img, height, width)

        # run k-means clustering on the colors present in your training data to determine the best K representative colors
        res2, center = conversion_k_representative_color(img, K)

        #crop image to half
        left_train_img = img[:, 0:int(width/2)]
        right_test_img = img[:, int(width/2):]

        left_train_gray_img = gray_img[:, 0:int(width/2)]
        right_test_gray_img = gray_img[:, int(width/2):]

        left_train_img_k = res2[:, 0:int(width/2)]
        right_test_img_k = res2[:, int(width/2):]


        #padding image
        test_i = cv.copyMakeBorder(right_test_gray_img,patcher,patcher,patcher,patcher,cv.BORDER_CONSTANT) 
        train_i = cv.copyMakeBorder(left_train_gray_img,patcher,patcher,patcher,patcher,cv.BORDER_CONSTANT) 

        test_i_height, test_i_width = test_i.shape
        train_i_height, train_i_width = train_i.shape

        
        #The Basic Coloring Agent
        if(not skip_knn):
            result_image_with_basic_agent, acc = basic_agent_with_knn(patcher, patch_size, similar_patch_num, center, train_i, test_i, res2)
            basic_agent_acc.append(acc)
        #The Improved Agent   
        result_image_with_improved_agent, acc, train_loss_list = improved_agent_with_nn(patcher, patch_size, K, center,
        iters_num, batch_size, learning_rate, hidden_size,
        train_i, test_i, left_train_img_k, right_test_img_k, res2)
        improved_agent_acc.append(acc)

        if(not skip_display):
            print('original image')
            cv.imshow('original image', img)
            cv.waitKey(0)

            print('clustered image')
            cv.imshow('clustured', res2)
            cv.waitKey(0)
            
            if(not skip_knn):
                print('result : basic agent...')
                cv.imshow('basic agent', result_image_with_basic_agent)
                cv.waitKey(0)

            plt.plot(train_loss_list)
            plt.title('model_loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            #plt.legend(['train', 'test'], loc = 'upper left')
            plt.show()

            print('result : improved agent...')
            cv.imshow('recolored image with improved agent', result_image_with_improved_agent)
            cv.waitKey(0)

        
    if(not skip_knn):

        bar_width = 0.1
        alpha = 0.2
        len_data_list = np.arange(len(data_list))

        p1 = plt.bar(len_data_list, basic_agent_acc, 
                    bar_width,
                    color='b',
                    alpha=alpha,
                    label='Basic agent')

        p2 = plt.bar(len_data_list + bar_width, improved_agent_acc,
                    bar_width, 
                    color='r',
                    alpha=alpha,
                    label='Improved agent')

        plt.title('Accuracy comparison', fontsize=20)
        plt.ylabel('accuracy', fontsize=18)
        plt.xlabel('data', fontsize=18)
        plt.xticks(len_data_list, data_list, fontsize=15)
        plt.legend((p1[0], p2[0]), ('Basic', 'Improved'), fontsize=8, loc='upper left')

        plt.show()



