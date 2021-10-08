# Preparation
The MNIST dataset contains 70,000 images of handwritten digits, divided into 60,000 training images and 10,000 testing images. We use only a part of images for digit “0” and digit “1” in this question. 

Therefore, we have the following statistics for the given dataset:

Number of samples in the training set:  "0": 5000 ;"1": 5000.

Number of samples in the testing set: "0": 980;   "1": 1135 

We assume that the prior probabilities are the same (P(Y=0) = P(Y=1) =0.5), although you may have noticed that these two digits have different numbers of samples in testing sets.

In the existing code, myID is a 4 digit string and please change this string to your last 4 digit of your own studentID; train0 is your trainset for digit0; train1 is your trainset for digit1; test0 is your testset for digit0; and test1 is your testset for digit1. They are all Numpy Arrays. You can also convert them into python arrays if you like.

Other than the string named 'myID', please DON'T change any existing code and just write your own logic with the existing code.
**Notes** 

# Programming
There are 4 tasks to do:
## Task 1
To extract features from the original trainset in order to convert the original data arrays to 2-Dimentional data points.
You are required to extract the following two features for each image:
- Feature 1: the average brightness of each image(average all pixel brightness values within a whole image array)
- Feature 2: The standard deviation of the brightness of each image (standard deviation of all pixel brightness values within a whole image array)
We assume that these two features are independent, and that each image is drawn from a normal distribution.
 **Notes** 
*1. each image is representd by a 28x28 array, each element in the array is a number from 0 - 255 which represent the grayscale of a pixel*
## Task 2
You need to calculate all the parameters for the two-class naive bayes classifiers respectively, based upon the 2-D data points you generated in Task1. (Totally you should have 8 parameters)
1. Mean of feature1 for digit0
2. Variance of feature1 for digit0
3. Mean of feature2 for digit0
4. Variance of feature2 for digit0
5. Mean of feature1 for digit1
6. Variance of feature1 for digit1
7. Mean of feature2 for digit1
8. Variance of feature2 for digit1
 **Notes** use numpy to calculate the man and variance

## Task 3
Since you get the NB classifiers' parameters from Task2, you need to implement their calculation formula according to their Mathematical Expressions. Then you use your implemented classifiers to classify/predict all the unknown labels of newly coming data points (your test data points converted from your original testset for both digit0 and digit1). Thus, in this task, you need to work with the testset for digit0 and digit1 (2 Numpy Arrays: test0 and test1 mentioned above) and you need to predict all the labels of them.

PS: Remember to first convert your original 2 test data arrays (test0 and test1) into 2-D data points as exactly the same way you did in task1.
**notes**
*Gaussian Naive Bayes,https://www.youtube.com/watch?v=H3EjCKtlVog*
1. 该项目中有两套train data, 其中一套全为0,另一套全为1.在task1中, 已经通过将原数据转换成了2d集而生成了2个feature.
2. 在NB classifier中, 假设feature都是相互独立. 那么它的label数量为2**(d+1), d为feature数量.
3. 利用Task2中求出的mean和variance, 可以得到(即求出数据集的mean和standard derivation(std可以由sqrt(variance)得到) )
    - digit0, feature1的高斯分布
    - digit0, feature2的高斯分布
    - digit1, feature1的高斯分布
    - digit1, feature2的高斯分布
4. 此时可以开始预测了,假设一个测试数据[avg_brightness, std_brightness]
    - 利用高斯分布的公式可以求得该测试数据feature1在“digit0, feature1的高斯分布”的概率p1
    - 利用高斯分布的公式可以求得该测试数据feature2在“digit0, feature2的高斯分布”的概率p2
    - 然后p0总概率 = p(prior) x p1 x p2
    - 重复以上步骤可以得到p1总概率. 
    - 如果p0>p1,那么这个测试数据更可能是digit 0,因此归类到0.
    - 由此可以将整套测试数据预测并归类.
5. 测试的准确率 = 正确归类数据/测试总数
## Task 4
In task3 you successfully predicted the labels for all the test data, now you need to calculate the accuracy of your predictions for testset for both digit0 and digit1 respectively.

**Notes** *s*