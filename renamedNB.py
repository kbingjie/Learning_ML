
# coding: utf-8

# In[14]:


import numpy as np
import scipy.io
import math
import geneNewData

# likelihood function, std = sqrt(var)


def likelihood(x, mean, std):
    exponent = math.exp(-((x-mean)**2)/(2*std**2))
    return (1/(math.sqrt(2*math.pi*(std**2))))*exponent


def predictProb(prob_f1, prob_f2, prior_prob):
    return prob_f1*prob_f2*prior_prob


def predict_model(testset, mean0, mean1, var0, var1, prob_zero, prob_one):
    lable_zero = 0
    lable_one = 0
    for e in testset:
        p0_f1 = likelihood(e[0], mean0[0], math.sqrt(var0[0]))
        p0_f2 = likelihood(e[1], mean0[1], math.sqrt(var0[1]))
        p0 = predictProb(prob_zero, p0_f1, p0_f2)
        p1_f1 = likelihood(e[0], mean1[0], math.sqrt(var1[0]))
        p1_f2 = likelihood(e[1], mean1[1], math.sqrt(var1[1]))
        p1 = predictProb(prob_one, p1_f1, p1_f2)
        if p0 > p1:
            lable_zero += 1
        else:
            lable_one += 1
    return [lable_zero, lable_one]


def main():
    myID = '2482'
    geneNewData.geneData(myID)
    Numpyfile0 = scipy.io.loadmat('digit0_stu_train'+myID+'.mat')
    Numpyfile1 = scipy.io.loadmat('digit1_stu_train'+myID+'.mat')
    Numpyfile2 = scipy.io.loadmat('digit0_testset'+'.mat')
    Numpyfile3 = scipy.io.loadmat('digit1_testset'+'.mat')
    # train data is 3 nested list, [image1,image2,image3...], each image has 28*28 pixel.
    # each pixel might have grayscale value
    train0 = Numpyfile0.get('target_img')
    train1 = Numpyfile1.get('target_img')
    test0 = Numpyfile2.get('target_img')
    test1 = Numpyfile3.get('target_img')
    print([len(train0), len(train1), len(test0), len(test1)])
    print('Your trainset and testset are generated successfully!')
# task 1, convert original data array to 2-d data points: avg brightness, standard deviation of brightness、
    twod_train0 = [[np.average(e), np.std(e)] for e in train0]
    twod_train1 = [[np.average(e), np.std(e)] for e in train1]
    twod_test0 = [[np.average(e), np.std(e)] for e in test0]
    twod_test1 = [[np.average(e), np.std(e)] for e in test1]
# task 2, mean, variance
    # mean for digit 0 and 1, first column is mean for feature 1, 2nd is mean for feature2
    mean0, mean1 = np.mean(twod_train0, axis=0), np.mean(twod_train1, axis=0)
#     mean_zero_f1,mean_zero_f2 = mean0[0],mean0[1]
#     mean_one_f1, mean_one_f2 = mean1[0],mean1[1]
    print("means and variance")

    # var for digit 0 and 1, first column is var for feature 1, 2nd is var for feature2
    var0, var1 = np.var(twod_train0, axis=0), np.var(twod_train1, axis=0)
#     var_zero_f1,var_zero_f2 = var0[0],var0[1]
#     var_one_f1,var_one_f2 = var1[0],var1[1]
    print("mean of f1 for 0, var of f1 for 0")
    print(mean0[0], var0[0])
    print("mean of f2 for 0, var of f2 for 0")
    print(mean0[1], var0[1])
    print("mean of f1 for 1, var of f1 for 1")
    print(mean1[0], var1[0])
    print("mean of f2 for 1, var of f2 for 1")
    print(mean1[1], var1[1])

    # task 3, calculate nb classifier predict label
    # calculate prior probability
    prob_zero = len(train0)/(len(train0)+len(train1))
    prob_one = len(train1)/(len(train0)+len(train1))

    first_test = predict_model(
        twod_test0, mean0, mean1, var0, var1, prob_zero, prob_one)
    second_test = predict_model(
        twod_test1, mean0, mean1, var0, var1, prob_zero, prob_one)
    print("lables count for two test set, [lable_zero, lable_one]")
    print(first_test)
    print(second_test)

    accuracy_for_digit0testset = first_test[0]/len(twod_test0)
    accuracy_for_digit1testset = second_test[1]/len(twod_test1)
    print("TASK 4, accuracy_for_digit0testset and accuracy_for_digit1testset")
    print(accuracy_for_digit0testset, accuracy_for_digit1testset)


if __name__ == '__main__':
    main()
