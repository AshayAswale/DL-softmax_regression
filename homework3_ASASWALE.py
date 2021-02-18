# from matplotlib import pyplot as plt
import numpy as np
import math
import random 
import copy

no_of_classes = 10


def random_rearrange (X_tr, y_tr, seed):
    np.random.seed(seed)
    np.random.shuffle(X_tr)
    np.random.seed(seed)
    np.random.shuffle(y_tr)


def getYHat(X_tr, y_tr, w, b):
    Z_w = np.matmul(X_tr, w) 
    Z = Z_w + b
    
    exp_z = np.exp(Z)
    exp_z_sums = np.sum(exp_z, axis=1)
    y_hat = (exp_z.T/exp_z_sums).T
    return y_hat


def updateW(X_tr, y_hat, y_tr, w, eps, alpha):
    no_data = X_tr.shape[0]
    f_grad = np.matmul(X_tr.T, (y_hat-y_tr))/no_data + alpha*w/no_data
    w -= eps*f_grad
    return w


def updateB(X_tr, y_hat, y_tr, b, eps):
    f_grad = (y_hat-y_tr)
    f_grad = np.mean(f_grad, axis=0)
    b -= eps*f_grad
    return b
    

def training(X_tr, y_tr, w, b, n_squig, eps, alpha, epochs):
    no_data = X_tr.shape[0]
    for epoch in range(0, epochs):
        print("Epoch:", epoch)
        test_data(X_tr, y_tr, w, b)
        data_remain = True
        n_curr = 0
        n_next = n_squig
        i = 0
        while(data_remain):
            # print(i)
            i+=1
            X_tr_temp = X_tr[n_curr:(min(n_next, no_data))]
            n_curr = n_next
            n_next += n_squig

            data_remain = True if n_next<no_data else False
            y_hat = getYHat(X_tr, y_tr, w, b)
            w = updateW(X_tr, y_hat, y_tr, w, eps, alpha)
            b = updateB(X_tr, y_hat, y_tr, b, eps)
            
    return w,b
        

def double_cross_validation(X_tr, y_tr):
    no_data = X_tr.shape[0]
    no_features = X_tr.shape[1]

    n_squig_set = np.array([68, 128])
    eps_set = np.array([0.01, 0.005])
    alpha_set = np.array([0.01, 0.1, 0.5])
    epochs_set = np.array([5, 10, 30])
    # 16 0.001 0.05 100
    # n_squig_set = np.array([10000, 50000])
    # eps_set = np.array([0.01])
    # alpha_set = np.array([0.01])
    # epochs_set = np.array([5,10])
    

    h_star = [n_squig_set[0], eps_set[0], alpha_set[0], epochs_set[0]]
    A = 0

    for n_squig in n_squig_set:
        for eps in eps_set:
            for alpha in alpha_set:
                for epochs in epochs_set:

                    w = np.random.rand(no_features, no_of_classes)
                    b = np.random.rand(1, no_of_classes)
                    w,b = training(X_tr, y_tr, w, b, n_squig, eps, alpha, epochs)
                    error_perct = test_data(X_tr, y_tr, w, b)
                    
                    print("######## Error:", error_perct, "h_star:", n_squig, eps, alpha, epochs, "########")
                    if(error_perct>A):
                        A = error_perct
                        h_star = [n_squig, eps, alpha, epochs]
                            # print("## h_star updated ##")
    
    print("######################")
    print("BEST HYPERPARAMETERS")
    print(n_squig, eps, alpha, epochs)
    print("######################")
    return h_star[0], h_star[1], h_star[2], h_star[3]
    

def stoch_grad_regression (X_tr, y_tr):

    no_data = X_tr.shape[0]
    no_features = X_tr.shape[1]
    # print("Here")
    # print(no_data)
    # print(no_features)

    # Step 1, random w and b generation

    # Randomizing the data
    randint = (random.randint(1, 99))
    random_rearrange(X_tr, y_tr, randint) #seed can be any random number

    ###############################
    #### Final Training Tuning ####
    ###############################
    value = int (input("Enter 1 for hyperparameter tuning\nEnter 2 for training on the tuned hyperparameters\n"))
    
    if (value == 1):
        print("Tuning Hyperparameters!")
        n_squig, eps, alpha, epochs = double_cross_validation(X_tr, y_tr)
    else:
        print("Training using pretuned hyperparameters")
        n_squig, eps, alpha, epochs = 16, 0.001, 0.05, 100

    print("New Params")
    w = np.random.rand(no_features, no_of_classes)
    b = np.random.rand(1, no_of_classes)
    w,b = training(X_tr, y_tr, w, b, n_squig, eps, alpha, epochs)
    return w,b
    

def test_data(X_te, y_te, w, b):
    y_te_raw = np.argmax(y_te, axis=1)
    y_hat = getYHat(X_te, y_te, w, b)
    
    y_cat = np.argmax(y_hat, axis=1)
    err_mat = y_cat - y_te_raw
    count = np.count_nonzero(err_mat == 0)
    no_of_data = y_te.shape[0]
    perct = count/no_of_data*100
    print("Correctness:", perct, "%")

    no_data = X_te.shape[0]
    err_mat = np.dot(y_te.T, np.log(y_hat))/no_data
    err = -np.mean(err_mat)
    print("CEE error:", err)
    return perct


    
def train_age_regressor ():# train_age_regressor()

    # Load data
    X_tr_raw = (np.load("fashion_mnist_train_images.npy"))
    y_tr_raw = np.load("fashion_mnist_train_labels.npy")
    X_te_raw = (np.load("fashion_mnist_test_images.npy"))
    y_te_raw = np.load("fashion_mnist_test_labels.npy")

    no_data = X_tr_raw.shape[0]

    brightness_value = 256
    X_te = X_te_raw/brightness_value
    X_tr = X_tr_raw/brightness_value
    
    y_tr = np.zeros([X_tr_raw.shape[0], no_of_classes])
    y_tr_raw = (np.atleast_2d(y_tr_raw).T)
    np.put_along_axis(y_tr, y_tr_raw, 1, axis=1)
    
    y_te = np.zeros([X_te_raw.shape[0], no_of_classes])
    y_te_raw = (np.atleast_2d(y_te_raw).T)
    np.put_along_axis(y_te, y_te_raw, 1, axis=1)

    # X_show = (X_tr_raw[40].reshape((28, 28)))
    # plt.imshow(X_show, interpolation='nearest', cmap='gray', vmin=0, vmax=255)
    # plt.show()

    w,b = stoch_grad_regression(X_tr, y_tr)
    testing_age = test_data(X_te, y_te, w, b)
    print("################################")
    print("FINAL TESTING ERROR:", testing_age)
    print("################################")
    

    # Report fMSE cost on the training and testing data (separately)
    # ...


if __name__ == '__main__':
    train_age_regressor()