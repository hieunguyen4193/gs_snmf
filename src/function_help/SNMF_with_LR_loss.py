import sys
from import_library_and_function import *

# adds a column of ones to the feature matrix X to account for the bias term in logistic regression.
def encode_X_with_ones(K):
    K_encoded = np.c_[np.ones((K.shape[0], 1)), K]
    return K_encoded

# sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# update K ~ W
def update_K(Y, K, X):
    numerator = Y.dot(X.T)
    denominator = K.dot(X).dot(X.T) + 1e-10
    K = K * (numerator / denominator)
    return K

# update weight of LR model
def update_weight(u, K, weight, E_g2, E_delta_2, alpha, epsilon):
    # get number of samples
    n = u.shape[0]
    
    # encode K
    W = encode_X_with_ones(K)
    
    j_list = list(range(n))
    random.shuffle(j_list)
    for j in j_list:
        w_i = W[j, :].reshape(1, -1)
        u_i = u[j].reshape(1, -1)

        # compute gradient
        h = sigmoid(w_i.dot(weight))
        gradient = w_i.T.dot(h - u_i).flatten()

        # accumulate gradient
        E_g2 = alpha * E_g2 + (1 - alpha) * gradient**2

        # compute update
        delta = - (np.sqrt(E_delta_2 + epsilon) / np.sqrt(E_g2 + epsilon)) * gradient

        # accumulate updates
        E_delta_2 = alpha * E_delta_2 + (1 - alpha) * delta**2

        # apply update
        weight += delta

    return weight, E_g2, E_delta_2

# update X ~ H
def update_X(Y, K, X, E_g2, E_delta_2, alpha, epsilon, u, weight, epsStab):
    # calculate Gradient
    gradient = K.T.dot(K.dot(X) - Y)

    # accumulate gradient
    E_g2 = alpha * E_g2 + (1 - alpha) * gradient**2

    # compute update
    delta = - (np.sqrt(E_delta_2 + epsilon) / np.sqrt(E_g2 + epsilon)) * gradient

    # accumulate updates
    E_delta_2 = alpha * E_delta_2 + (1 - alpha) * delta**2
    
    # check loss if update
    old_loss = cost_function(Y, K, X, u, weight, 'use_SL')
    check = True
    count_check = 0
    while(check):
        new_loss = cost_function(Y, K, X + delta, u, weight, 'use_SL')
        if new_loss > old_loss:
            delta /= 2
            count_check += 1
            if count_check > 20:
                check = False
        else:
            check = False

    # apply update
    X += delta
    
    # for improved stability
    X[X <= 0] = epsStab

    return X, E_g2, E_delta_2

# cost function
def cost_function(Y, K, X, u = np.array([]), weight = np.array([]), mode = 'use_SL'):
    # numfer of samples
    n = Y.shape[0]
    
    # encode K
    W = encode_X_with_ones(K)

    # NMF Loss
    loss1 = (1 / 2) * sum((Y - K.dot(X)).flatten()**2)
    
    # classification model loss
    if len(u) == 0 or len(weight) == 0:
        mode = 'ignore_SL'
        loss2 = 0
    else:
        loss2 = (1 / n) * ( sum(np.log(1 + np.exp(W.dot(weight)))) - sum(W.dot(weight) * u) )
    
    # SNMF loss
    if mode == 'ignore_SL':
        loss = loss1
    else:
        loss = loss1 + loss2
    
    return loss, loss1, loss2

# SNMF training
def SNMF(Y, u, rank, iter, tolerance, patience, epsStab, alpha, epsilon, init_mode):
    # initialize SNMF
    if init_mode in ['nndsvd', 'nndsvda', 'nndsvdar']:
        K, X = init_NMF(Y, rank, init_mode)
    else:
        K, X = init_NMF(Y, rank, 'random')

    # encode K
    W = encode_X_with_ones(K)

    # initialize weight of LR model
    weight = np.zeros(W.shape[1])
    
    # accumulators for X
    E_g2_X = np.zeros(X.shape)
    E_delta_X2 = np.zeros(X.shape)
    
    # accumulators for weight
    E_g2_weight = np.zeros(weight.shape)
    E_delta_weight2 = np.zeros(weight.shape)
    
    # lost history
    loss_list = []
    loss_list1 = []
    loss_list2 = []

    # X history
    X_history = []

    # weight history
    weight_history = []

    # early stopping
    best_loss = np.inf
    no_improvement_count = 0

    # Loop iteractions
    for i in range(iter):
        
        # update X
        X, E_g2_X, E_delta_X2 = update_X(Y, K, X, E_g2_X, E_delta_X2, 
                                         alpha, epsilon, 
                                         u, weight, epsStab)

        # save X to cache
        X_history.append(X)
        
        # update K
        K = update_K(Y, K, X)
        
        # update weight
        weight, E_g2_weight, E_delta_weight2 = update_weight(u, K, weight, E_g2_weight, E_delta_weight2, 
                                                                 alpha, epsilon)

        weight_history.append(weight)
        
        # loss function
        loss, loss1, loss2 = cost_function(Y, K, X, u, weight, 'use_SL')
        loss_list.append(loss)
        loss_list1.append(loss1)
        loss_list2.append(loss2)
        if i % 10 == 0:
            print("Iteration {},  loss: {:.6f} [{:.6f}, {:.6f}]".format(i, loss, loss1, loss2), flush=True)

        # early stopping
        if best_loss - loss > tolerance:
            best_loss = loss
            no_improvement_count = 0
            X_history = X_history[-1:]
            weight_history = weight_history[-1:]
        else:
            no_improvement_count += 1

        if no_improvement_count == patience:
            print("Early stopping at iteration {}".format(i), flush=True)
            best_case = - no_improvement_count - 1
            best_i = i - no_improvement_count
            print("Best iteration {},  loss: {:.6f} [{:.6f}, {:.6f}]".format(best_i, loss_list[best_i], loss_list1[best_i], loss_list2[best_i]), flush=True)
            return X_history[best_case], loss_list, loss_list1, loss_list2, weight_history[best_case], best_i
            
    best_case = - no_improvement_count - 1
    best_i = i - no_improvement_count
    print("Best iteration {},  loss: {:.6f} [{:.6f}, {:.6f}]".format(best_i, loss_list[best_i], loss_list1[best_i], loss_list2[best_i]), flush=True)
    return X_history[best_case], loss_list, loss_list1, loss_list2, weight_history[best_case], best_i


# SNMF transforming
def SNMF_transform_sample(Y, X, iter, tolerance, patience, epsStab, alpha, epsilon, u = np.array([]), weight = np.array([])):
    # number of rank
    rank = X.shape[0]
    
    # Initialize K
    K = init_W(Y, rank)
    
    # lost history
    loss_list = []
    loss_list1 = []
    loss_list2 = []

    # K history
    K_history = []
    
    # early stopping
    best_loss = np.inf
    no_improvement_count = 0
    
    # loop iteractions
    for i in range(iter):
        
        # update K
        K = update_K(Y, K, X)

        # save K to cache
        K_history.append(K)

        # loss function
        if len(u) == 0 or len(weight) == 0:
            loss, loss1, loss2 = cost_function(Y, K, X)
        else:
            loss, loss1, loss2 = cost_function(Y, K, X, u, weight, 'ignore_SL')

        loss_list.append(loss)
        loss_list1.append(loss1)
        loss_list2.append(loss2)

        # early stopping
        if best_loss - loss > tolerance:
            best_loss = loss
            no_improvement_count = 0
            K_history = K_history[-1:]
        else:
            no_improvement_count += 1

        if no_improvement_count >= patience:
            print("Early stopping at iteration {}".format(i), flush=True)
            best_case = - no_improvement_count - 1
            best_i = i - no_improvement_count
            print("Best iteration {},  loss: {:.6f} [{:.6f}, {:.6f}]".format(best_i, loss_list[best_i], loss_list1[best_i], loss_list2[best_i]), flush=True)
            return K_history[best_case], loss_list, loss_list1, loss_list2, best_i
        
    best_case = - no_improvement_count - 1
    best_i = i - no_improvement_count
    print("Best iteration {},  loss: {:.6f} [{:.6f}, {:.6f}]".format(best_i, loss_list[best_i], loss_list1[best_i], loss_list2[best_i]), flush=True)
    return K_history[best_case], loss_list, loss_list1, loss_list2, best_i