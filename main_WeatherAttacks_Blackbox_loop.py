import os
import sys
import math
import itertools
import datetime as dt
import time as t
import pandas as pd
import matplotlib.pyplot as plt
import keras
import scipy.stats as stats
from statsmodels.tsa import stattools

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from data import *
import tensorflow as tf
from util import reorganize, reorganize2, calculate_mae
from nn_model import rnn_model, nn_model, svm_model
from keras.optimizers import SGD
import csv
import random

random.seed(11)


def scaled_gradient(x, predictions, target):
    loss = tf.square(predictions - target)
    grad, = tf.gradients(loss, x)
    signed_grad = tf.sign(grad)
    return grad, signed_grad


def check_constraint(x_orig, x_new, bound):
    for i in range(10):
        for j in range(8):
            x_new[:, i, j] = np.clip(x_new[:,i,j],
                                     x_orig[:, i, j] - bound * np.abs(x_orig[:, i, j]),
                                     x_orig[:, i, j] + bound * np.abs(x_orig[:, i, j]))
            '''print('orig', x_orig[:,i,j])
            print('low', x_orig[:,i,j]-bound*np.abs(x_orig[:,i,j]))
            print('upper', x_orig[:,i,j]+bound*np.abs(x_orig[:,i,j]))
            print('new_val', x_new[:,i,j])'''
    return x_new

def bound_val(x_orig, bound):
    x_low=[]
    x_high=[]
    for i in range(np.shape(x_orig)[0]):
        x_low.append(x_orig[i,0,0]-bound*np.abs(x_orig[i, 0, 0]))
        x_high.append(x_orig[i, 0, 0] + bound * np.abs(x_orig[i, 0, 0]))
    x_low=np.array(x_low, dtype=float)
    x_high=np.array(x_high, dtype=float)

    return x_low, x_high

def temp_bound(adv, orig, temp_val):
    for i in range(1, 9):
        for j in range(len(adv)):
            adv[j, i]=np.clip(adv[j,i], orig[j,i]-temp_val, orig[j,i]+temp_val)
    return adv



# Which features from the dataset should be loaded:
features = ['actual', 'calendar', 'weather']

#Model hyperparameters
seq_length=24
batch_size=32
forecast_horizon= 1
forecast_time= 5
epochs = 30


#Program starts here
sess = tf.Session()
keras.backend.set_session(sess)

# Directory for dataset
path = os.path.join(os.path.abspath(''), '../data/fulldataset.csv')
loc_tz = pytz.timezone('Europe/Zurich')
split_date = loc_tz.localize(dt.datetime(2017, 1, 1, 0, 0, 0, 0))

# Load data and prepare for standardization
print("Data Path", path)
df = load_dataset(path=path, modules=features)
print("Datasize shape", df.shape)
df_scaled = df.copy()
df_scaled = df_scaled.dropna()
print("Datasize shape", df.shape)
print(df_scaled.head())
# Get all float type columns and standardize them
floats = [key for key in dict(df_scaled.dtypes) if dict(df_scaled.dtypes)[key] in ['float64']]
scaler = StandardScaler() #MinMaxScaler
scaled_columns = scaler.fit_transform(df_scaled[floats])
df_scaled[floats] = scaled_columns

print(scaler.mean_)
print(scaler.var_)


# Split in train and test dataset
df_train = df_scaled.loc[(df_scaled.index < split_date)].copy()
df_test = df_scaled.loc[df_scaled.index >= split_date].copy()

# Split in features and label data
y_train = df_train['actual'].copy()
X_train = df_train.drop('actual', 1).copy()
y_test = df_test['actual'].copy()
X_test = df_test.drop('actual', 1).copy()

X_train=np.array(X_train, dtype=float)
y_train=np.array(y_train, dtype=float)
X_test=np.array(X_test, dtype=float)
y_test=np.array(y_test, dtype=float)


x_train, Y_train = reorganize2(X_train, y_train, seq_length, forecast_horizon, forecast_time)
print(np.shape(x_train))
print(np.shape(Y_train))
x_test, y_test = reorganize2(X_test, y_test, seq_length, forecast_horizon, forecast_time)
x_train = np.array(x_train, dtype=float)
y_train = np.array(Y_train, dtype=float).reshape(-1, forecast_horizon)
x_test = np.array(x_test, dtype=float)
y_test = np.array(y_test, dtype=float).reshape(-1, forecast_horizon)
feature_dim = x_train.shape[2]


print("Training data shape", np.shape(x_train))
print("Training label shape", np.shape(y_train))
print("Training data shape", np.shape(x_test))
print("Training label shape", np.shape(y_test))


x = tf.placeholder(tf.float32, shape=(None, seq_length, feature_dim))
y = tf.placeholder(tf.float32, shape=(None, forecast_horizon))
target = tf.placeholder(tf.float32, shape=(None, forecast_horizon))

model = rnn_model(seq_length=seq_length, input_dim=feature_dim, output_dim=forecast_horizon)
predictions = model(x)  #Works for nn and rnn model
sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='mean_squared_error', optimizer='adam')
print("Training begins")

model.load_weights('rnn_cleanfivesteps.h5')
model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=epochs, shuffle=True)
#model.save_weights('rnn_cleanfivesteps.h5')
model.save_weights('rnn_substitute.h5')

y_result = model.predict(x_test)
y_real = y_test
#plt.plot(y_result[:300], 'r')
#plt.plot(y_real[:300], 'b')
plt.plot(y_result[:300, 0], 'r')
plt.plot(y_real[:300, 0], 'b')
plt.show()


# Optimization step starts here!

print("Optimization begins")

with sess.as_default():
    for outer_loop in range(1, 11):
        X_new = []
        grad_new = []
        X_train2 = np.copy(x_test)
        print("I am in loop %d!" %outer_loop)
        # Attack parameters
        eps = 0.01 * outer_loop  # Feature value change
        opt_length = len(x_test)
        bound = 0.4
        temp_bound_val = 0.5 * outer_loop
        model.load_weights('rnn_substitute.h5')


        counter = 0
        # Initialize the SGD optimizer
        grad, sign_grad = scaled_gradient(x, predictions, target)
        for q in range(opt_length - seq_length):
            if counter % 100 == 0 and counter > 0:
                print("Optimization Time step" + str(counter))
            '''if (q < 1600):
                random_num = 0
            else:
                random_num = 1'''
            random_num=np.random.randint(2)


            # Y_target = (0 * Y_test[counter:counter + mpc_scope]).reshape(-1, 1)
            Y_target = y_test[counter].reshape(-1, 1)*0.0

            # Define input: x_t, x_{t+1},...,x_{t+pred_scope}
            X_input = X_train2[counter]
            X_input = X_input.reshape(1, seq_length, feature_dim)
            X_new_group = np.copy(X_input)

            for it in range(5): #Outer iterations are for number of gradient steps
                for j in range(8):  #the data coming from api, inner iterations are for each dimension of the data
                    #for k in range(seq_length):
                    gradient_value, grad_sign = sess.run([grad, sign_grad], feed_dict={x: X_new_group,
                                                                                                target: Y_target,
                                                                                                keras.backend.learning_phase(): 0})
                    signed_grad = np.zeros(np.shape(X_input))
                    signed_grad[:, :, j] = grad_sign[:, :, j]

                    if random_num == 0:
                        X_new_group = X_new_group + eps * signed_grad
                    else:
                        X_new_group = X_new_group - eps * signed_grad
            X_new_group = check_constraint(X_input, X_new_group, bound)
            #y_new=model.predict(X_new_group)
            #for time_step in range(1, seq_length+1):
                #X_train2[counter+time_step, -time_step, -1] = y_new

            if X_new == []:
                X_new = X_new_group[0].reshape([1, seq_length, feature_dim])
                #grad_new = gradient_value[0]
            else:
                X_new = np.concatenate((X_new, X_new_group[0].reshape([1, seq_length, feature_dim])), axis=0)
                #grad_new = np.concatenate((grad_new, gradient_value[0]), axis=0)

            counter += 1

        X_new = np.array(X_new, dtype=float) #X_new is adversarial data
        print("Adversarial X shape", np.shape(X_new))
        #model.load_weights('model/rnn_clean.h5')
        model.load_weights('rnn_cleanfivesteps.h5')
        y_adv = model.predict(X_new, batch_size=64)
        y_pred = model.predict(x_test[:opt_length-seq_length], batch_size=32)
        y_orig = y_test[:opt_length-seq_length]

        dime = 0
        #x_low, x_high = bound_val(x_test[0:len(X_new), :, :], 0.2)
        #X_temp = (x_test[0:len(X_new), 0, dime]).reshape(-1, 1)
        #X_temp_new = (X_new[0:len(X_new), 0, dime]).reshape(-1, 1)

        x_temp = x_test[0:len(X_new), 0, :76].reshape(-1, 76)

        for i in range(len(X_new)-seq_length-1):
            for time_step in range(1, seq_length + 1):
                # X_train2[counter+time_step, -time_step, -1] = y_new
                X_new[i+time_step+1, -time_step, -1]=y_adv[i]

        x_temp_new = X_new[0:len(X_new), 0, :76].reshape(-1, 76)

        print("shape of ypred", np.shape(y_pred))
        print ("shape of xtemp", np.shape(x_temp))
        x_pred = np.concatenate((y_pred, x_temp), axis=1)
        x_adversarial = np.concatenate((y_adv, x_temp_new), axis=1)
        x_orig = np.concatenate((y_orig, x_temp), axis=1)

        df_1 = pd.DataFrame(x_pred, columns=df.columns.values)
        pred_data = scaler.inverse_transform(df_1[floats])
        df_2 = pd.DataFrame(x_adversarial, columns=df.columns.values)
        adversarial_data = scaler.inverse_transform(df_2[floats])
        df_3 = pd.DataFrame(x_orig, columns=df.columns.values)
        original_data = scaler.inverse_transform(df_3[floats])

        adversarial_data = np.array(adversarial_data, dtype=float)
        pred_data = np.array(pred_data, dtype=float)
        original_data = np.array(original_data, dtype=float)

        mae_val = calculate_mae(adversarial_data[:, 5], original_data[:, 5])
        print("adversarial MAPE is: %f with noise %f"%(mae_val, eps))
        mae_val = calculate_mae(pred_data[:, 5], original_data[:, 5])

        '''plt.plot(adversarial_data[:, 5], 'r', label='Adversarial')
        plt.plot(pred_data[:, 5], 'g', label='Predicted')
        plt.plot(original_data[:, 5],'b', label='Original')
        plt.ylabel('Load (MW)')
        plt.legend()
        plt.show()'''

        with open('load_attack_noise%.2f_bb4.csv' %eps, 'wb') as f:
            writer = csv.writer(f)
            writer.writerows(adversarial_data)

        adversarial_data = temp_bound(adversarial_data, original_data, temp_bound_val)

        '''plt.plot(pred_data[:, 1], 'g', label="predicted")
        plt.plot(adversarial_data[:, 1], 'r', label="adversarial")
        plt.plot(original_data[:, 1], 'b', label="original")
        # plt.plot(x_low,label="Low value")
        # plt.plot(x_high,label="high value")
        plt.legend()
        plt.ylabel('Temperature (F)')
        plt.show()'''



        with open('load_attack_noise%.2f_bb3.csv' %eps, 'wb') as f:
            writer = csv.writer(f)
            writer.writerows(adversarial_data)




print("Doing for gradient estimation attack!!!")
model.load_weights('rnn_cleanfivesteps.h5')
for outer_loop in range(1, 11):
    X_new = []
    grad_new = []
    X_train2 = np.copy(x_test)
    print("I am in loop %d!" % outer_loop)
    # Attack parameters
    delta = 0.05  # For gradient calculation
    eps = 0.01 * outer_loop  # Feature value change
    opt_length = len(x_test)
    bound = 0.4
    temp_bound_val = 0.5 * outer_loop
    model.load_weights('rnn_substitute.h5')

    counter = 0
    # Initialize the SGD optimizer
    grad, sign_grad = scaled_gradient(x, predictions, target)
    for q in range(opt_length - seq_length):
        if counter % 100 == 0 and counter > 0:
            print("Optimization Time step" + str(counter))
        random_num = np.random.randint(2)

        # Y_target = (0 * Y_test[counter:counter + mpc_scope]).reshape(-1, 1)
        Y_target = y_test[counter].reshape(-1, 1) * 0.0

        # Define input: x_t, x_{t+1},...,x_{t+pred_scope}
        X_input = X_train2[counter]
        X_input = X_input.reshape(1, seq_length, feature_dim)
        X_new_group = np.copy(X_input)

        for it in range(5):  # Outer iterations are for number of gradient steps
            for j in range(8):  # the data coming from api, inner iterations are for each dimension of the data
                # for k in range(seq_length):
                saliency_mat = np.zeros(np.shape(X_input))
                saliency_mat[:, :, j] = 1.0
                X_new_group_plus = np.copy(X_new_group) + delta * saliency_mat
                X_new_group_minus = np.copy(X_new_group) - delta * saliency_mat

                gradient_value = model.predict(X_new_group_plus) - model.predict(X_new_group_minus)
                signed_grad = np.zeros(np.shape(X_input))
                signed_grad[:, :, j] = np.sign(gradient_value)

                if random_num == 0:
                    X_new_group = X_new_group + eps * signed_grad
                else:
                    X_new_group = X_new_group - eps * signed_grad
        X_new_group = check_constraint(X_input, X_new_group, bound)
        # y_new=model.predict(X_new_group)
        # for time_step in range(1, seq_length+1):
        # X_train2[counter+time_step, -time_step, -1] = y_new

        if X_new == []:
            X_new = X_new_group[0].reshape([1, seq_length, feature_dim])
            # grad_new = gradient_value[0]
        else:
            X_new = np.concatenate((X_new, X_new_group[0].reshape([1, seq_length, feature_dim])), axis=0)
            # grad_new = np.concatenate((grad_new, gradient_value[0]), axis=0)

        counter += 1

    X_new = np.array(X_new, dtype=float)  # X_new is adversarial data
    print("Adversarial X shape", np.shape(X_new))
    # model.load_weights('model/rnn_clean.h5')
    y_adv = model.predict(X_new, batch_size=64)
    y_pred = model.predict(x_test[:opt_length - seq_length], batch_size=32)
    y_orig = y_test[:opt_length - seq_length]

    dime = 0
    # x_low, x_high = bound_val(x_test[0:len(X_new), :, :], 0.2)
    # X_temp = (x_test[0:len(X_new), 0, dime]).reshape(-1, 1)
    # X_temp_new = (X_new[0:len(X_new), 0, dime]).reshape(-1, 1)

    x_temp = x_test[0:len(X_new), 0, :76].reshape(-1, 76)

    for i in range(len(X_new) - seq_length - 1):
        for time_step in range(1, seq_length + 1):
            # X_train2[counter+time_step, -time_step, -1] = y_new
            X_new[i + time_step + 1, -time_step, -1] = y_adv[i]

    x_temp_new = X_new[0:len(X_new), 0, :76].reshape(-1, 76)

    print("shape of ypred", np.shape(y_pred))
    print ("shape of xtemp", np.shape(x_temp))
    x_pred = np.concatenate((y_pred, x_temp), axis=1)
    x_adversarial = np.concatenate((y_adv, x_temp_new), axis=1)
    x_orig = np.concatenate((y_orig, x_temp), axis=1)

    df_1 = pd.DataFrame(x_pred, columns=df.columns.values)
    pred_data = scaler.inverse_transform(df_1[floats])
    df_2 = pd.DataFrame(x_adversarial, columns=df.columns.values)
    adversarial_data = scaler.inverse_transform(df_2[floats])
    df_3 = pd.DataFrame(x_orig, columns=df.columns.values)
    original_data = scaler.inverse_transform(df_3[floats])

    adversarial_data = np.array(adversarial_data, dtype=float)
    pred_data = np.array(pred_data, dtype=float)
    original_data = np.array(original_data, dtype=float)

    mae_val = calculate_mae(adversarial_data[:, 5], original_data[:, 5])
    print("adversarial MAPE is: %f with noise %f" % (mae_val, eps))
    mae_val = calculate_mae(pred_data[:, 5], original_data[:, 5])

    '''plt.plot(adversarial_data[:, 5], 'r', label='Adversarial')
    plt.plot(pred_data[:, 5], 'g', label='Predicted')
    plt.plot(original_data[:, 5],'b', label='Original')
    plt.ylabel('Load (MW)')
    plt.legend()
    plt.show()'''

    with open('load_grad_noise%.2f_bb4.csv' % eps, 'wb') as f:
        writer = csv.writer(f)
        writer.writerows(adversarial_data)

    adversarial_data = temp_bound(adversarial_data, original_data, temp_bound_val)

    '''plt.plot(pred_data[:, 1], 'g', label="predicted")
    plt.plot(adversarial_data[:, 1], 'r', label="adversarial")
    plt.plot(original_data[:, 1], 'b', label="original")
    # plt.plot(x_low,label="Low value")
    # plt.plot(x_high,label="high value")
    plt.legend()
    plt.ylabel('Temperature (F)')
    plt.show()'''

    with open('load_grad_noise%.2f_bb3.csv' % eps, 'wb') as f:
        writer = csv.writer(f)
        writer.writerows(adversarial_data)
with open('load_orig.csv', 'wb') as f:
    writer = csv.writer(f)
    writer.writerows(original_data)
