from numpy import shape
import numpy as np
import numpy as np



def reorganize(X_train, Y_train, seq_length, time_lag):
    # Organize the input and output to feed into RNN model
    x_data = []
    for i in range(len(X_train) - seq_length-time_lag):
        x_new = X_train[i:i + seq_length]
        y_new = Y_train[i:i+seq_length].reshape(-1,1)
        x_new = np.concatenate((x_new, y_new), axis=1)
        x_data.append(x_new)

    # Y_train
    y_data = Y_train[seq_length+time_lag:]
    y_data = y_data.reshape((-1, 1))

    return x_data, y_data


def reorganize2(X_train, Y_train, seq_length, forecast_horizon, forecast_time):
    # Organize the input and output to feed into RNN model
    x_data = []
    y_data=[]
    for i in range(len(X_train) - seq_length-forecast_horizon-forecast_time):
        x_new = X_train[i:i + seq_length]
        y_new = Y_train[i:i + seq_length].reshape(-1,1)
        x_new = np.concatenate((x_new, y_new), axis=1)
        x_data.append(x_new)
        y_new = Y_train[i+forecast_time-1+seq_length:i+forecast_horizon+forecast_time+seq_length-1].reshape(-1, forecast_horizon)
        y_data.append(y_new)

    return x_data, y_data


def reorganize_pred(X_train, Y_train, seq_length, forecast_horizon, forecast_time):
    # Organize the input and output to feed into RNN model
    x_data = []
    y_data=[]
    for i in range(len(X_train) - seq_length-forecast_horizon-forecast_time):
        x_new = X_train[i:i + seq_length]
        y_new = Y_train[i:i+seq_length].reshape(-1,1)
        x_new = np.concatenate((x_new, y_new), axis=1)
        x_data.append(x_new)
        y_new = Y_train[i+forecast_time+seq_length:i+forecast_horizon+forecast_time+seq_length].reshape(-1, forecast_horizon)
        y_data.append(y_new)

    return x_data, y_data




def check_load_shedding(time, num_gen, adv_generator_schedule, capacity, load):
    total_capacity=0.0
    for i in range(num_gen):
        if adv_generator_schedule[time, i] == 1:
            total_capacity += capacity[i]

    if total_capacity < load[time]: #Return true if max capacity is smaller than load
        print("total capacity", total_capacity)
        print("Current load: ", load[time])
        print("current combination: ", adv_generator_schedule[time])
        return True
    else:
        return False


def check_infeasible_generation(time, load, min_p, num_gen, adv_generator_schedule, capacity):
    total_capacity=0.0
    for i in range(num_gen):
        if adv_generator_schedule[time, i] == 1:
            total_capacity += min_p[i]*capacity[i]
    print("total min capacity", total_capacity)

    if total_capacity > load[time]: #Return true if min_capacity is larger than load
        return True
    else:
        return False

def check_ramp(total_time, generations, num_gen, up_ramp, down_ramp, gen_schedule, capacity):
    ramp_val=up_ramp*capacity
    #print(ramp_val)
    for t in range(1, total_time):
        for i in range(num_gen):
            if gen_schedule[t-1, i]==gen_schedule[t, i]:
                if generations[t, i]-generations[t-1, i] > ramp_val[i]:
                    print("Go into up ramps!")
                    print("Time", t)
                    print("Generator", i)
                    print("ramp values", ramp_val[i])
                    print("Real changes: ", np.abs(generations[t, i]-generations[t-1, i]))
                    return True

    ramp_val=down_ramp*capacity
    for t in range(1, total_time):
        for i in range(num_gen):
            if gen_schedule[t-1,i]==gen_schedule[t,i]:
                if generations[t-1, i]-generations[t, i] > ramp_val[i]:
                    print("Go into down ramps!")
                    print("Time", t)
                    print("Generator", i)
                    print("ramp values", ramp_val[i])
                    print("Real changes: ", np.abs(generations[t, i]-generations[t-1, i]))
                    return True
    return False

    return total_time

def cal_startup_cost(total_time, gen_schedule, startup_costs, num_gen):
    total_costs=0.0
    startup_costs=np.array(startup_costs, dtype=float).reshape(-1, 1)
    startup_vec=np.zeros((num_gen, 1), dtype=float)
    for i in range(total_time):
        for j in range(num_gen):
            if gen_schedule[i,j]==1:
                startup_vec[j,0]=1

    print("The startup vectors:", startup_vec)
    for q in range(num_gen):
        total_costs+=startup_vec[q, 0]*startup_costs[q, 0]
    return total_costs

def calculate_mae(pred, orig):
    mae=0.0
    for i in range(len(pred)):
        mae+=np.abs(pred[i]-orig[i])/orig[i]
        #print (np.abs(pred[i]-orig[i])/orig[i])
    mae=mae/len(pred)
    #print("This group's error is: ", mae)
    return mae