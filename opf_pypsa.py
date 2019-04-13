import pypsa
import numpy as np
import csv
from util import *
import matplotlib.pyplot as plt

with open('results/down_bb.csv', 'r') as csvfile:  # good dataset/data2.csv
    rows = [row for row in csv.reader(csvfile.read().splitlines())]
rows=np.array(rows, dtype=float) #column 0 for ground truth, 1 for adv, 2 for forecasts
#rows=rows[50*24:,:]
print("Dataset shape", np.shape(rows))
print(np.max(rows, axis=0))

'''with open('results/up_bb_adv_load.csv', 'r') as csvfile:  
    reader = csv.reader(csvfile)
    rows2 = [row for row in reader]
rows2 = np.array(rows2, dtype=float)
rows2 = rows2[50*24:,:]
print("SHape of adv", np.shape(rows2))'''

#Coal and gas generators, 5 buses
marginal_costs=[6.78, 6.75, 7.75, 28.20, 27.78, 28.83]
startup_costs=[4000.0, 8000.0, 3000.0, 2000.0, 2500.0, 2000.0]
shut_down_costs=[30.0, 20.0, 50.0, 10.0, 15.0, 32.0]
mini_up_time=[5, 5, 2, 1, 2, 2]
mini_down_time=[3, 3, 3, 1, 1, 1, 1]
p_min_pu=[0.2, 0.4, 0.4, 0.1, 0.1, 0.1]
#capacity=[3800.0, 3500.0, 4000.0, 2000.0, 1800.0]
capacity=[2500.0, 2200.0, 1600.0, 1500.0, 1600.0, 2500.0]
up_ramp=[0.1, 0.1, 0.2, 0.3, 0.3, 0.3]
down_ramp=[0.15, 0.15, 0.15, 0.3, 0.3, 0.3]

anomaly_adv=np.zeros((132, 3), dtype=float)
anomaly_forecast=np.zeros((132, 3), dtype=float)
day_costs_adv=np.zeros((132, 2), dtype=float)
day_costs_forecast=np.zeros((132, 2), dtype=float)



for days in range(132):
    print("We are solving for day %d", days)
    #Normal UC#########################################################
    nu = pypsa.Network()
    nu.set_snapshots(range(24))
    nu.add("Bus", "bus")
    nu.add("Generator", "coal1", bus="bus",
           marginal_cost=marginal_costs[0],
           committable=True,
           p_min_pu=p_min_pu[0],
           initial_status=0,
           ramp_limit_up=up_ramp[0],
           ramp_limit_down=down_ramp[0],
           min_up_time=mini_up_time[0],
           min_down_time=mini_down_time[0],
           start_up_cost=startup_costs[0],
           shut_down_cost=shut_down_costs[0],
           p_nom=capacity[0])

    nu.add("Generator", "coal2", bus="bus",
           marginal_cost=marginal_costs[1],
           committable=True,
           p_min_pu=p_min_pu[1],
           initial_status=0,
           ramp_limit_up=up_ramp[1],
           ramp_limit_down=down_ramp[1],
           min_up_time=mini_up_time[1],
           min_down_time=mini_down_time[1],
           start_up_cost=startup_costs[1],
           shut_down_cost=shut_down_costs[1],
           p_nom=capacity[1])

    nu.add("Generator", "gas1", bus="bus",
           marginal_cost=marginal_costs[2],
           committable=True,
           p_min_pu=p_min_pu[2],
           initial_status=0,
           ramp_limit_up=up_ramp[2],
           ramp_limit_down=down_ramp[2],
           min_up_time=mini_up_time[2],
           min_down_time=mini_down_time[2],
           start_up_cost=startup_costs[2],
           shut_down_cost=shut_down_costs[2],
           p_nom=capacity[2])

    nu.add("Generator", "gas2", bus="bus",
           marginal_cost=marginal_costs[3],
           committable=True,
           p_min_pu=p_min_pu[3],
           initial_status=1,
           ramp_limit_up=up_ramp[3],
           ramp_limit_down=down_ramp[3],
           min_up_time=mini_up_time[3],
           min_down_time=mini_down_time[3],
           start_up_cost=startup_costs[3],
           shut_down_cost=shut_down_costs[3],
           p_nom=capacity[3])

    nu.add("Generator", "gas3", bus="bus",
           marginal_cost=marginal_costs[4],
           committable=True,
           p_min_pu=p_min_pu[4],
           initial_status=1,
           ramp_limit_up=up_ramp[4],
           ramp_limit_down=down_ramp[4],
           min_up_time=mini_up_time[4],
           min_down_time=mini_down_time[4],
           start_up_cost=startup_costs[4],
           shut_down_cost=shut_down_costs[4],
           p_nom=capacity[4])

    nu.add("Generator", "gas4", bus="bus",
           marginal_cost=marginal_costs[5],
           committable=True,
           p_min_pu=p_min_pu[5],
           initial_status=1,
           ramp_limit_up=up_ramp[5],
           ramp_limit_down=down_ramp[5],
           min_up_time=mini_up_time[5],
           min_down_time=mini_down_time[5],
           start_up_cost=startup_costs[5],
           shut_down_cost=shut_down_costs[5],
           p_nom=capacity[5])



    nu.add("Load", "load", bus="bus", p_set=(rows[days*24:(days+1) * 24, 2]*1.03).reshape(-1,))

    print("Starting to solve the problem under forecasted load!")
    nu.lopf(nu.snapshots)
    print(nu.generators_t.status)
    print(nu.generators_t.p)
    generator_schedule = np.array(nu.generators_t.status, dtype=float)
    #print(generator_schedule)
    print("Forecasted load", rows[days*24:(days+1)*24, 2])
    print("Final costs", nu.objective)




    '''print("############### Starting to solve the ED under forecasted UC! ############")
    costs = 0.0
    gen = np.zeros((24, 6), dtype=float)
    for t in range(24):
        print("time: ", t)
        print("Current schedule: ", generator_schedule[t])
        network = pypsa.Network()
        # nu.set_snapshots(range(1))
        network.add("Bus", "bus")

        total_capacity = 0.0

        # Check generation capacity->Load shedding or not: generation too small
        check_loadshedding_result = check_load_shedding(time=t, num_gen=6,
                                                        adv_generator_schedule=generator_schedule,
                                                        capacity=capacity, load=rows[days*24:(days+1)*24, 0])
        print("total load", rows[t, 0])
        if check_loadshedding_result == True:
            print("Load shedding at time %d!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" % t)
            anomaly_forecast[days, 0] = 1
            break

        # Check infeasible generation: too many generators
        check_generation = check_infeasible_generation(time=t, load=rows[days*24:(days+1)*24, 0], min_p=p_min_pu,
                                                       num_gen=6, adv_generator_schedule=generator_schedule,
                                                       capacity=capacity)
        if check_generation == True:
            print("Infeasible at time %d!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" % t)
            anomaly_forecast[days, 2] = 1
            break

        for i in range(6):
            if generator_schedule[t, i] == 1:  # only add the generator to the snapshots if it is on
                network.add("Generator",
                            "{}".format(i),
                            bus="bus",
                            initial_status=0,
                            p_min_pu=p_min_pu[i],
                            p_nom=capacity[i],
                            marginal_cost=marginal_costs[i])
        print("Current load", rows[t, 0])
        network.add("Load", "load", bus="bus", p_set=rows[days*24+t, 0])
        network.lopf()

        # Print out the solution value here! Take a look
        print("current step costs:", network.objective)
        print("The generation", network.generators_t.p)
        print("Generation schedule shape", np.shape(network.generators_t.p))
        power_val = np.array(network.generators_t.p, dtype=float).reshape(1, -1)
        j = 0
        for i in range(6):
            if generator_schedule[t, i] == 1:
                gen[t, i] = power_val[0, j]
                j += 1
        costs += network.objective

    # print("Final costs normal", costs)
    print("Final costs forecasted", costs)
    print("Generation values: ", gen)
    round_start_costs = cal_startup_cost(total_time=24, gen_schedule=generator_schedule, startup_costs=startup_costs, num_gen=6)
    print("Startup costs:", round_start_costs)
    print("total  costs", costs+round_start_costs)
    day_costs_forecast[days, 0]=costs+round_start_costs
    day_costs_forecast[days, 1] = round_start_costs
    # print("ramp", up_ramp)
    # print("capacity", capacity)
    check_ramp_condition = check_ramp(total_time=24, generations=gen, num_gen=6,
                                      up_ramp=np.array(up_ramp), down_ramp=np.array(down_ramp),
                                      gen_schedule=generator_schedule, capacity=np.array(capacity))

    if check_ramp_condition == True:
        print("There is ramping!!!!!")
        anomaly_forecast[days, 1] = 1
    print("This day's forecasted final anomaly profile: ", anomaly_forecast[days])'''












    #Adversarial UC############################################################################
    nu2 = pypsa.Network()
    nu2.set_snapshots(range(24))
    nu2.add("Bus", "bus")
    nu2.add("Generator", "coal1", bus="bus",
           marginal_cost=marginal_costs[0],
           committable=True,
           p_min_pu=p_min_pu[0],
           initial_status=0,
           ramp_limit_up=up_ramp[0],
           ramp_limit_down=down_ramp[0],
           min_up_time=mini_up_time[0],
           min_down_time=mini_down_time[0],
           start_up_cost=startup_costs[0],
           shut_down_cost=shut_down_costs[0],
           p_nom=capacity[0])

    nu2.add("Generator", "coal2", bus="bus",
           marginal_cost=marginal_costs[1],
           committable=True,
           p_min_pu=p_min_pu[1],
           initial_status=0,
           ramp_limit_up=up_ramp[1],
           ramp_limit_down=down_ramp[1],
           min_up_time=mini_up_time[1],
           min_down_time=mini_down_time[1],
           start_up_cost=startup_costs[1],
           shut_down_cost=shut_down_costs[1],
           p_nom=capacity[1])

    nu2.add("Generator", "gas1", bus="bus",
           marginal_cost=marginal_costs[2],
           committable=True,
           p_min_pu=p_min_pu[2],
           initial_status=0,
           ramp_limit_up=up_ramp[2],
           ramp_limit_down=down_ramp[2],
           min_up_time=mini_up_time[2],
           min_down_time=mini_down_time[2],
           start_up_cost=startup_costs[2],
           shut_down_cost=shut_down_costs[2],
           p_nom=capacity[2])

    nu2.add("Generator", "gas2", bus="bus",
           marginal_cost=marginal_costs[3],
           committable=True,
           p_min_pu=p_min_pu[3],
           initial_status=1,
           ramp_limit_up=up_ramp[3],
           ramp_limit_down=down_ramp[3],
           min_up_time=mini_up_time[3],
           min_down_time=mini_down_time[3],
           start_up_cost=startup_costs[3],
           shut_down_cost=shut_down_costs[3],
           p_nom=capacity[3])

    nu2.add("Generator", "gas3", bus="bus",
           marginal_cost=marginal_costs[4],
           committable=True,
           p_min_pu=p_min_pu[4],
           initial_status=1,
           ramp_limit_up=up_ramp[4],
           ramp_limit_down=down_ramp[4],
           min_up_time=mini_up_time[4],
           min_down_time=mini_down_time[4],
           start_up_cost=startup_costs[4],
           shut_down_cost=shut_down_costs[4],
           p_nom=capacity[4])

    nu2.add("Generator", "gas4", bus="bus",
           marginal_cost=marginal_costs[5],
           committable=True,
           p_min_pu=p_min_pu[5],
           initial_status=1,
           ramp_limit_up=up_ramp[5],
           ramp_limit_down=down_ramp[5],
           min_up_time=mini_up_time[5],
           min_down_time=mini_down_time[5],
           start_up_cost=startup_costs[5],
           shut_down_cost=shut_down_costs[5],
           p_nom=capacity[5])

    nu2.add("Load", "load", bus="bus", p_set=(rows[days*24:(days+1) * 24, 1]*1.03).reshape(-1,))

    print("Starting to solve the problem under adversarial load!")
    nu2.lopf(nu2.snapshots)
    print(nu2.generators_t.status)
    print(nu2.generators_t.p)
    generator_schedule = np.array(nu2.generators_t.status, dtype=float)
    #print(generator_schedule)
    print("Adversarial load", rows[days * 24:(days + 1) * 24, 1])
    print("Final costs", nu.objective)


    #plt.plot(rows[days*24:(days+1)*24, 1], 'r', label='Adversarial')
    #plt.plot(rows[days*24:(days+1)*24, 2], 'g', label='Predicted')
    #plt.plot(rows[days*24:(days+1)*24, 0], 'b', label='Original')
    #plt.show()

    adv_generator_schedule = np.array(nu2.generators_t.status, dtype=float)






    print("############### Starting to solve the ED under adversarial UC! ############")
    costs2=0.0
    gen=np.zeros((24, 6), dtype=float)
    for t in range(24):
           print("time: ", t)
           print("Current schedule: ", adv_generator_schedule[t])
           network = pypsa.Network()
           #nu.set_snapshots(range(1))
           network.add("Bus", "bus")

           total_capacity=0.0

           #Check generation capacity->Load shedding or not: generation too small
           check_loadshedding_result=check_load_shedding(time=t, num_gen=6, adv_generator_schedule = adv_generator_schedule,
                                                         capacity=capacity, load=rows[days*24:(days+1)*24, 0])
           print("total load", rows[t, 0])
           if check_loadshedding_result==True:
                print("Load shedding at time %d!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"%t)
                anomaly_adv[days, 0] = 1
                continue
                #break

           #Check infeasible generation: too many generators
           check_generation = check_infeasible_generation(time=t, load=rows[days*24:(days+1)*24, 0], min_p=p_min_pu,
                                                          num_gen=6, adv_generator_schedule=adv_generator_schedule,
                                                          capacity=capacity)
           if check_generation==True:
                print("Infeasible at time %d!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"%t)
                anomaly_adv[days, 2] = 1
                continue
                #break

           for i in range(6):
                if adv_generator_schedule[t, i] == 1: #only add the generator to the snapshots if it is on
                     network.add("Generator",
                                 "{}".format(i),
                                 bus="bus",
                                 initial_status=0,
                                 p_min_pu=p_min_pu[i],
                                 p_nom=capacity[i],
                                 marginal_cost=marginal_costs[i])
           print("Current load", rows[t,0])
           network.add("Load", "load", bus="bus", p_set=rows[days*24+t, 0])
           network.lopf()

           #Print out the solution value here! Take a look
           print("current step costs:", network.objective)
           print("The generation", network.generators_t.p)
           print("Generation schedule shape", np.shape(network.generators_t.p))
           power_val = np.array(network.generators_t.p, dtype=float).reshape(1, -1)
           j = 0
           for i in range(6):
               if adv_generator_schedule[t, i]==1:
                   gen[t, i] = power_val[0, j]
                   j+=1
           costs2 += network.objective

    #print("Final costs normal", costs)
    print("Final costs adv", costs2)
    print("Generation values: ", gen)
    round_start_costs = cal_startup_cost(total_time=24, gen_schedule=adv_generator_schedule, startup_costs=startup_costs, num_gen=6)
    print("Startup costs:", round_start_costs)
    print("total adv costs", costs2+round_start_costs)
    day_costs_adv[days, 0] = costs2+round_start_costs
    day_costs_adv[days, 1] = round_start_costs
    #print("ramp", up_ramp)
    #print("capacity", capacity)
    check_ramp_condition = check_ramp(total_time=24, generations=gen, num_gen=6,
                                    up_ramp=np.array(up_ramp), down_ramp=np.array(down_ramp),
                                    gen_schedule=adv_generator_schedule, capacity=np.array(capacity))

    if check_ramp_condition == True:
        print("There is ramping!!!!!")
        anomaly_adv[days, 1] = 1
    print("This day's adversarial final anomaly profile: ", anomaly_adv[days])

    with open('day_costs_forecast.csv', 'wb') as f:
        writer = csv.writer(f)
        writer.writerows(day_costs_forecast)

    with open('day_costs_adv.csv', 'wb') as f:
        writer = csv.writer(f)
        writer.writerows(day_costs_adv)

    with open('anomaly_forecast.csv', 'wb') as f:
        writer = csv.writer(f)
        writer.writerows(anomaly_forecast)

    with open('anomaly_adv.csv', 'wb') as f:
        writer = csv.writer(f)
        writer.writerows(anomaly_adv)

