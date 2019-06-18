#May 12th: Currently working on IEEE 118 bus system
#Working on one single day's shedding case
#5 generators: 1,2,3,6,8
#11 loads: 3,4,5,6,7,9,10,11,12,13,14
#20 lines

import pypsa
import numpy as np
import csv
from util import *
import matplotlib.pyplot as plt


#Read in load data: Switzerland case
#Col1: orig pred; Col2: real load; Col3-7: 1f to 5f load variation
with open('../data/data_all.csv', 'r') as csvfile:  # good dataset/data2.csv
    rows = [row for row in csv.reader(csvfile.read().splitlines())]
rows = np.array(rows, dtype=float) #column 0 for ground truth, 1 for adv_up, 2 for forecasts

#print("Maximum load", np.max(rows, axis=0))


with open('Generator.csv', 'r') as csvfile2:
    generator = [row for row in csv.reader(csvfile2.read().splitlines())]
generator = np.array(generator, dtype=float)
with open('load2.csv', 'r') as csvfile3:
    load = [row for row in csv.reader(csvfile3.read().splitlines())]
load = np.array([51,20,
39,39,0,52,19,28,0, 0, 70,
47,34,14,90,25,11,60,45,
18,14,10,7,13,0,0,71,17,24,
0,43,59,23,59,33,31,0,0,
27,66,37,96,18,16,53,28,34,
20,87,17,17,18,23,113,63,84,
12,12,277,78,0,77,0,0,0,39,
28,0,0,66,0,12,6,68,47,68,61,
71,39,130,0,54,20,11,24,21,0,48,
0,163,10,65,12,30,42,38,15,34,
42,37,22,5,23,38,31,43,50,2,
8,39,0,68,6,8,22,184,20,33])


with open('Branch_data.csv', 'r') as csvfile4:
    branch = [row for row in csv.reader(csvfile4.read().splitlines())]
branch = np.array(branch, dtype=float)

#186 lines, 54 generators,
print(np.shape(generator))
print(np.shape(load))
print(np.shape(branch))




#############Network Topology Parameters#######################
n_buses = 118
n_lines = 186
n_gen=54
simulation_hours=24

#Rewrite the line matrix
line_matrix=np.zeros((186,2))
for i in range(n_lines):
    line_matrix[i,0]=np.int(branch[i,0]-1)
    line_matrix[i,1]=np.int(branch[i,1]-1)

line_capacity=np.ones((186,1))
line_capacity = line_capacity * 200 #Tune this parameter for different network configuration!
print("Line matrix shape", np.shape(line_matrix))
#print("Line capacity", line_capacity)

load_ratio=load/(np.float(np.sum(load)))
print(load_ratio)
print(np.shape(load_ratio))






###############Generator parameters##############################
#Coal and gas generators, 5 buses
gen_bus=generator[:,0]
marginal_costs=np.ones((54,1))*7


startup_costs=np.ones((54,1))*1000
shut_down_costs=np.ones((54,1))*30
mini_up_time=np.ones((54,1))*3
mini_down_time=np.ones((54,1))*2
p_min_pu=np.ones((54,1))*0.1
capacity=generator[:,1]*1.5
print("SUM of capacity", np.sum(capacity))
min_gen=np.multiply(capacity, p_min_pu)
up_ramp=np.ones((54,1))*0.2
down_ramp=np.ones((54,1))*0.2








def solve_UC(t, adv_index, temp):
    network = pypsa.Network()
    network.set_snapshots(range(simulation_hours))

    for i in range(n_buses):
        network.add("Bus", "bus {}".format(i))

    for i in range(n_lines):
        network.add("Line", "line {}".format(i),
                    bus0="bus %d" % line_matrix[i, 0],
                    bus1="bus %d" % line_matrix[i, 1],
                    x=0.0001,
                    s_nom=line_capacity[i]
                    )

    for i in range(n_gen):
        network.add("Generator", "gen {}".format(i), bus="bus {}".format(np.int(generator[i,0]-1)),
               committable=True,
               marginal_cost=marginal_costs[i,0],
               p_min_pu=0.1,
               initial_status=0,
               ramp_limit_up=up_ramp[i],
               ramp_limit_down=down_ramp[i],
               min_up_time=mini_up_time[i,0],
               min_down_time=mini_down_time[i,0],
               start_up_cost=startup_costs[i,0],
               shut_down_cost=shut_down_costs[i,0],
               p_nom=capacity[i])


    for i in range(n_buses):
        if adv_index[i] == 1:
            network.add("Load", "load {}".format(i), bus="bus {}".format(i),
                        p_set=(rows[t:t + 24, temp + 1] * 1.03 * load_ratio[i]).reshape(-1, ))
        else:
            network.add("Load", "load {}".format(i), bus="bus {}".format(i),
                        p_set=(rows[t:t + 24, 0] * 1.03 * load_ratio[i]).reshape(-1, ))

    #print(network.buses)
    #print(network.lines)
    #print(network.generators)
    #print(network.loads)
    #print("Power Network Initialization finished")
    network.lopf(network.snapshots)
    generator_val=np.array(network.generators_t.p, dtype=float)
    line_flow=np.array(network.lines_t.p0, dtype=float)[0]
    generator_schedule = np.array(network.generators_t.status, dtype=float)
    print("Current schedule", np.shape(generator_schedule))
    #print("Current dispatch", generator_val)
    #print("Line flow", line_flow)
    #print("Line flow", np.array(network.lines_t.p1, dtype=float)[0])
    return generator_schedule, generator_val, line_flow








def solve_ED(t, adv_index, temp, schedule):
    network = pypsa.Network()
    network.set_snapshots(range(1))

    for i in range(n_buses):
        network.add("Bus", "bus {}".format(i))

    for i in range(n_lines):
        network.add("Line", "line {}".format(i),
                    bus0="bus %d" % line_matrix[i, 0],
                    bus1="bus %d" % line_matrix[i, 1],
                    x=0.0001,
                    s_nom=line_capacity[i]
                    )

    for i in range(n_gen):
        if schedule[i]==1:
            network.add("Generator", "gen {}".format(i), bus="bus {}".format(np.int(generator[i,0]-1)),
               committable=True,
               marginal_cost=marginal_costs[i,0],
               p_min_pu=0.1,
               p_nom=capacity[i])


    for i in range(n_buses):
            network.add("Load", "load {}".format(i), bus="bus {}".format(i),
                        p_set=(rows[t:t + 1, 1] * 1.03 * load_ratio[i]).reshape(-1, ))
            network.add("Generator", "gen_adv {}".format(i), bus="bus {}".format(i),
                        marginal_cost=100000,
                        committable=True,
                        p_min_pu=0,
                        initial_status=1,
                        p_nom=100000)

    #print(network.buses)
    #print(network.lines)
    #print(network.generators)
    #print(network.loads)
    #print("Power Network Initialization finished")
    network.lopf(network.snapshots)
    generator_val=np.array(network.generators_t.p, dtype=float)
    line_flow=np.array(network.lines_t.p0, dtype=float)[0]
    generator_schedule = np.array(network.generators_t.status, dtype=float)
    print("Current schedule", np.shape(generator_schedule))
    #print("Current dispatch", generator_val)
    print("Line flow", np.shape(line_flow))

    print("Generation", generator_val[:10])
    #print("Line flow", np.array(network.lines_t.p1, dtype=float)[0])
    return generator_schedule, generator_val, line_flow






# Case of branch and search
# Working on a pre-defined schedule
attack_seq=np.array([58, 115, 89,79,53,41,14,48,26,77,73,55,39,10,59])
temperature = 5



#Only show one day's example of attack
days=5
adv_vector = np.zeros([118, 1])

'''print("###############################We are solving for day %d###########################" % days)
print("###############Clean Solution")
print("This day's load:", rows[days * 24:days * 24 + 24, 0])

schedule_clean, generation, line_flow = solve_UC(t=days * 24, adv_index=adv_vector, temp=temperature)
print("Schedule clean", schedule_clean)

with open('clean_118.csv', 'wb') as f:
    writer = csv.writer(f)
    writer.writerows(schedule_clean)

with open('clean_118_gen.csv', 'wb') as f:
    writer = csv.writer(f)
    writer.writerows(generation)



attack_num = 5
for i in range(attack_num):
    adv_vector[attack_seq[i],0]=1


print("#######################Solving 5 bus attack#########################")
schedule_adv5, generation, line_flow = solve_UC(t=days * 24, adv_index=adv_vector, temp=temperature)
print("Schedule clean", schedule_adv5)

with open('adv5_118.csv', 'wb') as f:
    writer = csv.writer(f)
    writer.writerows(schedule_adv5)

with open('adv5_118_gen.csv', 'wb') as f:
    writer = csv.writer(f)
    writer.writerows(generation)




print("#######################Solving 10 bus attack#######################")
attack_num = 10
for i in range(attack_num):
    adv_vector[attack_seq[i],0]=1

schedule_adv10, generation, line_flow = solve_UC(t=days * 24, adv_index=adv_vector, temp=temperature)
print("Schedule clean", schedule_adv10)

with open('adv10_118.csv', 'wb') as f:
    writer = csv.writer(f)
    writer.writerows(schedule_adv10)

with open('adv10_118_gen.csv', 'wb') as f:
    writer = csv.writer(f)
    writer.writerows(generation)'''



'''schedule_adv5 = np.array([0,0,0,0,1,1,0,0,0,0,0,1,
                          0,0,0,0,0,0,0,0,1,1,0,0,
                          1,1,0,1,1,1,0,0,0,0,0,0,
                          1,0,0,1,0,0,0,0,1,1,0,0,
                          0,0,1,0,0,0])


print("######################Solving ED using adv5 schedule")
ed_5, generation, line_flow = solve_ED(t=days * 24+12, adv_index=adv_vector, temp=temperature, schedule=schedule_adv5)
with open('ED5_118.csv', 'wb') as f:
    writer = csv.writer(f)
    writer.writerows(ed_5)

with open('ED5_118_lineflow.csv', 'wb') as f:
    writer = csv.writer(f)
    writer.writerows(line_flow.reshape(-1,1))

schedule_adv10 = np.array([0,0,0,0,1,0,0,0,0,0,0,1,
                          0,0,0,0,0,0,0,0,1,1,0,0,
                          1,1,0,1,1,1,0,0,0,0,0,0,
                          0,0,0,1,0,0,0,0,1,1,0,0,
                          0,0,0,0,0,0])

print("######################Solving ED using adv10 schedule")
ed_10, generation, line_flow = solve_ED(t=days * 24+12, adv_index=adv_vector, temp=temperature, schedule=schedule_adv10)
with open('ED10_118.csv', 'wb') as f:
    writer = csv.writer(f)
    writer.writerows(ed_10)
with open('ED10_118_lineflow.csv', 'wb') as f:
    writer = csv.writer(f)
    writer.writerows(line_flow.reshape(-1,1))'''


#Hasn't completely edited
schedule_normal = np.array([0,0,0,0,1,1,0,0,0,0,1,1,
                          0,0,0,0,0,0,0,0,1,1,0,0,
                          1,1,0,1,1,1,0,0,0,0,0,0,
                          1,0,0,1,0,0,0,0,1,1,0,0,
                          0,0,1,0,0,0])

print("######################Solving ED using adv10 schedule")
ed_10, generation, line_flow = solve_ED(t=days * 24+12, adv_index=adv_vector, temp=temperature, schedule=schedule_normal)
with open('ED10_118.csv', 'wb') as f:
    writer = csv.writer(f)
    writer.writerows(ed_10)
with open('ED10_118_lineflow.csv', 'wb') as f:
    writer = csv.writer(f)
    writer.writerows(line_flow.reshape(-1,1))





