import pypsa
import numpy as np
import csv
from util import check_load_shedding, check_infeasible_generation, check_ramp



#Coal and gas generators, 5 buses
marginal_costs = [2.0, 3.0]
startup_costs = [8.0, 1.0]
mini_up_time = [2, 1]
p_min_pu = [0.2, 0.2]
capacity = [4.0, 2.0]
up_ramp = [1.0, 1.0]
down_ramp=[1.0, 1.0]

'''#Case 1: Incur load shedding: decrease load a bit
adversarial_load=[2,4,4,2]
forecast_load_set=[2, 5, 5, 2]
load_set=[2, 5, 5, 2]'''

'''#Case 2: Increased costs: adversarial loads are larger, cause extra generators on or more expensive ones on
adversarial_load=[2, 5, 5, 2]
forecast_load_set=[2, 4, 4, 2]
load_set=[2, 4, 4, 2]'''

'''#Case 3: Incur infeasible generation: adversarial loads are larger, cause infeasible generation combination
adversarial_load=[2, 5, 5, 2]
forecast_load_set=[2, 4, 4, 2]
load_set=[2, 4, 4, 2]
p_min_pu = [0.6, 0.9]'''

'''#Case 4: Incur extra generators on or more expensive ones on: ramping load
adversarial_load=[2, 5, 5, 5]
forecast_load_set=[2, 5, 5, 4]
load_set=[2, 5, 5, 4]'''

#Case 5: Incur ramp constrains failure: Flat the original ramp
adversarial_load=[3, 5, 5, 4]
forecast_load_set=[2, 5, 5, 4]
load_set=[2, 5, 5, 4]
up_ramp = [0.1, 0.1]



#Case 1: load_shedding under normal data
nu = pypsa.Network()
nu.set_snapshots(range(4))
nu.add("Bus", "bus")
nu.add("Generator", "coal1", bus="bus",
       marginal_cost=marginal_costs[0],
       committable=True,
       p_min_pu=p_min_pu[0],
       initial_status=0,
       ramp_limit_up=up_ramp[0],
       min_up_time=mini_up_time[0],
       start_up_cost=startup_costs[0],
       p_nom=capacity[0])

nu.add("Generator", "gas1", bus="bus",
       marginal_cost=marginal_costs[1],
       committable=True,
       p_min_pu=p_min_pu[1],
       initial_status=0,
       ramp_limit_up=up_ramp[1],
       min_up_time=mini_up_time[1],
       start_up_cost=startup_costs[1],
       p_nom=capacity[1])

nu.add("Load", "load", bus="bus", p_set=forecast_load_set)


print("################  Starting to solve the normal UC problem! ################")
nu.lopf(nu.snapshots)
print("The generator status", nu.generators_t.status)
print("The load", nu.generators_t.p)
generator_schedule=np.array(nu.generators_t.status, dtype=float)
print("Final costs", nu.objective)
print("SHape", np.shape(generator_schedule))


#Case 2: Unit commitment under adversarial data
nu2 = pypsa.Network()
nu2.set_snapshots(range(4))
nu2.add("Bus", "bus")
nu2.add("Generator", "coal1", bus="bus",
       marginal_cost=marginal_costs[0],
       committable=True,
       p_min_pu=p_min_pu[0],
       initial_status=0,
       ramp_limit_up=up_ramp[0],
       min_up_time=mini_up_time[0],
       start_up_cost=startup_costs[0],
       p_nom=capacity[0])

nu2.add("Generator", "gas1", bus="bus",
       marginal_cost=marginal_costs[1],
       committable=True,
       p_min_pu=p_min_pu[1],
       initial_status=0,
       ramp_limit_up=up_ramp[1],
       min_up_time=mini_up_time[1],
       start_up_cost=startup_costs[1],
       p_nom=capacity[1])

nu2.add("Load", "load", bus="bus", p_set=adversarial_load)

print("################  Starting to solve the adversarial UC problem! ################")
nu2.lopf(nu.snapshots)
print("The generator status", nu2.generators_t.status)
print("The generation", nu2.generators_t.p)
adv_generator_schedule=np.array(nu2.generators_t.status, dtype=float)
print("Final costs", nu2.objective)




#Start to solve ED under normal UC
print("############### Starting to solve the ED under normal UC! ############")
costs=0.0
for t in range(4):
       print("time: ", t)
       network = pypsa.Network()
       #nu.set_snapshots(range(1))
       network.add("Bus", "bus")

       for i in range(2):
            if generator_schedule[t, i] == 1: #only add the generator to the snapshots if it is on
                 print("generator %d is on"%i)
                 network.add("Generator",
                             "{}".format(i),
                             bus="bus",
                             initial_status=0,
                             p_nom=capacity[i],
                             marginal_cost=marginal_costs[i])
       print("Current load", load_set[t])
       network.add("Load", "load", bus="bus", p_set=load_set[t])
       network.lopf()
       #Print out the solution value here! Take a look
       print("current step costs:", network.objective)
       print("The generation", network.generators_t.p)
       costs += network.objective
print("Final costs 2", costs)


#Start to solve ED under adversarial UC
print("############### Starting to solve the ED under adversarial UC! ############")
costs2=0.0
gen=np.zeros((4, 2), dtype=float)
for t in range(4):
       print("time: ", t)
       network = pypsa.Network()
       #nu.set_snapshots(range(1))
       network.add("Bus", "bus")

       total_capacity=0.0

       #Check generation capacity->Load shedding or not: generation too small
       check_loadshedding_result=check_load_shedding(time=t, num_gen=2, adv_generator_schedule=adv_generator_schedule,
                                                     capacity=capacity, load=load_set)
       if check_loadshedding_result==True:
            print("Load shedding at time %d!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"%t)
            continue

       #Check infeasible generation: too many generators
       check_generation = check_infeasible_generation(time=t, load=load_set, min_p=p_min_pu,
                                                      num_gen=2, adv_generator_schedule=adv_generator_schedule,
                                                      capacity=capacity)
       if check_generation==True:
            print("Infeasible at time %d!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"%t)
            continue

       for i in range(2):
            if adv_generator_schedule[t, i] == 1: #only add the generator to the snapshots if it is on
                 network.add("Generator",
                             "{}".format(i),
                             bus="bus",
                             initial_status=0,
                             p_min_pu=p_min_pu[i],
                             p_nom=capacity[i],
                             marginal_cost=marginal_costs[i])
       print("Current load", load_set[t])
       network.add("Load", "load", bus="bus", p_set=load_set[t])
       network.lopf()

       #Print out the solution value here! Take a look
       print("current step costs:", network.objective)
       print("The generation", network.generators_t.p)
       print(np.shape(network.generators_t.p))
       power_val = np.array(network.generators_t.p, dtype=float).reshape(1, -1)
       j=0
       for i in range(2):
           if adv_generator_schedule[t, i]==1:
               gen[t, i] = power_val[0, j]
               j+=1
       costs2 += network.objective

print("Final costs normal", costs)
print("Final costs adv", costs2)
print("Generation values: ", gen)
print("ramp", up_ramp)
print("capacity", capacity)
check_ramp_condition = check_ramp(total_time=4, generations=gen, num_gen=2,
                                up_ramp=np.array(up_ramp), down_ramp=np.array(down_ramp),
                                gen_schedule=adv_generator_schedule, capacity=np.array(capacity))