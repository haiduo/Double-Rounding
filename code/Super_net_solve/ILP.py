''''
pip install pulp
'''
import numpy as np 
import pulp

def cal_ILP(avg_bit, bit_list, Hutchinson_trace, logging=None):
    if (bit_list[-1]-bit_list[0]) != len(bit_list)-1:
        min_bit = bit_list[0]
        new_bit_list = list(np.array(bit_list)/min_bit)  # 8,6,4,2 => 4,3,2,1
        lowB, upB = new_bit_list[0], new_bit_list[-1]
    else:
        lowB, upB = bit_list[0], bit_list[-1]
        min_bit = 1

    prob = pulp.LpProblem("Model_bits", pulp.LpMaximize)
    num_variable = len(Hutchinson_trace)
    variable = {}
    for i in range(num_variable):
        variable[f"x{i}"] = pulp.LpVariable(f"x{i}", lowBound=lowB, upBound=upB, cat=pulp.LpInteger)
    prob += pulp.lpSum([variable[f"x{i}"]*Hutchinson_trace[i]*min_bit for i in range(num_variable)])
    prob += pulp.lpSum([variable[f"x{i}"] for i in range(num_variable)]) /num_variable * min_bit == avg_bit
    prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=2))
    # prob.solve(pulp.GLPK_CMD(msg=False, timeLimit=2))
    if pulp.LpStatus[prob.status] != "Optimal":
        return

    result = []
    for i in range(num_variable):
        result.append(int(pulp.value(variable[f"x{i}"])))

    first_optim = list(np.array(result)*min_bit)

    if logging:
        logging.info(f"avg_bits_{avg_bit},{first_optim},{pulp.value(prob.objective):.4f}")
    else:
        print(f"avg_bits_{avg_bit},{first_optim},{pulp.value(prob.objective):.4f}")
    #-------------------------Find other optimal combinations----------------------
    all_candidate_bits = []
    if len(np.unique(first_optim)) == 1:
        if np.unique(first_optim)[0] != 0:
            all_candidate_bits.append(first_optim)
            return all_candidate_bits
        else:
            return all_candidate_bits
    all_candidate_bits.append(first_optim)
    i_max = bit_list.index(max(first_optim))
    for i, cur_bit in enumerate(first_optim):
        for replace_bit in bit_list[:i_max]:
            if replace_bit == cur_bit:
                continue
            else:
                prob += (variable[f"x{i}"]*min_bit == replace_bit)
                prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=2)) #msg=0 Do not print log, 1: print
                # prob.solve(pulp.GLPK_CMD(msg=False, timeLimit=2))
                if pulp.LpStatus[prob.status] == "Optimal":
                    result = []
                    for i in range(num_variable):
                        result.append(int(pulp.value(variable[f"x{i}"])))
                    result = list(np.array(result)*min_bit)
                    if result not in all_candidate_bits:
                        all_candidate_bits.append(result)
                        if logging:
                            logging.info(f"avg_bits_{avg_bit},{result},{pulp.value(prob.objective):.4f}")
                        else:
                            print(f"avg_bits_{avg_bit},{result},{pulp.value(prob.objective):.4f}")
                else:
                    continue

                prob.constraints.popitem() # Delete newly added restrictions

    return all_candidate_bits

if __name__ == '__main__':
    # example
    # avg_bit_list = [2,4,6,8]   
    avg_bit_list = list(np.arange(4, 8.05, 0.05)) 
    bit_list = [2, 4, 6, 8]    
    Hutchinson_trace = [0.1655166950175371, 0.031237508303352764, 0.005463641462108445, 0.04536974264515771, 0.008738782020315292, 0.06347734138133033, 0.013690473541380867, 0.04473570840699332, 0.0009587626490328047, 0.015475474771053072, 0.004567343712089554, 0.012828331558950364, 0.007550859143809667, 0.01605478354862758, 0.0022888322848649252, 0.006276593913161566, 0.0035778317630054454, 0.004113129827947844, -0.0004940806252379266, -0.0024140890352018587]
    # Hutchinson_trace = Hutchinson_trace - np.min(Hutchinson_trace) 

    for avg_bit in avg_bit_list:
        all_candidate_bits = cal_ILP(avg_bit, bit_list, Hutchinson_trace, logging=None)
        # print(f"{all_candidate_bits}")
        






