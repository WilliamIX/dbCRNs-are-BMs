import numpy as np
import itertools
from bioscrape.types import Model
from bioscrape.simulator import py_simulate_model

def db_rxns(inputs, outputs, EnergyDict, k = 1.):
    dE = sum([EnergyDict[o] for o in outputs])-sum([EnergyDict[i] for i in inputs])
    rxn1 = [inputs, outputs, "massaction", {"k":k*np.exp(-dE/2)}]
    rxn2 = [outputs, inputs, "massaction", {"k":k*np.exp(dE/2)}]
    return [rxn1, rxn2]

def chemostat_db_rxns(inputs, outputs, input_chemostats, output_chemostats, EnergyDict, k = 1):
    dE = sum([EnergyDict[o] for o in outputs])-sum([EnergyDict[i] for i in inputs])
    
    kf = ""
    for i in input_chemostats+inputs:
        kf += i+"*"
    kf += str(np.exp(-dE/2)*k)
    
    kr = ""
    for o in output_chemostats+outputs:
        kr += o + "*"
    kr += str(np.exp(dE/2)*k)

    rxn1 = [inputs, outputs, "general", {"rate":kf}]
    rxn2 = [outputs, inputs, "general", {"rate":kr}]
    
    return [rxn1, rxn2]
    #print(inputs, "-->", outputs, "@", kf)
    #print(outputs, "-->", inputs, "@", kr)
    
def add_learning_rule(M, P, Q, X, k, dt, N = None):
    if isinstance(k, tuple):
        kx = k[0]
        kq = k[1]
    else:
        kx = k
        kq = k
        
    if N is not None:
        kx = kx/N
    
    M.create_rule("assignment", {"equation":f"{P} = {P}+{P}*({kx}*{X} - {kq}*{Q})*{dt}"})

#k is learning factor times epsilon
#delta makes this rule reversible
def create_rule_tuple(P, Q, X, k, dt, N = None, delta = 0):
    if isinstance(k, tuple):
            kx = k[0]
            kq = k[1]
    else:
        kx = k
        kq = k

    if N is not None:
        kx = kx/N
    
    #Non revesible
    #Reactions: 
    # P + X --> 2 P + X @ kx
    # Q + P --> Q @ kq
    dPdt = f"{P}*({kx}*{X} - {kq}*{Q})"
    
    if delta != 0:
        #Reactions:
        # 2P + X --> P + X @ delta
        # Q --> Q + P @ delta
        dPdt += f"+{delta}*({Q} - {X}*{P}^2)"
        
    rule = ("assignment", {"equation":f"{P} = {P}+({dPdt})*{dt}"})
    return rule
    
    
def ECBM_rxns(flip, neighbors, edges, potentials, E, clamped_species = [], k = 1, v = None):
    #v is the vesicle
    if v is None:
        v_s = ""
    else:
        v_s = f"_{v}"
        
    reactions = []
    combos = itertools.product([0, 1], repeat = len(neighbors))
    for c in combos:
        inputs = [flip+"_0"+v_s]
        outputs = [flip + "_1"+v_s]
        chemo_in = []
        chemo_out = []
        if flip in potentials:
            #Potentials don't have v strings
            if isinstance(potentials[flip], list):
                for p in potentials[flip]:
                    chemo_in.append(p+"_0")
                    chemo_out.append(p+"_1")
            else:
                chemo_in.append(potentials[flip]+"_0")
                chemo_out.append(potentials[flip]+"_1")
        for i in range(len(c)):
            inputs.append(neighbors[i]+f"_{c[i]}"+v_s)
            outputs.append(neighbors[i]+f"_{c[i]}"+v_s)
            if c[i] == 1:
                inputs.append(edges(flip, neighbors[i])+"_0"+v_s)
                outputs.append(edges(flip, neighbors[i])+"_1"+v_s)
                
                if edges(flip, neighbors[i]) in potentials and flip not in clamped_species:
                    
                    if isinstance(potentials[edges(flip, neighbors[i])], list):
                        for p in potentials[edges(flip, neighbors[i])]:
                            chemo_in.append(p+"_0")
                            chemo_out.append(p+"_1")
                    else:
                        chemo_in.append(potentials[edges(flip, neighbors[i])]+"_0")
                        chemo_out.append(potentials[edges(flip, neighbors[i])]+"_1")
        reactions += chemostat_db_rxns(inputs, outputs, chemo_in, chemo_out, E, k)
    return reactions



def clamping_reactions(P, Q, S, k=1.0, epsilon = .01, delta = None, opposite_side_potential = False, V = 1.0):
    if isinstance(k, tuple):
        k_Q = k[0]/V
        k_S = k[1]/V
    else:
        k_Q = k/V
        k_S = k/V
    
    rxns = []
    
    if not opposite_side_potential:
        if delta is None:#Irreversible clamping
            rxns.append(([Q, P], [Q], 'massaction', {"k":k_Q*epsilon}))
            rxns.append(([S, P], [S, P, P], 'massaction', {"k":k_S*epsilon}))
        else:#this is used for thermodynamics section
            rxns.append(([Q, P], [Q, f"rxn_{Q}_{P}"], 'massaction', {"k":k_Q*epsilon}))
            rxns.append(([S, P], [S, P, P, f"rxn_{S}_{P}"], 'massaction', {"k":k_S*epsilon}))
            rxns.append(([Q], [Q, P, f"rxn_{Q}_{P}_r"], 'massaction', {"k":k_Q*epsilon*delta}))
            rxns.append(([S, P, P], [S, P, f"rxn_{S}_{P}_r"], 'massaction', {"k":k_S*epsilon*delta}))
            
    else:
        if delta is None:#Irreversible clamping
            rxns.append(([Q, P], [Q, P, P], 'massaction', {"k":k_Q*epsilon}))
            rxns.append(([S, P], [S], 'massaction', {"k":k_S*epsilon/V}))
        elif delta is not None:#this is used for thermodynamics section
            rxns.append(([Q, P], [Q, P, P, f"rxn_{Q}_{P}"], 'massaction', {"k":k_Q*epsilon}))
            rxns.append(([S, P], [S, f"rxn_{S}_{P}"], 'massaction', {"k":k_S*epsilon/V}))
            rxns.append(([Q, P, P], [Q, P, f"rxn_{Q}_{P}_r"], 'massaction', {"k":k_Q*epsilon*delta}))
            rxns.append(([S], [S, P, f"rxn_{S}_{P}_r"], 'massaction', {"k":k_S*epsilon*delta}))
    
    return rxns

def learning_vesicle(v, kdb, k_learn, k_copy, E = None, delta_clamp = 0, delta_learn = 0, dt = .001):
    #Learn an XOR Toggle with an ECBM
    
    if v == None:
        vs = ""
    else:
        vs = f"_{v}"
    
    #Free Species
    speciesX = [f"X0_0{vs}", f"X0_1{vs}", f"X1_0{vs}", f"X1_1{vs}", f"H_0{vs}", f"H_1{vs}"]
    speciesW =[f"W0_0{vs}", f"W0_1{vs}", f"W1_0{vs}", f"W1_1{vs}"]

    #Clamped Species
    speciesQ =[f"Q0_0{vs}", f"Q0_1{vs}", f"Q1_0{vs}", f"Q1_1{vs}", f"QH_0{vs}", f"QH_1{vs}"]
    speciesQW =[f"QW0_0{vs}", f"QW0_1{vs}", f"QW1_0{vs}", f"QW1_1{vs}"]

    #Potential Species
    speciesP =["PX0_0", "PX0_1", "PX1_0", "PX1_1", "PH_0", "PH_1", "PW0_0", "PW0_1", "PW1_0", "PW1_1", "PQ0_0", "PQ0_1", "PQ1_0", "PQ1_1"]

    #Environment Species
    species = speciesX+speciesQ+speciesW+speciesQW+speciesP

    if E is None:
        E = {s:0 for s in species}
    else:
        for s in species:
            if s not in E:
                E[s] = 0
    
    #print("E", E)

    #potentials = {s.split("_")[0]:"P"+s.split("_")[0] for s in speciesX + speciesW}
    #potentials.update({s.split("_")[0]:"P"+s.split("_")[0][1:] for s in speciesQW+[f"QH"]})
    #potentials.update({f"Q0":["PQ0", "PX0"], f"Q1":["PQ1", "PX1"]})
    potentials = {
        "X0" : "PX0", "X1": "PX1", "H":"PH",
        "W0" : "PW0", "W1":"PW1",
        "Q0": ["PX0", "PQ0"], "Q1": ["PX1", "PQ1"], "QH":"PH",
        "QW0": "PW0", "QW1": "PW1"
    }
    
    #potentials = {}
    
    #print(potentials)

    edge_dict = {
        ("X0", "H"):"W0", ("X1", "H"):"W1",
        ("Q0", "QH"):"QW0", ("Q1", "QH"):"QW1",
    }
    def edges(x, y):
        if (x, y) in edge_dict:
            return edge_dict[x, y]
        elif (y, x) in edge_dict:
            return edge_dict[y, x]
        else:
            raise ValueError(f"Edge {x}, {y} not in edge_dict")

    clamped = []
    reactions_free = []
    reactions_free += ECBM_rxns("X0", ["H"], edges, potentials, E, clamped, kdb, v = v)
    reactions_free += ECBM_rxns("X1", ["H"], edges, potentials, E, clamped, kdb, v = v)
    reactions_free += ECBM_rxns("H", ["X0", "X1"], edges, potentials, E, clamped, kdb, v = v)

    reactions_clamped = []
    reactions_clamped += ECBM_rxns("Q0", ["QH"], edges, potentials, E, clamped, kdb, v = v)
    reactions_clamped += ECBM_rxns("Q1", ["QH"], edges, potentials, E, clamped, kdb, v = v)
    reactions_clamped += ECBM_rxns("QH", ["Q0", "Q1"], edges, potentials, E, clamped, kdb, v = v)

    reactions = reactions_free + reactions_clamped
    
    x0 = {}
    
    for s in species:
        if s not in x0:
            s_val = s.split("_")[1]
            if s_val == "0" and "P" not in s:
                x0[s] = abs(v%2)
            elif s_val == "1" and "P" not in s:
                x0[s] = abs((v%2)-1)
            else:
                x0[s] = 1

    
    #for s in species:
    #    M._add_species(s)
    #for r in reactions:
    #    M._add_reaction(r)
    #M.add_species(species)
    #M.add_reactions(reactions)
    #M.set_species(x0)
    #M = Model(species = species, reactions = reactions, initial_condition_dict = x0)

    rules = []
    rules += [create_rule_tuple("PX0_0", f"Q0_0{vs}", f"X0_0{vs}", k_learn, dt, delta = delta_learn)]
    rules += [create_rule_tuple("PX0_1", f"Q0_1{vs}", f"X0_1{vs}", k_learn, dt, delta = delta_learn)]
    rules += [create_rule_tuple("PX1_0", f"Q1_0{vs}", f"X1_0{vs}", k_learn, dt, delta = delta_learn)]
    rules += [create_rule_tuple("PX1_1", f"Q1_1{vs}", f"X1_1{vs}", k_learn, dt, delta = delta_learn)]

    rules += [create_rule_tuple("PH_0", f"QH_0{vs}", f"H_0{vs}", k_learn, dt, delta = delta_learn)]
    rules += [create_rule_tuple("PH_1", f"QH_1{vs}", f"H_1{vs}", k_learn, dt, delta = delta_learn)]


    rules += [create_rule_tuple("PW0_0", f"QW0_0{vs}", f"W0_0{vs}", k_learn, dt, delta = delta_learn)]
    rules += [create_rule_tuple("PW0_1", f"QW0_1{vs}", f"W0_1{vs}", k_learn, dt, delta = delta_learn)]
    rules += [create_rule_tuple("PW1_0", f"QW1_0{vs}", f"W1_0{vs}", k_learn, dt, delta = delta_learn)]
    rules += [create_rule_tuple("PW1_1", f"QW1_1{vs}", f"W1_1{vs}", k_learn, dt, delta = delta_learn)]

    #rules += [create_rule_tuple(M, "PQ0_0", "R0", f"Q0_0_{v}", k_copy, dt)]
    
    #rules += [create_rule_tuple("PQ0_0", f"Q0_1{vs}", "R0", k_copy, dt)]
    rules += [create_rule_tuple("PQ0_1", "R0", f"Q0_1{vs}", k_copy, dt, delta = delta_clamp)]
    
    
    
    #rules += [create_rule_tuple(M, "PQ1_0", "E1_0", f"Q1_0_{v}", k_copy, dt)]
    #rules += [create_rule_tuple("PQ1_0", f"Q1_1{vs}", "R1", k_copy, dt)]
    rules += [create_rule_tuple("PQ1_1", "R1", f"Q1_1{vs}", k_copy, dt, delta = delta_clamp)]

    #for r in reactions:
    #    print(r)
    
    return species, reactions, rules, E, x0

#DKL between two binary distributions.
def DKL_bin(P, Q):
    dkl = 0
    for i in [0, 1]:
        for j in [0, 1]:
            dkl += P[i, j]*np.log(P[i, j])-P[i, j]*np.log(Q[i, j])
    return dkl

def MPfunc(args):
    ( N, n_iters, save_path, iter_factor, kdb, k_learn, k_copy, epsilon_copy, epsilon_learn, alpha, di_copy, di_learn, dt, timepoints, x0, species, reactions, rules) = args
    delta_copy = epsilon_copy/di_copy
    delta_learn = epsilon_learn/di_learn
        
    #Here are the different potentials for each vesicle
    # A dictionary P --> X, Q, k, delta [of the moment learning reactions]
    #v denotes vesicle index and will be replaced in the loop
    potentials = {
        "PQ0_1":("Q0_1_v", "R0", epsilon_copy, delta_copy), 
        "PQ1_1":("Q1_1_v", "R1", epsilon_copy, delta_copy),
        "PX0_0":("X0_0_v", "Q0_0_v", epsilon_learn, delta_learn), 
        "PX0_1":("X0_1_v", "Q0_1_v", epsilon_learn, delta_learn),
        "PX1_0":("X1_0_v", "Q1_0_v", epsilon_learn, delta_learn),
        "PX1_1":("X1_1_v", "Q1_1_v", epsilon_learn, delta_learn),
        "PH_0":("H_0_v", "QH_0_v", epsilon_learn, delta_learn), 
        "PH_1":("H_1_v", "QH_1_v", epsilon_learn, delta_learn),
        "PW0_0":("W0_0_v", "QW0_0_v", epsilon_learn, delta_learn),
        "PW0_1":("W0_1_v", "QW0_1_v", epsilon_learn, delta_learn), 
        "PW1_0":("W1_0_v", "QW1_0_v", epsilon_learn, delta_learn), 
        "PW1_1":("W1_1_v", "QW1_1_v", epsilon_learn, delta_learn)
    }

    print("\nStarting di_copy", di_copy, "di_learn", di_learn, end = "...")
    #Create N vesicles
    for v in range(N):
        species_v, reactions_v, rules_v, E_v, x0_v = learning_vesicle(v, kdb, k_learn, k_copy, E = None, delta_learn = delta_learn, delta_clamp = delta_copy, dt = dt)
        x0.update(x0_v)
        species += species_v
        reactions += reactions_v
        rules += rules_v

    M = Model(species = species, reactions = reactions, initial_condition_dict = x0, rules = rules)


    dist_Q = np.zeros((2, 2))
    dist_E = np.zeros((500, 500))

    print("simulating", end = "...")
    np.random.seed()
    results_list = []
    for iteration in range(n_iters):
        dist_X = np.zeros((2, 2))
        print(iteration, end = "...")
        results_i = py_simulate_model(timepoints, M, stochastic = True, return_dataframe = False)
        df_i = results_i.py_get_dataframe(Model = M)
        M.set_species({
                s:df_i[s].to_numpy()[-1] for s in M.get_species_dictionary()
            }
        )
        #results_list.append((results_i, df_i))

        #Compute distributions and entropies every iter_factor simulations
        if (iter_factor == 1) or ((iteration+1) % iter_factor == 0) or iteration == 0:
            print("computing distributions", end = "...")
            #compute distributions
            maxR = int(max(max(df_i["R0"]), max(df_i["R1"]))+1)
            #dist_E[:maxR+1, :maxR+1] += results_i.py_empirical_distribution(species = ["R0", "R1"], start_time = 0, Model = M, max_counts = [maxR, maxR])/n_iters

            for v in range(N):
                dist_X += results_i.py_empirical_distribution(species = [f"X0_1_{v}", f"X1_1_{v}"], start_time = 0, Model = M, max_counts = [1, 1])/(N*n_iters/iter_factor)
                dist_Q += results_i.py_empirical_distribution(species = [f"Q0_1_{v}", f"Q1_1_{v}"], start_time = 0, Model = M, max_counts = [1, 1])/(n_iters/iter_factor)

            #Calculate Binary Version of dist E
            #dist_E_bin = np.zeros((2, 2))
            #for i in range(dist_E.shape[0]):
            #    for j in range(dist_E.shape[1]):
            #        dist_E_bin[min(int(i*alpha), 1), min(int(j*alpha), 1)] += dist_E[i, j]

            #Compute DKLs
            #DKL_EQ = DKL_bin(dist_E_bin, dist_Q)
            #DKL_EX = DKL_bin(dist_E_bin, dist_X)
            #DKL_QX = DKL_bin(dist_Q, dist_X)

            #Compute fluxes
            print("computing thermodynamics", end = "...")
            # J_i^S = [P_i] * <S_i> * (epsilon - delta [P_i])
            JS = {p:np.zeros(len(timepoints[::100])) for p in potentials}
            # J_i^Q = <Q_i>*(epsilon [P_i] - delta)
            JQ = {p:np.zeros(len(timepoints[::100])) for p in potentials}
            #T dEntropy_i/dt = RT(J_i^S log( epsilon/(delta P_i)) - J_i^Q log( epsilon P_i / delta ))
            dSdt = {p:np.zeros(len(timepoints[::100])) for p in potentials}

            for p in potentials:
                X, Q, k, d = potentials[p]

                for i in range(N):
                    x = X.replace("v", str(i))
                    if "v" in Q:
                        q = Q.replace("v", str(i))
                    else:
                        q = Q

                    JS[p] += df_i[p].to_numpy()[::100]*df_i[x].to_numpy()[::100]*(k - d*df_i[p].to_numpy()[::100])
                    JQ[p] += df_i[q].to_numpy()[::100]*(k*df_i[p].to_numpy()[::100]-d)

                dSdt[p] += JS[p] * np.log(k / (d*df_i[p].to_numpy()[::100])) + JQ[p] * np.log(k*df_i[p].to_numpy()[::100]/d)

            fnameQ = f"distQ_dcopy={di_copy}_dlearn={di_learn}_iter={iteration}"
            fnameX = f"distX_dcopy={di_copy}_dlearn={di_learn}_iter={iteration}"
            #fnameE = f"distE_dcopy={di_copy}_dlearn={di_learn}_iter={iteration}"
            fnameS = f"dSdt_dcopy={di_copy}_dlearn={di_learn}_iter={iteration}"
            #fnameDKLEQ = f"dklEQ_dcopy={di_copy}_dlearn={di_learn}_iter={iteration}"
            #fnameDKLEX = f"dklEX_dcopy={di_copy}_dlearn={di_learn}_iter={iteration}"
            #fnameDKLQX = f"dklQX_dcopy={di_copy}_dlearn={di_learn}_iter={iteration}"

            print("Saving Data", end = "...")
            np.save(save_path+fnameQ, dist_Q)
            np.save(save_path+fnameX, dist_X)
            #np.save(save_path+fnameE, dist_E_bin)
            np.save(save_path+fnameS, dSdt)
            #np.save(save_path+fnameDKLEQ, DKL_EQ)
            #np.save(save_path+fnameDKLEX, DKL_EX)
            #np.save(save_path+fnameDKLQX, DKL_QX)