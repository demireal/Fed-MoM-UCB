'''
Adapted from: https://github.com/Chengshuai-Shi/FMAB, The code for:

C. Shi and C. Shen, “Federated multi-armed bandits,” Proceedings
of the AAAI Conference on Artificial Intelligence, vol. 35,
no. 11, pp. 9603–9611, 2021
'''

import numpy as np
import matplotlib.pyplot as plt

def fp(p):
    fp = 10  # s(p) = 10*p
    return int(fp)


def gp(p):
    gp = 10  # can change to 2**p, change the save directory accordingly
    return int(gp)


def Byz_sample(k, optimal_indices):
    if k in optimal_indices:
        obs = np.random.rand()/10  # [0, 0.1]
    else:
        obs = np.random.rand()/10 + 0.9  # [0.9, 1]
        
    return obs

scenario = 1
mu_global = np.array([0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.95])  
K = len(mu_global)
lam = 0.2  # Byzantine fraction  
global T
T = int(5e4)
N = 100  # repeat times
sigma = 1/2  # arm noise
sigma_c = 1/100  # client noise

omega = 0.139
epsilon = 0.05
beta = omega + epsilon  # regret is defined w.r.t. this

mu_max = np.max(mu_global)
optimal_index = np.argmax(mu_global)  # best arm index
optimal_indices = [*np.where(np.max(mu_global) - mu_global < beta)[0]]  # optimal arm indices
suboptimal_indices = [*set([*range(K)]) - set(optimal_indices)]
arm_regrets = [0 if k in optimal_indices else mu_max - mu_global[k] - beta for k in range(K)]

print('Scenario: {}\n***************************'.format(scenario))   
print('Lambda: {}\nOmega: {:.3f}\nEpsilon: {:.3f}\nBeta: {:.3f}\nBeta-suboptimal arms: {}'.format(lam, omega, epsilon, beta, suboptimal_indices))

comm_c = 0
regret = np.zeros([N,T])
comm_regret = np.zeros([N,T])
M_arr = np.zeros(N)

for rep in range(N):
    t = 1
    p = 1
    M = 0
    
    active_arm = np.array(range(K),dtype = int)
    C = 1  # communication loss
    
    Byz_indices = []

    while t < T:
        '''
        round p
        '''
        
        '''
        local players
        '''
        if len(active_arm) > 1:
            player_add_num = gp(p)
            if M==0:
                M += 1
                pull_num = np.zeros([1,K])
                reward_local = np.zeros([1,K])
                mu_local = np.zeros([1,K])
                for k in range(K):
                    mu_local[M-1,k] = np.random.normal(mu_global[k], sigma_c) #generated local mean
                player_add_num -= 1

            for m in range(player_add_num):
                if m < np.floor(player_add_num*lam):
                    Byz_indices.append(M + m)
            
            for m in range(player_add_num):
                M += 1
                pull_num = np.r_[pull_num, np.zeros([1,K])]
                reward_local = np.r_[reward_local, np.zeros([1,K])]
                mu_local = np.r_[mu_local, np.zeros([1,K])]
                for k in range(K):
                    mu_local[M-1,k] = np.random.normal(mu_global[k], sigma_c) #generated local mean
            
            expl_len = fp(p)
            p += 1
        
        if len(active_arm) > 1:
            for k in active_arm:
                for _ in range(min(T-t,expl_len)):
                    for m in range(M):
                        if m in Byz_indices:
                            reward_local[m,k] += Byz_sample(k, optimal_indices)  # arm sampling
                        else:
                            reward_local[m,k] += np.random.normal(mu_local[m,k],sigma)
                        pull_num[m,k] += 1

                    regret[rep, t] = regret[rep, t - 1] + M*arm_regrets[k]
                    comm_regret[rep, t] = comm_regret[rep, t - 1]  # comment this line out to ignore communication loss
                    t = t+1
            mu_local_sample = reward_local/pull_num

        if len(active_arm) == 1:
            regret[rep, t:] = regret[rep, t - 1] + np.arange(T-t)*M*arm_regrets[active_arm[0]]
            comm_regret[rep, t:] = comm_regret[rep, t - 1]
            print('Experiment number n: {}'.format(rep), 100*' ')
            print('Elimination Stopped at t = {}, p = {}'.format(t, p))
            print('Continuing to play the active arms left: {}'.format(active_arm))
            print('Beta-suboptimal arms were: {}\n************************'.format(suboptimal_indices))
            break
        
        '''
        global server
        '''
        if len(active_arm) > 1:
            regret[rep, t-1] += M*C #comment this line out to ignore communication loss
            comm_regret[rep, t - 1] = comm_regret[rep, t - 2] + M*C #comment this line out to ignore communication loss
            comm_c += M*C
            E = np.array([])
            mu_global_sample = 1/M*sum(mu_local_sample)
            eta_p = 0
            for i in range(1,p): # p has been added one above
                F_d = 0
                for j in range(i,p):
                    F_d += fp(j)
                eta_p += 1/M**2*gp(i)/F_d
            
            conf_bnd = np.sqrt(6*sigma**2*eta_p*np.log(T))+np.sqrt(6*sigma_c**2*np.log(T)/(M)) #the constants are tuned from the original ones in the paper to get better performance

            elm_max = np.nanmax(mu_global_sample)-conf_bnd
            for index in range(len(active_arm)):
                arm = active_arm[index]
                if mu_global_sample[arm]+conf_bnd<elm_max:
                    E = np.append(E,np.array([arm]))
        
            for i in range(len(E)):
                active_arm = np.delete(active_arm, np.where(active_arm == E[i]))

    print('*'*50)
    print('n = {}, t = {}, p = {}'.format(rep, t, p))
    print('Beta-suboptimal arms were: {}'.format(suboptimal_indices))
    print('Remanining arms were: {}\n************************'.format(active_arm))
    print('Recruited # of clients were: {}'.format(M))
    M_arr[rep] = M

no_comm_reg = regret - comm_regret
np.save('./np_arrays/mean_fucb_sce_{}_regret_sp10p_mp10.npy'.format(scenario), np.mean(regret, axis=0))
np.save('./np_arrays/std_fucb_sce_{}_regret_sp10p_mp10.npy'.format(scenario), np.std(regret, axis=0))
np.save('./np_arrays/mean_fucb_sce_{}_comm_regret_sp10p_mp10.npy'.format(scenario), np.mean(comm_regret, axis=0))
np.save('./np_arrays/std_fucb_sce_{}_comm_regret_sp10p_mp10.npy'.format(scenario), np.std(comm_regret, axis=0))
np.save('./np_arrays/mean_fucb_sce_{}_nocomm_regret_sp10p_mp10.npy'.format(scenario), np.mean(no_comm_reg, axis=0))
np.save('./np_arrays/std_fucb_sce_{}_nocomm_regret_sp10p_mp10.npy'.format(scenario), np.std(no_comm_reg, axis=0))
np.save('./np_arrays/fucb_sce_{}_M_arr_sp10p_mp10.npy'.format(scenario), M_arr)


coeff = 2
plt.figure()
avg_regret = np.mean(regret, axis = 0)
std_regret = np.std(regret, axis = 0)
avg_nocomm_regret = np.mean(no_comm_reg, axis = 0)
std_nocomm_regret = np.std(no_comm_reg, axis = 0)
plt.plot(range(T),avg_regret, label='Regret')
plt.plot(range(T),avg_nocomm_regret, label='No Comm. Regret')
plt.fill_between(range(T), avg_regret - coeff*std_regret, avg_regret + coeff*std_regret, alpha=0.2)
plt.legend()
plt.show()

plt.figure()
avg_regret = np.mean(comm_regret, axis = 0)
std_regret = np.std(comm_regret, axis = 0)
plt.plot(range(T),avg_regret, label='Comm Regret')
plt.fill_between(range(T), avg_regret - coeff*std_regret, avg_regret + coeff*std_regret, alpha=0.2)
plt.legend()
plt.show()