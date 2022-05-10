import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib

def sp_fn(p):
    sp = 10*p  # can change to 2**p, change the save directory accordingly
    return int(sp)


def Ep_fn(sp, sigma, sigma_c, eta_lam, B):
    Ep = 2*(sigma/np.sqrt(sp) + sigma_c)*np.sqrt((4*eta_lam-1)*np.log(2)/((2*eta_lam - 1)*B))
    return Ep


def lam_map(lam=0.1, type='Polynomial'):
    if type == 'Arithmetic':
        alpha_lam = (1 + 2*lam)/2 
    elif type == 'Geometric':
        alpha_lam = np.sqrt(2*lam)
    elif type == 'Harmonic':
        alpha_lam = 4*lam/(1 + 2*lam)
    else:
        alpha_lam = lam*(2.5 - lam)
    eta_lam = (alpha_lam - lam)/alpha_lam

    return alpha_lam, eta_lam


def Byz_sample(k, optimal_indices):
    if k in optimal_indices:
        obs = np.random.rand(1)[0]/10  # [0, 0.1]
    else:
        obs = np.random.rand(1)[0]/10 + 0.9  # [0.9, 1]
        
    return obs


scenario = 1
mu_global = np.array([0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.95])  
K = len(mu_global)
lam = 0.2  # Byzantine fraction
epsilon = 0.05
global T
T = int(5e4)
N = 100  # repeat times
sigma = 1/2  # arm noise
sigma_c = 1/100  # client noise

delta = 0.001  # 1-delta prob..
map_type = 'Arithmetic'
alpha_lam, eta_lam = lam_map(lam=lam, type=map_type)  # mappings to determine number of groups/clients in groups etc.

M = int(np.ceil(4*np.log(K*T/delta)/(np.log(2)*(4*eta_lam - 1)*alpha_lam)))  # client count
M_Byz = int(np.floor(M*lam))
B = int(np.floor(M/np.ceil(M*alpha_lam)))  #  clients in a group
G = int(np.floor(M/B))  # group count 

omega = 8*sigma_c*np.sqrt(np.log(2)*(4*eta_lam - 1)/(B*(2*eta_lam - 1)))  # unavoidable error margin
beta = omega + epsilon  # regret is defined w.r.t. this

mu_max = np.max(mu_global)
optimal_index = np.argmax(mu_global)  # best arm index
optimal_indices = [*np.where(np.max(mu_global) - mu_global < beta)[0]]  # optimal arm indices
suboptimal_indices = [*set([*range(K)]) - set(optimal_indices)]
arm_regrets = [0 if k in optimal_indices else mu_max - mu_global[k] - beta for k in range(K)]

print('Scenario: {}\n***************************'.format(scenario))  
print('Mapping: {}\nLambda: {}\nK: {}\nM: {}\nM_Byz: {}\nG: {}\nB: {}\nOmega: {:.3f}\nEpsilon: {:.3f}\nBeta: {:.3f}\nBeta-suboptimal arms: {}'.format(map_type, lam, K, M, M_Byz, G, B, omega, epsilon, beta, suboptimal_indices))

regret = np.zeros([N,T])
comm_regret = np.zeros([N,T])
C = 1  # communication loss


for rep in range(N):

    Byz_indices = [*np.random.randint(M, size=M_Byz)]  # indices of Byzantine clients
    
    t = 1
    p = 1
    Ep = 1e6

    active_arms = [*range(K)]

    mu_local = np.zeros([M,K])
    reward_local = np.zeros([M,K])
    pull_num = np.zeros([M,K])

    for m in range(M):
        for k in range(K):
            mu_local[m,k] = np.random.normal(mu_global[k], sigma_c)  # generate local mean
  
    while t<T:

        if Ep > beta/4 and len(active_arms) > 1:
            '''
            local players
            '''
            expl_len = sp_fn(p) - sp_fn(p-1)
            Ep = Ep_fn(sp_fn(p), sigma, sigma_c, eta_lam, B)
            p += 1
            print('n = {}, t = {}, p = {}, Ep = {:.3f}, Beta/4 = {:.3f}, Active Arms: {}'.format(rep, t, p, Ep, beta/4, active_arms), 20* ' ', end='\r')

            for k in active_arms:
                for _ in range(min(T-t,expl_len)):
                    for m in range(M):
                        if m in Byz_indices:
                            reward_local[m,k] += Byz_sample(k, optimal_indices)  # arm sampling
                        else:
                            reward_local[m,k] += np.random.normal(mu_local[m,k], sigma)  # arm sampling
                        pull_num[m,k] += 1

                    regret[rep, t] = regret[rep, t - 1] + M*arm_regrets[k]
                    comm_regret[rep, t] = comm_regret[rep, t - 1]  # comment this line out to ignore communication loss
                    t += 1
            mu_local_sample = reward_local/pull_num

            '''
            global server
            '''
            regret[rep, t - 1] += M*C
            comm_regret[rep, t - 1] = comm_regret[rep, t - 2] + M*C #comment this line out to ignore communication loss
            group_means = np.zeros([G, K])
            for g in range(G):
                start = B*g
                end = B*(g+1) if g < G - 1 else M
                size = end - start
                group_means[g]  = (1/size)*sum(mu_local_sample[start:end,:])

            U_k = np.median(group_means, axis=0)
            comp_val = np.max(U_k) - 2*Ep

            arms_tbd = []  # arms to be deleted
            for arm in active_arms:
                if U_k[arm] <= comp_val:
                    arms_tbd.append(arm)
            
            active_arms = [*set(active_arms) - set(arms_tbd)]

        else: 
            temp_res_reg = np.zeros(T-t)
            for arm in active_arms:
                temp_res_reg += np.arange(T-t)*M*arm_regrets[arm]
            regret[rep, t:] = regret[rep, t - 1] + temp_res_reg
            comm_regret[rep, t:] = comm_regret[rep, t - 1]
            print('Experiment number n: {}'.format(rep), 100*' ')
            print('Elimination Stopped at t = {}, p = {}, Ep = {:.4f}, Beta/4 = {:.4f}'.format(t, p, Ep, beta/4))
            print('Continuing to play the active arms left: {}'.format(active_arms))
            print('Beta-suboptimal arms were: {}\n************************'.format(suboptimal_indices))
            break

    print('*'*50)
    print('n = {}, t = {}, p = {}, Ep = {:.3f}, Beta/4 = {:.3f}\n************************'.format(rep, t, p, Ep, beta/4))
    print('Beta-suboptimal arms were: {}'.format(suboptimal_indices))
    print('Remanining arms were: {}\n************************'.format(active_arms))
    print('Recruited # of clients were: {}'.format(M))

no_comm_reg = regret - comm_regret
np.save('./np_arrays/mean_fmomucb_sce_{}_regret_sp10p_map_{}.npy'.format(scenario, map_type), np.mean(regret, axis=0))
np.save('./np_arrays/std_fmomucb_sce_{}_regret_sp10p_map_{}.npy'.format(scenario, map_type), np.std(regret, axis=0))
np.save('./np_arrays/mean_fmomucb_sce_{}_comm_regret_sp10p_map_{}.npy'.format(scenario, map_type), np.mean(comm_regret, axis=0))
np.save('./np_arrays/std_fmomucb_sce_{}_comm_regret_sp10p_map_{}.npy'.format(scenario, map_type), np.std(comm_regret, axis=0))
np.save('./np_arrays/mean_fmomucb_sce_{}_nocomm_regret_sp10p_map_{}.npy'.format(scenario, map_type), np.mean(no_comm_reg, axis=0))
np.save('./np_arrays/std_fmomucb_sce_{}_nocomm_regret_sp10p_map_{}.npy'.format(scenario, map_type), np.std(no_comm_reg, axis=0))


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

matplotlib.rcParams['pdf.fonttype'] = 42  # no type-3
matplotlib.rcParams['ps.fonttype'] = 42
plt.rcParams['axes.formatter.useoffset'] = False

size = 100
cp = 'colorblind'

lam_arr = np.linspace(start=1e-12, stop=0.499, num=size)
lam_arr_p = np.linspace(start=1e-12, stop=0.5, num=size)
mapping_arr = ['Arithmetic', 'Geometric', 'Harmonic', 'Polynomial']
omega_arr = np.zeros([size,4])
alpha_arr = np.zeros([size,4])
gamma3_arr = np.zeros([size,4])
gamma4_arr = np.zeros([size,4])
gamma5_arr = np.zeros([size,4])
M_arr = np.zeros([size,4])

color_ind = [6, 0, 2, 1]
for index1, lam in enumerate(lam_arr):
    for index2, mapping in enumerate(mapping_arr):
        alpha_lam, eta_lam = lam_map(lam=lam, type=mapping)  
        M = int(np.ceil(4*np.log(K*T/delta)/(np.log(2)*(4*eta_lam - 1)*alpha_lam)))  
        B = int(np.floor(M/np.ceil(M*alpha_lam)))
        omega_arr[index1, index2] = 8*sigma_c*np.sqrt(np.log(2)*(4*eta_lam - 1)/(B*(2*eta_lam - 1))) 
        alpha_arr[index1, index2] = alpha_lam
        gamma3_arr[index1, index2] = 64*sigma**2*np.log(2)*(4*eta_lam -1)/(B*(2*eta_lam - 1))
        gamma4_arr[index1, index2] = 4/((4*eta_lam - 1)*np.log(2)*alpha_lam)
        gamma5_arr[index1, index2] = gamma3_arr[index1, index2]*gamma4_arr[index1, index2]
        M_arr[index1, index2] = M

fs = (6.4,4.8)

plt.figure(figsize=fs)
sns.set_theme(style='whitegrid')
plt.ylim(0, 0.3)
plt.margins(x=0,y=0)
for index2, mapping in enumerate(mapping_arr):
    plt.plot(lam_arr_p, omega_arr[:,index2], label=mapping, color=sns.color_palette(cp)[color_ind[index2]], LineWidth=3)
plt.xlabel(r'$\lambda$')
plt.ylabel(r'$\omega$')
plt.legend(facecolor='white', fontsize=15)
plt.savefig('./figures/omega.pdf')
plt.show()

plt.figure(figsize=fs)
plt.ylim(0, 30)
plt.margins(x=0,y=0)
sns.set_theme(style='whitegrid')
for index2, mapping in enumerate(mapping_arr):
    plt.plot(lam_arr_p, gamma4_arr[:,index2], label=mapping, color=sns.color_palette(cp)[color_ind[index2]], LineWidth=3)
plt.xlabel(r'$\lambda$')
plt.ylabel(r'$\Gamma_4$')
plt.legend(facecolor='white', fontsize=15)
plt.savefig('./figures/gamma4.pdf')
plt.show()

plt.figure(figsize=fs)
plt.ylim(0, 2000)
plt.margins(x=0,y=0)
sns.set_theme(style='whitegrid')
for index2, mapping in enumerate(mapping_arr):
    plt.plot(lam_arr_p, gamma5_arr[:,index2], label=mapping, color=sns.color_palette(cp)[color_ind[index2]], LineWidth=3)
plt.xlabel(r'$\lambda$')
plt.ylabel(r'$\Gamma_5$')
plt.legend(facecolor='white', fontsize=15)
plt.savefig('./figures/gamma5.pdf')
plt.show()

plt.figure(figsize=fs)
plt.margins(x=0,y=0)
sns.set_theme(style='whitegrid')
for index2, mapping in enumerate(mapping_arr):
    plt.plot(lam_arr, alpha_arr[:,index2], label=mapping, color=sns.color_palette(cp)[color_ind[index2]], LineWidth=3)
plt.xlabel(r'$\lambda$')
plt.ylabel(r'$\alpha(\lambda)$')
plt.legend(facecolor='white')
plt.savefig('./figures/alpha.pdf')
plt.show()