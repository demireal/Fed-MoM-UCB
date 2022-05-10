import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

matplotlib.rcParams['pdf.fonttype'] = 42  # no type-3
matplotlib.rcParams['ps.fonttype'] = 42
plt.rcParams['axes.formatter.useoffset'] = False

cp = 'colorblind'

#####################################################      SCENARIO 1       #####################################################

fmomucb_sce_1_sp10p_ar = np.load('./np_arrays/mean_fmomucb_sce_1_regret_sp10p_map_Arithmetic.npy')
fmomucb_sce_1_sp10p_sr = np.load('./np_arrays/std_fmomucb_sce_1_regret_sp10p_map_Arithmetic.npy')

fmomucb_sce_1_sp10p_comm_ar = np.load('./np_arrays/mean_fmomucb_sce_1_comm_regret_sp10p_map_Arithmetic.npy')
fmomucb_sce_1_sp10p_comm_sr = np.load('./np_arrays/std_fmomucb_sce_1_comm_regret_sp10p_map_Arithmetic.npy')

fmomucb_sce_1_sp10p_nocomm_ar = np.load('./np_arrays/mean_fmomucb_sce_1_nocomm_regret_sp10p_map_Arithmetic.npy')
fmomucb_sce_1_sp10p_nocomm_sr = np.load('./np_arrays/std_fmomucb_sce_1_nocomm_regret_sp10p_map_Arithmetic.npy')

###########################################################################################################################

fmomucb_sce_1_sp2_to_p_ar = np.load('./np_arrays/mean_fmomucb_sce_1_regret_sp2_to_p_map_Arithmetic.npy')
fmomucb_sce_1_sp2_to_p_sr = np.load('./np_arrays/std_fmomucb_sce_1_regret_sp2_to_p_map_Arithmetic.npy')

fmomucb_sce_1_sp2_to_p_comm_ar = np.load('./np_arrays/mean_fmomucb_sce_1_comm_regret_sp2_to_p_map_Arithmetic.npy')
fmomucb_sce_1_sp2_to_p_comm_sr = np.load('./np_arrays/std_fmomucb_sce_1_comm_regret_sp2_to_p_map_Arithmetic.npy')

fmomucb_sce_1_sp2_to_p_nocomm_ar = np.load('./np_arrays/mean_fmomucb_sce_1_nocomm_regret_sp2_to_p_map_Arithmetic.npy')
fmomucb_sce_1_sp2_to_p_nocomm_sr = np.load('./np_arrays/std_fmomucb_sce_1_nocomm_regret_sp2_to_p_map_Arithmetic.npy')

###########################################################################################################################

fucb_sce_1_sp10p_mp10_ar = np.load('./np_arrays/mean_fucb_sce_1_regret_sp10p_mp10.npy')
fucb_sce_1_sp10p_mp10_sr = np.load('./np_arrays/std_fucb_sce_1_regret_sp10p_mp10.npy')

fucb_sce_1_sp10p_mp10_comm_ar = np.load('./np_arrays/mean_fucb_sce_1_comm_regret_sp10p_mp10.npy')
fucb_sce_1_sp10p_mp10_comm_sr = np.load('./np_arrays/std_fucb_sce_1_comm_regret_sp10p_mp10.npy')

fucb_sce_1_sp10p_mp10_nocomm_ar = np.load('./np_arrays/mean_fucb_sce_1_nocomm_regret_sp10p_mp10.npy')
fucb_sce_1_sp10p_mp10_nocomm_sr = np.load('./np_arrays/std_fucb_sce_1_nocomm_regret_sp10p_mp10.npy')

fucb_sce_1_sp10p_mp10_M_arr = np.load('./np_arrays/fucb_sce_1_M_arr_sp10p_mp10.npy')

###########################################################################################################################

fucb_sce_1_sp10p_mp2_to_p_ar = np.load('./np_arrays/mean_fucb_sce_1_regret_sp10p_mp2_to_p.npy')
fucb_sce_1_sp10p_mp2_to_p_sr = np.load('./np_arrays/std_fucb_sce_1_regret_sp10p_mp2_to_p.npy')

fucb_sce_1_sp10p_mp2_to_p_comm_ar = np.load('./np_arrays/mean_fucb_sce_1_comm_regret_sp10p_mp2_to_p.npy')
fucb_sce_1_sp10p_mp2_to_p_comm_sr = np.load('./np_arrays/std_fucb_sce_1_comm_regret_sp10p_mp2_to_p.npy')

fucb_sce_1_sp10p_mp2_to_p_nocomm_ar = np.load('./np_arrays/mean_fucb_sce_1_nocomm_regret_sp10p_mp2_to_p.npy')
fucb_sce_1_sp10p_mp2_to_p_nocomm_sr = np.load('./np_arrays/std_fucb_sce_1_nocomm_regret_sp10p_mp2_to_p.npy')

fucb_sce_1_sp10p_mp2_to_p_M_arr = np.load('./np_arrays/fucb_sce_1_M_arr_sp10p_mp2_to_p.npy')

###########################################################################################################################

T = 50000
b = 2

t1 = [*range(0,T,int(T/1000))]
t2 = [*range(0,T,int(T/1000))]

i1 = [*range(0,len(t1),int(len(t1)/10))]
i2 = [*range(0,len(t1),int(len(t1)/10))]

sty1 = {
    'fillstyle': 'none',
    'linewidth': 1.0,
    'markevery': i1}

sty2 = {
    'fillstyle': 'none',
    'linewidth': 1.0,
    'markevery': i2}

print('SCENARIO 1 PLOTS\n', '*'*50)


fs = (6.4, 4.8)
lw = 1

plt.figure(figsize=fs)
plt.ylim(-2e5,8.1e6)
plt.ticklabel_format(axis="y", style="sci", scilimits=(1,0))
plt.ticklabel_format(axis="x", style="sci", scilimits=(1,0))
plt.plot(t1, fmomucb_sce_1_sp10p_ar[t1], label=r'Fed-MoM-UCB, $s(p) = 10p$', **sty1, marker='*', color = sns.color_palette(cp)[0])
plt.fill_between(t1, fmomucb_sce_1_sp10p_ar[t1] - b*fmomucb_sce_1_sp10p_sr[t1], fmomucb_sce_1_sp10p_ar[t1] + b*fmomucb_sce_1_sp10p_sr[t1], color = sns.color_palette(cp)[0], alpha=0.2)
plt.plot(t1, fmomucb_sce_1_sp2_to_p_ar[t1], label=r'Fed-MoM-UCB, $s(p) = 2^p$', **sty1, marker='d', color = sns.color_palette(cp)[1])
plt.fill_between(t1, fmomucb_sce_1_sp2_to_p_ar[t1] - b*fmomucb_sce_1_sp2_to_p_sr[t1], fmomucb_sce_1_sp2_to_p_ar[t1] + b*fmomucb_sce_1_sp2_to_p_sr[t1], color = sns.color_palette(cp)[1], alpha=0.2)
plt.plot(t1, fucb_sce_1_sp10p_mp10_ar[t1], label=r'Fed2-UCB, $s(p) = 10p$, $M(p) = 10$', **sty1, marker='s', color = sns.color_palette(cp)[6])
plt.fill_between(t1, fucb_sce_1_sp10p_mp10_ar[t1] - b*fucb_sce_1_sp10p_mp10_sr[t1], fucb_sce_1_sp10p_mp10_ar[t1] + b*fucb_sce_1_sp10p_mp10_sr[t1],color = sns.color_palette(cp)[6], alpha=0.2)
plt.plot(t1, fucb_sce_1_sp10p_mp2_to_p_ar[t1], label=r'Fed2-UCB, $s(p) = 10p$, $M(p) = 2^p$', **sty1, color = sns.color_palette(cp)[2])
plt.fill_between(t1, fucb_sce_1_sp10p_mp2_to_p_ar[t1] - b*fucb_sce_1_sp10p_mp2_to_p_sr[t1], fucb_sce_1_sp10p_mp2_to_p_ar[t1] + b*fucb_sce_1_sp10p_mp2_to_p_sr[t1],color = sns.color_palette(cp)[2], alpha=0.2)
plt.legend(facecolor='white', framealpha=1, fontsize=12)
plt.xlabel('t')
plt.ylabel('Cumulative regret')
plt.savefig('./figures/sce1_reg.pdf')
plt.show()

##########################################################################################

plt.figure(figsize=fs)
plt.ylim(-1e4,3e5)
plt.ticklabel_format(axis="y", style="sci", scilimits=(1,0))
plt.ticklabel_format(axis="x", style="sci", scilimits=(1,0))
plt.plot(t1, fmomucb_sce_1_sp10p_ar[t1], label=r'Fed-MoM-UCB, $s(p) = 10p$', **sty1, marker='*', color = sns.color_palette(cp)[0])
plt.fill_between(t1, fmomucb_sce_1_sp10p_ar[t1] - b*fmomucb_sce_1_sp10p_sr[t1], fmomucb_sce_1_sp10p_ar[t1] + b*fmomucb_sce_1_sp10p_sr[t1], color = sns.color_palette(cp)[0], alpha=0.2)
plt.plot(t1, fmomucb_sce_1_sp2_to_p_ar[t1], label=r'Fed-MoM-UCB, $s(p) = 2^p$', **sty1, marker='d', color = sns.color_palette(cp)[1])
plt.fill_between(t1, fmomucb_sce_1_sp2_to_p_ar[t1] - b*fmomucb_sce_1_sp2_to_p_sr[t1], fmomucb_sce_1_sp2_to_p_ar[t1] + b*fmomucb_sce_1_sp2_to_p_sr[t1], color = sns.color_palette(cp)[1], alpha=0.2)
plt.plot(t1, fucb_sce_1_sp10p_mp10_ar[t1], label=r'Fed2-UCB, $s(p) = 10p$, $M(p) = 10$', **sty1, marker='s', color = sns.color_palette(cp)[6])
plt.fill_between(t1, fucb_sce_1_sp10p_mp10_ar[t1] - b*fucb_sce_1_sp10p_mp10_sr[t1], fucb_sce_1_sp10p_mp10_ar[t1] + b*fucb_sce_1_sp10p_mp10_sr[t1],color = sns.color_palette(cp)[6], alpha=0.2)
plt.plot(t1, fucb_sce_1_sp10p_mp2_to_p_ar[t1], label=r'Fed2-UCB, $s(p) = 10p$, $M(p) = 2^p$', **sty1, color = sns.color_palette(cp)[2])
plt.fill_between(t1, fucb_sce_1_sp10p_mp2_to_p_ar[t1] - b*fucb_sce_1_sp10p_mp2_to_p_sr[t1], fucb_sce_1_sp10p_mp2_to_p_ar[t1] + b*fucb_sce_1_sp10p_mp2_to_p_sr[t1],color = sns.color_palette(cp)[2], alpha=0.2)
plt.legend(facecolor='white', framealpha=1, fontsize=12)
plt.xlabel('t')
plt.ylabel('Cumulative regret')
plt.savefig('./figures/sce1_reg_zoom.pdf')
plt.show()

##########################################################################################

plt.figure(figsize=fs)
plt.ticklabel_format(axis="y", style="sci", scilimits=(1,0))
plt.ticklabel_format(axis="x", style="sci", scilimits=(1,0))
plt.plot(t1, fmomucb_sce_1_sp10p_nocomm_ar[t1], label=r'Fed-MoM-UCB, $s(p) = 10p$', **sty1, marker='*', color = sns.color_palette(cp)[0])
plt.fill_between(t1, fmomucb_sce_1_sp10p_nocomm_ar[t1] - b*fmomucb_sce_1_sp10p_nocomm_sr[t1], fmomucb_sce_1_sp10p_nocomm_ar[t1] + b*fmomucb_sce_1_sp10p_nocomm_sr[t1], color = sns.color_palette(cp)[0], alpha=0.2)
plt.plot(t1, fmomucb_sce_1_sp2_to_p_nocomm_ar[t1], label=r'Fed-MoM-UCB, $s(p) = 2^p$', **sty1, marker='o', color = sns.color_palette(cp)[1])
plt.fill_between(t1, fmomucb_sce_1_sp2_to_p_nocomm_ar[t1] - b*fmomucb_sce_1_sp2_to_p_nocomm_sr[t1], fmomucb_sce_1_sp2_to_p_nocomm_ar[t1] + b*fmomucb_sce_1_sp2_to_p_nocomm_sr[t1], color = sns.color_palette(cp)[1], alpha=0.2)
plt.plot(t1, fucb_sce_1_sp10p_mp10_nocomm_ar[t1], label=r'Fed2-UCB, $s(p) = 10p$, $M(p) = 10$', **sty1, marker='s', color = sns.color_palette(cp)[6])
plt.fill_between(t1, fucb_sce_1_sp10p_mp10_nocomm_ar[t1] - b*fucb_sce_1_sp10p_mp10_nocomm_sr[t1], fucb_sce_1_sp10p_mp10_nocomm_ar[t1] + b*fucb_sce_1_sp10p_mp10_nocomm_sr[t1],color = sns.color_palette(cp)[6], alpha=0.2)
plt.plot(t1, fucb_sce_1_sp10p_mp2_to_p_nocomm_ar[t1], label=r'Fed2-UCB, $s(p) = 10p$, $M(p) = 2^p$', **sty1, color = sns.color_palette(cp)[2])
plt.fill_between(t1, fucb_sce_1_sp10p_mp2_to_p_nocomm_ar[t1] - b*fucb_sce_1_sp10p_mp2_to_p_nocomm_sr[t1], fucb_sce_1_sp10p_mp2_to_p_nocomm_ar[t1] + b*fucb_sce_1_sp10p_mp2_to_p_nocomm_sr[t1],color = sns.color_palette(cp)[2], alpha=0.2)
plt.legend(facecolor='white', framealpha=1, fontsize=12)
plt.xlabel('t')
plt.ylabel('Cumulative regret without communication cost')
plt.gca().set_ylim(bottom = -5e3, top=1e5)
plt.savefig('./figures/sce1_nocomm_reg.pdf')
plt.show()

##########################################################################################

plt.figure(figsize=fs)
plt.ticklabel_format(axis="y", style="sci", scilimits=(1,0))
plt.ticklabel_format(axis="x", style="sci", scilimits=(1,0))
plt.plot(t2, fmomucb_sce_1_sp10p_comm_ar[t2], label=r'Fed-MoM-UCB, $s(p) = 10p$', **sty2, marker='*', color = sns.color_palette(cp)[0], LineWidth=lw)
plt.fill_between(t2, fmomucb_sce_1_sp10p_comm_ar[t2] - b*fmomucb_sce_1_sp10p_comm_sr[t2], fmomucb_sce_1_sp10p_comm_ar[t2] + b*fmomucb_sce_1_sp10p_comm_sr[t2], color = sns.color_palette(cp)[0], alpha=0.2)
plt.plot(t2, fmomucb_sce_1_sp2_to_p_comm_ar[t2], label=r'Fed-MoM-UCB, $s(p) = 2^p$', **sty2, marker='o', color = sns.color_palette(cp)[1], LineWidth=lw)
plt.fill_between(t2, fmomucb_sce_1_sp2_to_p_comm_ar[t2] - b*fmomucb_sce_1_sp2_to_p_comm_sr[t2], fmomucb_sce_1_sp2_to_p_comm_ar[t2] + b*fmomucb_sce_1_sp2_to_p_comm_sr[t2], color = sns.color_palette(cp)[1], alpha=0.2)
plt.plot(t2, fucb_sce_1_sp10p_mp10_comm_ar[t2], label=r'Fed2-UCB, $s(p) = 10p$, $M(p) = 10$', **sty2, marker='s', color = sns.color_palette(cp)[6], LineWidth=lw)
plt.fill_between(t2, fucb_sce_1_sp10p_mp10_comm_ar[t2] - b*fucb_sce_1_sp10p_mp10_comm_sr[t2], fucb_sce_1_sp10p_mp10_comm_ar[t2] + b*fucb_sce_1_sp10p_mp10_comm_sr[t2], color = sns.color_palette(cp)[6], alpha=0.2)
plt.plot(t2, fucb_sce_1_sp10p_mp2_to_p_comm_ar[t2], label=r'Fed2-UCB, $s(p) = 10p$, $M(p) = 2^p$', **sty2, color = sns.color_palette(cp)[2], LineWidth=lw)
plt.fill_between(t2, fucb_sce_1_sp10p_mp2_to_p_comm_ar[t2] - b*fucb_sce_1_sp10p_mp2_to_p_comm_sr[t2], fucb_sce_1_sp10p_mp2_to_p_comm_ar[t2] + b*fucb_sce_1_sp10p_mp2_to_p_comm_sr[t2], color = sns.color_palette(cp)[2], alpha=0.2)
plt.legend(loc='upper left', facecolor='white', framealpha=1, fontsize=12)
plt.xlabel('t')
plt.ylabel('Communication cost')
plt.gca().set_ylim(bottom=-1e4, top=2e5)
plt.savefig('./figures/sce1_comm_reg.pdf')
plt.show()

##########################################################################################

print('Fed2-UCB Number of Clients M(p) = 10, s(p) = 10p: {} +- {}'.format(np.mean(fucb_sce_1_sp10p_mp10_M_arr), np.std(fucb_sce_1_sp10p_mp10_M_arr)))
print('Fed2-UCB Number of Clients M(p) = 2^p, s(p) = 10p: {} +- {}'.format(np.mean(fucb_sce_1_sp10p_mp2_to_p_M_arr), np.std(fucb_sce_1_sp10p_mp2_to_p_M_arr)))
print('FED-MoM-UCB Number of Clients: {} +- {}'.format(89, 0))  


#####################################################      SCENARIO 2       #####################################################

fmomucb_sce_2_v1_ar = np.load('./np_arrays/mean_fmomucb_sce_2_regret_sp10p_map_Arithmetic_lam_0.2.npy')
fmomucb_sce_2_v1_sr = np.load('./np_arrays/std_fmomucb_sce_2_regret_sp10p_map_Arithmetic_lam_0.2.npy')

fmomucb_sce_2_v1_comm_ar = np.load('./np_arrays/mean_fmomucb_sce_2_comm_regret_sp10p_map_Arithmetic_lam_0.2.npy')
fmomucb_sce_2_v1_comm_sr = np.load('./np_arrays/std_fmomucb_sce_2_comm_regret_sp10p_map_Arithmetic_lam_0.2.npy')

fmomucb_sce_2_v1_nocomm_ar = np.load('./np_arrays/mean_fmomucb_sce_2_nocomm_regret_sp10p_map_Arithmetic_lam_0.2.npy')
fmomucb_sce_2_v1_nocomm_sr = np.load('./np_arrays/std_fmomucb_sce_2_nocomm_regret_sp10p_map_Arithmetic_lam_0.2.npy')

##########################################################################################

fmomucb_sce_2_v2_ar = np.load('./np_arrays/mean_fmomucb_sce_2_regret_sp10p_map_Arithmetic_lam_0.25.npy')
fmomucb_sce_2_v2_sr = np.load('./np_arrays/std_fmomucb_sce_2_regret_sp10p_map_Arithmetic_lam_0.25.npy')

fmomucb_sce_2_v2_comm_ar = np.load('./np_arrays/mean_fmomucb_sce_2_comm_regret_sp10p_map_Arithmetic_lam_0.25.npy')
fmomucb_sce_2_v2_comm_sr = np.load('./np_arrays/std_fmomucb_sce_2_comm_regret_sp10p_map_Arithmetic_lam_0.25.npy')

fmomucb_sce_2_v2_nocomm_ar = np.load('./np_arrays/mean_fmomucb_sce_2_nocomm_regret_sp10p_map_Arithmetic_lam_0.25.npy')
fmomucb_sce_2_v2_nocomm_sr = np.load('./np_arrays/std_fmomucb_sce_2_nocomm_regret_sp10p_map_Arithmetic_lam_0.25.npy')

##########################################################################################

fucb_sce_2_v1_ar =  np.load('./np_arrays/mean_fucb_sce_2_regret_sp10p_mp20_lam_0.2.npy')
fucb_sce_2_v1_sr =  np.load('./np_arrays/std_fucb_sce_2_regret_sp10p_mp20_lam_0.2.npy')

fucb_sce_2_v1_comm_ar = np.load('./np_arrays/mean_fucb_sce_2_comm_regret_sp10p_mp20_lam_0.2.npy')
fucb_sce_2_v1_comm_sr = np.load('./np_arrays/std_fucb_sce_2_comm_regret_sp10p_mp20_lam_0.2.npy')

fucb_sce_2_v1_nocomm_ar = np.load('./np_arrays/mean_fucb_sce_2_nocomm_regret_sp10p_mp20_lam_0.2.npy')
fucb_sce_2_v1_nocomm_sr = np.load('./np_arrays/std_fucb_sce_2_nocomm_regret_sp10p_mp20_lam_0.2.npy')

fucb_sce_2_v1_M_arr = np.load('./np_arrays/fucb_sce_2_M_arr_sp10p_mp20_lam_0.2.npy')

##########################################################################################

fucb_sce_2_v2_ar = np.load('./np_arrays/mean_fucb_sce_2_regret_sp10p_mp20_lam_0.25.npy')
fucb_sce_2_v2_sr = np.load('./np_arrays/std_fucb_sce_2_regret_sp10p_mp20_lam_0.25.npy')

fucb_sce_2_v2_comm_ar = np.load('./np_arrays/mean_fucb_sce_2_comm_regret_sp10p_mp20_lam_0.25.npy')
fucb_sce_2_v2_comm_sr = np.load('./np_arrays/std_fucb_sce_2_comm_regret_sp10p_mp20_lam_0.25.npy')

fucb_sce_2_v2_nocomm_ar = np.load('./np_arrays/mean_fucb_sce_2_nocomm_regret_sp10p_mp20_lam_0.25.npy')
fucb_sce_2_v2_nocomm_sr = np.load('./np_arrays/std_fucb_sce_2_nocomm_regret_sp10p_mp20_lam_0.25.npy')

fucb_sce_2_v2_M_arr = np.load('./np_arrays/fucb_sce_2_M_arr_sp10p_mp20_lam_0.25.npy')

##########################################################################################

T = 100000
b = 2

t1 = [*range(0,T,int(T/2000))]
t2 = [*range(0,T,int(T/2000))]

i1 = [*range(0,len(t1),int(len(t1)/10))]
i2 = [*range(0,len(t1),int(len(t1)/10))]

sty1 = {
    'fillstyle': 'none',
    'linewidth': 1.0,
    'markevery': i1}

sty2 = {
    'fillstyle': 'none',
    'linewidth': 1.0,
    'markevery': i2}

print('SCENARIO 2 PLOTS\n', '*'*50)

plt.figure(figsize=fs)
plt.ticklabel_format(axis="y", style="sci", scilimits=(1,0))
plt.ticklabel_format(axis="x", style="sci", scilimits=(1,0))
plt.plot(t1, fmomucb_sce_2_v1_ar[t1], label=r'Fed-MoM-UCB, $\lambda = 0.2$', **sty1, marker='*', color = sns.color_palette(cp)[0])
plt.fill_between(t1, fmomucb_sce_2_v1_ar[t1] - b*fmomucb_sce_2_v1_sr[t1], fmomucb_sce_2_v1_ar[t1] + b*fmomucb_sce_2_v1_sr[t1], color = sns.color_palette(cp)[0], alpha=0.2)
plt.plot(t1, fmomucb_sce_2_v2_ar[t1], label=r'Fed-MoM-UCB, $\lambda = 0.25$', **sty1, marker='o', color = sns.color_palette(cp)[1])
plt.fill_between(t1, fmomucb_sce_2_v2_ar[t1] - b*fmomucb_sce_2_v2_sr[t1], fmomucb_sce_2_v2_ar[t1] + b*fmomucb_sce_2_v2_sr[t1], color = sns.color_palette(cp)[1], alpha=0.2)
plt.plot(t1, fucb_sce_2_v1_ar[t1], label=r'Fed2-UCB, $\lambda = 0.2$', **sty1, marker='s', color = sns.color_palette(cp)[6])
plt.fill_between(t1, fucb_sce_2_v1_ar[t1] - b*fucb_sce_2_v1_sr[t1], fucb_sce_2_v1_ar[t1] + b*fucb_sce_2_v1_sr[t1],color = sns.color_palette(cp)[6], alpha=0.2)
plt.plot(t1, fucb_sce_2_v2_ar[t1], label=r'Fed2-UCB, $\lambda = 0.25$', **sty1, color = sns.color_palette(cp)[2])
plt.fill_between(t1, fucb_sce_2_v2_ar[t1] - b*fucb_sce_2_v2_sr[t1], fucb_sce_2_v2_ar[t1] + b*fucb_sce_2_v2_sr[t1],color = sns.color_palette(cp)[2], alpha=0.2)
plt.xlabel('t') 
plt.ylabel('Cumulative regret')
plt.legend(facecolor='white', framealpha=1, fontsize=12)
plt.ylim(bottom = -5e4, top = 1e6)
plt.savefig('./figures/sce2_reg.pdf')
plt.show()

##########################################################################################

plt.figure(figsize=fs)
plt.ticklabel_format(axis="y", style="sci", scilimits=(1,0))
plt.ticklabel_format(axis="x", style="sci", scilimits=(1,0))
plt.plot(t1, fmomucb_sce_2_v1_ar[t1], label=r'Fed-MoM-UCB, $\lambda = 0.2$', **sty1, marker='*', color = sns.color_palette(cp)[0])
plt.fill_between(t1, fmomucb_sce_2_v1_ar[t1] - b*fmomucb_sce_2_v1_sr[t1], fmomucb_sce_2_v1_ar[t1] + b*fmomucb_sce_2_v1_sr[t1], color = sns.color_palette(cp)[0], alpha=0.2)
plt.plot(t1, fmomucb_sce_2_v2_ar[t1], label=r'Fed-MoM-UCB, $\lambda = 0.25$', **sty1, marker='o', color = sns.color_palette(cp)[1])
plt.fill_between(t1, fmomucb_sce_2_v2_ar[t1] - b*fmomucb_sce_2_v2_sr[t1], fmomucb_sce_2_v2_ar[t1] + b*fmomucb_sce_2_v2_sr[t1], color = sns.color_palette(cp)[1], alpha=0.2)
plt.plot(t1, fucb_sce_2_v1_ar[t1], label=r'Fed2-UCB, $\lambda = 0.2$', **sty1, marker='s', color = sns.color_palette(cp)[6])
plt.fill_between(t1, fucb_sce_2_v1_ar[t1] - b*fucb_sce_2_v1_sr[t1], fucb_sce_2_v1_ar[t1] + b*fucb_sce_2_v1_sr[t1],color = sns.color_palette(cp)[6], alpha=0.2)
plt.plot(t1, fucb_sce_2_v2_ar[t1], label=r'Fed2-UCB, $\lambda = 0.25$', **sty1, color = sns.color_palette(cp)[2])
plt.fill_between(t1, fucb_sce_2_v2_ar[t1] - b*fucb_sce_2_v2_sr[t1], fucb_sce_2_v2_ar[t1] + b*fucb_sce_2_v2_sr[t1],color = sns.color_palette(cp)[2], alpha=0.2)
plt.xlabel('t') 
plt.ylabel('Cumulative regret')
plt.ylim(bottom = -2e3, top = 5e4)
plt.legend(facecolor='white', framealpha=1, fontsize=12)
plt.savefig('./figures/sce2_reg_zoom.pdf')
plt.show()

##########################################################################################

plt.figure(figsize=fs)
plt.ticklabel_format(axis="y", style="sci", scilimits=(1,0))
plt.ticklabel_format(axis="x", style="sci", scilimits=(1,0))
plt.plot(t1, fmomucb_sce_2_v1_nocomm_ar[t1], label=r'Fed-MoM-UCB, $\lambda = 0.2$', **sty1, marker='*', color = sns.color_palette(cp)[0])
plt.fill_between(t1, fmomucb_sce_2_v1_nocomm_ar[t1] - b*fmomucb_sce_2_v1_nocomm_sr[t1], fmomucb_sce_2_v1_nocomm_ar[t1] + b*fmomucb_sce_2_v1_nocomm_sr[t1], color = sns.color_palette(cp)[0], alpha=0.2)
plt.plot(t1, fmomucb_sce_2_v2_nocomm_ar[t1], label=r'Fed-MoM-UCB, $\lambda = 0.25$', **sty1, marker='o', color = sns.color_palette(cp)[1])
plt.fill_between(t1, fmomucb_sce_2_v2_nocomm_ar[t1] - b*fmomucb_sce_2_v2_nocomm_sr[t1], fmomucb_sce_2_v2_nocomm_ar[t1] + b*fmomucb_sce_2_v2_nocomm_sr[t1], color = sns.color_palette(cp)[1], alpha=0.2)
plt.plot(t1, fucb_sce_2_v1_nocomm_ar[t1], label=r'Fed2-UCB, $\lambda = 0.2$', **sty1, marker='s', color = sns.color_palette(cp)[6])
plt.fill_between(t1, fucb_sce_2_v1_nocomm_ar[t1] - b*fucb_sce_2_v1_nocomm_sr[t1], fucb_sce_2_v1_nocomm_ar[t1] + b*fucb_sce_2_v1_nocomm_sr[t1],color = sns.color_palette(cp)[6], alpha=0.2)
plt.plot(t1, fucb_sce_2_v2_nocomm_ar[t1], label=r'Fed2-UCB, $\lambda = 0.25$', **sty1, color = sns.color_palette(cp)[2])
plt.fill_between(t1, fucb_sce_2_v2_nocomm_ar[t1] - b*fucb_sce_2_v2_nocomm_sr[t1], fucb_sce_2_v2_nocomm_ar[t1] + b*fucb_sce_2_v2_nocomm_sr[t1],color = sns.color_palette(cp)[2], alpha=0.2)
plt.xlabel('t')
plt.ylabel('Cumulative regret without communication cost')
plt.gca().set_ylim(bottom = -2e3, top=5e4)
plt.legend(facecolor='white', framealpha=1, fontsize=12)
plt.savefig('./figures/sce2_nocomm_reg.pdf')
plt.show()

##########################################################################################

plt.figure(figsize=fs)
plt.ticklabel_format(axis="y", style="sci", scilimits=(1,0))
plt.ticklabel_format(axis="x", style="sci", scilimits=(1,0))
plt.plot(t1, fmomucb_sce_2_v1_comm_ar[t1], label=r'Fed-MoM-UCB, $\lambda = 0.2$', **sty1, marker='*', color = sns.color_palette(cp)[0])
plt.fill_between(t1, fmomucb_sce_2_v1_comm_ar[t1] - b*fmomucb_sce_2_v1_comm_sr[t1], fmomucb_sce_2_v1_comm_ar[t1] + b*fmomucb_sce_2_v1_comm_sr[t1], color = sns.color_palette(cp)[0], alpha=0.2)
plt.plot(t1, fmomucb_sce_2_v2_comm_ar[t1], label=r'Fed-MoM-UCB, $\lambda = 0.25$', **sty1, marker='o', color = sns.color_palette(cp)[1])
plt.fill_between(t1, fmomucb_sce_2_v2_comm_ar[t1] - b*fmomucb_sce_2_v2_comm_sr[t1], fmomucb_sce_2_v2_comm_ar[t1] + b*fmomucb_sce_2_v2_comm_sr[t1], color = sns.color_palette(cp)[1], alpha=0.2)
plt.plot(t1, fucb_sce_2_v1_comm_ar[t1], label=r'Fed2-UCB, $\lambda = 0.2$', **sty1, marker='s', color = sns.color_palette(cp)[6])
plt.fill_between(t1, fucb_sce_2_v1_comm_ar[t1] - b*fucb_sce_2_v1_comm_sr[t1], fucb_sce_2_v1_comm_ar[t1] + b*fucb_sce_2_v1_comm_sr[t1],color = sns.color_palette(cp)[6], alpha=0.2)
plt.plot(t1, fucb_sce_2_v2_comm_ar[t1], label=r'Fed2-UCB, $\lambda = 0.25$', **sty1, color = sns.color_palette(cp)[2])
plt.fill_between(t1, fucb_sce_2_v2_comm_ar[t1] - b*fucb_sce_2_v2_comm_sr[t1], fucb_sce_2_v2_comm_ar[t1] + b*fucb_sce_2_v2_comm_sr[t1],color = sns.color_palette(cp)[2], alpha=0.2)
plt.xlabel('t')
plt.ylabel('Communication cost')
plt.gca().set_ylim(bottom = -1e3, top=3.5e4)
plt.legend(loc='upper left', facecolor='white', framealpha=1, fontsize=12)
plt.savefig('./figures/sce2_comm_reg.pdf')
plt.show()

##########################################################################################

print('Fed2-UCB Number of Clients lam = 0.2: {} +- {}'.format(np.mean(fucb_sce_2_v1_M_arr), np.std(fucb_sce_2_v1_M_arr)))
print('Fed2-UCB Number of Clients lam = 0.25: {} +- {}'.format(np.mean(fucb_sce_2_v2_M_arr), np.std(fucb_sce_2_v2_M_arr)))
print('FED-MoM-UCB Number lam = 0.2: {} +- {}'.format(92, 0))  
print('FED-MoM-UCB Number lam = 0.25: {} +- {}'.format(96, 0))  

#####################################################      SCENARIO 3        #####################################################


fmomucb_sce_3_sp10p_ar_geo = np.load('./np_arrays/mean_fmomucb_sce_3_regret_sp10p_map_Geometric.npy')
fmomucb_sce_3_sp10p_sr_geo = np.load('./np_arrays/std_fmomucb_sce_3_regret_sp10p_map_Geometric.npy')

fmomucb_sce_3_sp10p_comm_ar_geo = np.load('./np_arrays/mean_fmomucb_sce_3_comm_regret_sp10p_map_Geometric.npy')
fmomucb_sce_3_sp10p_comm_sr_geo = np.load('./np_arrays/std_fmomucb_sce_3_comm_regret_sp10p_map_Geometric.npy')

fmomucb_sce_3_sp10p_nocomm_ar_geo = np.load('./np_arrays/mean_fmomucb_sce_3_comm_regret_sp10p_map_Geometric.npy')
fmomucb_sce_3_sp10p_nocomm_sr_geo = np.load('./np_arrays/std_fmomucb_sce_3_comm_regret_sp10p_map_Geometric.npy')

##########################################################################################

fmomucb_sce_3_sp10p_ar_har = np.load('./np_arrays/mean_fmomucb_sce_3_regret_sp10p_map_Harmonic.npy')
fmomucb_sce_3_sp10p_sr_har = np.load('./np_arrays/std_fmomucb_sce_3_regret_sp10p_map_Harmonic.npy')

fmomucb_sce_3_sp10p_comm_ar_har = np.load('./np_arrays/mean_fmomucb_sce_3_comm_regret_sp10p_map_Harmonic.npy')
fmomucb_sce_3_sp10p_comm_sr_har = np.load('./np_arrays/std_fmomucb_sce_3_comm_regret_sp10p_map_Harmonic.npy')

fmomucb_sce_3_sp10p_nocomm_ar_har = np.load('./np_arrays/mean_fmomucb_sce_3_comm_regret_sp10p_map_Harmonic.npy')
fmomucb_sce_3_sp10p_nocomm_sr_har = np.load('./np_arrays/std_fmomucb_sce_3_comm_regret_sp10p_map_Harmonic.npy')


##########################################################################################

fmomucb_sce_3_sp10p_ar_poly = np.load('./np_arrays/mean_fmomucb_sce_3_regret_sp10p_map_Polynomial.npy')
fmomucb_sce_3_sp10p_sr_poly = np.load('./np_arrays/std_fmomucb_sce_3_regret_sp10p_map_Polynomial.npy')

fmomucb_sce_3_sp10p_comm_ar_poly = np.load('./np_arrays/mean_fmomucb_sce_3_comm_regret_sp10p_map_Polynomial.npy')
fmomucb_sce_3_sp10p_comm_sr_poly = np.load('./np_arrays/std_fmomucb_sce_3_comm_regret_sp10p_map_Polynomial.npy')

fmomucb_sce_3_sp10p_nocomm_ar_poly = np.load('./np_arrays/mean_fmomucb_sce_3_comm_regret_sp10p_map_Polynomial.npy')
fmomucb_sce_3_sp10p_nocomm_sr_poly = np.load('./np_arrays/std_fmomucb_sce_3_comm_regret_sp10p_map_Polynomial.npy')


##########################################################################################

T = 20000
b = 2

t1 = [*range(0,T,int(T/400))]
t2 = [*range(0,T,int(T/400))]

i1 = [*range(0,len(t1),int(len(t1)/10))]
i2 = [*range(0,len(t1),int(len(t1)/10))]

sty1 = {
    'fillstyle': 'none',
    'linewidth': 1.0,
    'markevery': i1}

sty2 = {
    'fillstyle': 'none',
    'linewidth': 1.0,
    'markevery': i2}

print('SCENARIO 3 PLOTS\n', '*'*50)

plt.figure()
plt.ticklabel_format(axis="y", style="sci", scilimits=(1,0))
plt.ticklabel_format(axis="x", style="sci", scilimits=(1,0))
plt.plot(t1, fmomucb_sce_3_sp10p_ar_geo[t1], label=r'Fed-MoM-UCB, Geometric $\alpha(\lambda)$', **sty1, marker='*', color = sns.color_palette(cp)[0])
plt.fill_between(t1, fmomucb_sce_3_sp10p_ar_geo[t1] - b*fmomucb_sce_3_sp10p_sr_geo[t1], fmomucb_sce_3_sp10p_ar_geo[t1] + b*fmomucb_sce_3_sp10p_sr_geo[t1], color = sns.color_palette(cp)[0], alpha=0.2)
plt.plot(t1, fmomucb_sce_3_sp10p_ar_har[t1], label=r'Fed-MoM-UCB, Harmonic $\alpha(\lambda)$', **sty1, marker='o', color = sns.color_palette(cp)[1])
plt.fill_between(t1, fmomucb_sce_3_sp10p_ar_har[t1] - b*fmomucb_sce_3_sp10p_sr_har[t1], fmomucb_sce_3_sp10p_ar_har[t1] + b*fmomucb_sce_3_sp10p_sr_har[t1], color = sns.color_palette(cp)[1], alpha=0.2)
plt.plot(t1, fmomucb_sce_3_sp10p_ar_poly[t1], label=r'Fed-MoM-UCB, Polynomial $\alpha(\lambda)$', **sty1, marker='s', color = sns.color_palette(cp)[6])
plt.fill_between(t1, fmomucb_sce_3_sp10p_ar_poly[t1] - b*fmomucb_sce_3_sp10p_sr_poly[t1], fmomucb_sce_3_sp10p_ar_poly[t1] + b*fmomucb_sce_3_sp10p_sr_poly[t1],color = sns.color_palette(cp)[6], alpha=0.2)
plt.legend(facecolor='white', framealpha=1, fontsize=11, loc=[0.25,0.55])
plt.xlabel('t') 
plt.ylabel('Cumulative regret')
plt.savefig('./figures/sce3_reg.pdf')
plt.show()

##########################################################################################

plt.figure()
plt.ticklabel_format(axis="y", style="sci", scilimits=(1,0))
plt.ticklabel_format(axis="x", style="sci", scilimits=(1,0))
plt.plot(t1, fmomucb_sce_3_sp10p_comm_ar_geo[t1], label=r'Fed-MoM-UCB, Geometric $\alpha(\lambda)$', **sty1, marker='*', color = sns.color_palette(cp)[0])
plt.fill_between(t1, fmomucb_sce_3_sp10p_comm_ar_geo[t1] - b*fmomucb_sce_3_sp10p_comm_sr_geo[t1], fmomucb_sce_3_sp10p_comm_ar_geo[t1] + b*fmomucb_sce_3_sp10p_comm_sr_geo[t1], color = sns.color_palette(cp)[0], alpha=0.2)
plt.plot(t1, fmomucb_sce_3_sp10p_comm_ar_har[t1], label=r'Fed-MoM-UCB, Harmonic $\alpha(\lambda)$', **sty1, marker='o', color = sns.color_palette(cp)[1])
plt.fill_between(t1, fmomucb_sce_3_sp10p_comm_ar_har[t1] - b*fmomucb_sce_3_sp10p_comm_sr_har[t1], fmomucb_sce_3_sp10p_comm_ar_har[t1] + b*fmomucb_sce_3_sp10p_comm_sr_har[t1], color = sns.color_palette(cp)[1], alpha=0.2)
plt.plot(t1, fmomucb_sce_3_sp10p_comm_ar_poly[t1], label=r'Fed-MoM-UCB, Polynomial $\alpha(\lambda)$', **sty1, marker='s', color = sns.color_palette(cp)[6])
plt.fill_between(t1, fmomucb_sce_3_sp10p_comm_ar_poly[t1] - b*fmomucb_sce_3_sp10p_comm_sr_poly[t1], fmomucb_sce_3_sp10p_comm_ar_poly[t1] + b*fmomucb_sce_3_sp10p_comm_sr_poly[t1],color = sns.color_palette(cp)[6], alpha=0.2)
plt.xlabel('t') 
plt.ylabel('Communication cost')
plt.savefig('./figures/sce3_comm.pdf')
plt.show()

##########################################################################################

print('FED-MoM-UCB Number of Clients Geometric: {} +- {}'.format(123, 0))  
print('FED-MoM-UCB Number of Clients Harmonic: {} +- {}'.format(193, 0))  
print('FED-MoM-UCB Number of Clients Polynomial: {} +- {}'.format(362, 0))  