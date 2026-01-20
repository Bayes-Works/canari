import numpy as np
import matplotlib.pylab as plt
from scipy.stats import norm, beta

## Analysis configuration
nb_obs = 100         #Number of observations
nb_sim = 100        #Number of simulations
nb_mcs = 100        #Number of MC samples
sequential_processing = True
sequential_conditionnal = True

## Model hyperparameters
mu_real = 0         #Mean of the real model
var_AB = 6          #Variance of the models Note: s_A = s_B = s_real = 1
mu_A = -1           #Mean of model A
var_mu = 0.001       #Epistemic variance associated with mu_A & mu_B
pr_A = 0.5          #Prior probability of model A
mu_B = 0.01          #Mean of model B
pr_B = 1-pr_A       #Prior probability of model A
discount = 1

## Initialization
prA_post = np.zeros(nb_sim)  #classical BMS
prAc_post = np.zeros(nb_sim) #conditionnal BMS
a_post = np.zeros(nb_sim)    #\#obs model A (Beta PDF)
b_post = np.zeros(nb_sim)    #\#obs model B (Beta PDF)

## Loop
for i in range(nb_sim):
    ## BMS
    y = np.random.normal(loc=mu_real,scale=np.sqrt(var_AB),size=nb_obs)                  #Batch Observations
    fA_y = np.exp(np.sum(np.log(norm.pdf(y,loc=mu_A,scale=np.sqrt(var_mu+var_AB)))))     #joint likelihood of model A
    fB_y = np.exp(np.sum(np.log(norm.pdf(y,loc=mu_B,scale=np.sqrt(var_mu+var_AB)))))     #joint likelihood of model B
    prA_post[i] = (fA_y * pr_A) / (fA_y * pr_A + fB_y * pr_B)                            #Posterior probability of model A

    ## BetaBMS & CBMS
    if sequential_processing:
        prAc_post[i] = pr_A
        for j in range(nb_obs):
            prAc = 0                                                                     #initialize for the j-th observation
            if sequential_conditionnal: # BetaCBMS
                a = np.zeros(nb_mcs)
                b = np.zeros(nb_mcs)
                p = np.zeros(nb_mcs)
                for s in range(nb_mcs):
                    v = np.random.normal(loc=0,scale=np.sqrt(var_AB),size=1)
                    fAc_y = np.exp(np.log(norm.pdf(y[j],loc=mu_A+v,scale=np.sqrt(var_mu))))
                    fBc_y = np.exp(np.log(norm.pdf(y[j],loc=mu_B+v,scale=np.sqrt(var_mu))))
                    if fAc_y==0 or fBc_y==0:
                        if abs(mu_A+v-y[j]) < abs(mu_B+v-y[j]):
                            fAc_y = 1000
                            fBc_y = 1
                        elif abs(mu_A+v-y[j]) > abs(mu_B+v-y[j]):
                            fAc_y = 1
                            fBc_y = 1000
                        else:
                            fAc_y = 1
                            fBc_y = 1

                    a[s] = fAc_y / (fAc_y + fBc_y)  #Posterior probability of model A
                    b[s] = fBc_y / (fAc_y + fBc_y)  #Posterior probability of model B
                    prAc_post[i] = prAc/nb_mcs
                    a_post[i] += np.sum(a)/nb_mcs
                    b_post[i] += np.sum(b)/nb_mcs
                    a_post[i] *= discount
                    b_post[i] *= discount
            else: # BetaBMS
                fAc_y = np.exp(np.log(norm.pdf(y[j],loc=mu_A,scale=np.sqrt(var_mu+var_AB))))
                fBc_y = np.exp(np.log(norm.pdf(y[j],loc=mu_B,scale=np.sqrt(var_mu+var_AB))))
                a = fAc_y / (fAc_y + fBc_y)  #Posterior probability of model A
                b = fBc_y / (fAc_y + fBc_y)  #Posterior probability of model B
                a_post[i] += a
                b_post[i] += b
                a_post[i] *= discount
                b_post[i] *= discount

        prAc_post[i] = a_post[i]/(a_post[i]+b_post[i])

    else: #CBMS
        for s in range(nb_mcs):
            v = np.random.normal(loc=0,scale=np.sqrt(var_AB),size=nb_obs)
            fAc_y = np.exp(np.sum(np.log(norm.pdf(y,loc=mu_A+v,scale=np.sqrt(var_mu)))))
            fBc_y = np.exp(np.sum(np.log(norm.pdf(y,loc=mu_B+v,scale=np.sqrt(var_mu)))))
            prAc_post[i] += (fAc_y * pr_A) / (fAc_y * pr_A + fBc_y * pr_B)  #Posterior probability of model A (Conditional)
        prAc_post[i] /= nb_mcs

## Plotting
fig, axs = plt.subplots(3, 1,figsize=(6, 8.5))
x = np.linspace(-5,5,500)
fA = norm.pdf(x,loc=mu_A,scale=np.sqrt(var_mu+var_AB))
fB = norm.pdf(x,loc=mu_B,scale=np.sqrt(var_mu+var_AB))
fmuA = norm.pdf(x,loc=mu_A,scale=np.sqrt(var_mu))
fmuB = norm.pdf(x,loc=mu_B,scale=np.sqrt(var_mu))
fmuA *= max(fA)/max(fmuA)
fmuB *= max(fB)/max(fmuB)
axs[0].plot(x,fA,label='Model A')
axs[0].fill_between(x, fmuA, color='blue', alpha=0.3)
axs[0].plot(x,fB,label='Model B')
axs[0].fill_between(x, fmuB, color='orange', alpha=0.3)
axs[0].plot(y, np.zeros_like(y), '+')
axs[0].set_xlabel('x')
axs[0].set_ylabel('f(x|*)')
axs[0].legend()
axs[1].hist(prA_post, bins=100, range=(0,1))
axs[1].set_xlabel('Pr(A|y)')
axs[1].set_ylabel('#')
axs[1].set_xlim([0,1.01])
axs[1].set_xticks([0,0.5,1])
axs[2].hist(prAc_post, bins=100, range=(0,1))
axs[2].set_xlabel('Pr(A|y) (Conditionnal)')
axs[2].set_ylabel('#')
axs[2].set_xlim([0,1.01])
axs[2].set_xticks([0,0.5,1])
fig.tight_layout()
plt.show()

