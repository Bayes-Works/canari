import numpy as np
import matplotlib.pylab as plt
from scipy.stats import norm, beta

## Analysis configuration
nb_obs = 100        #Number of observations
nb_sim = 1          #Number of simulations

## Model hyperparameters
mu_real = 0         #Mean of the real model
var_AB = 1          #Variance of the models Note: s_A = s_B = s_real = 1
mu_A = -0.5         #Mean of model A
mu_B = 0.25          #Mean of model B
var_mu = 0.05       #Epistemic variance associated with mu_A & mu_B
discount_coeff = 1  #Discount coefficient \in(0,1) in order to model a non-stationary probability

## Initialization
prA_post = np.zeros(nb_sim)  #classical BMS
prAc_post = np.zeros(nb_sim) #conditionnal BMS
a_post = np.zeros(nb_sim)    #\#obs model A (Beta PDF)
b_post = np.zeros(nb_sim)    #\#obs model B (Beta PDF)

## Loop
for i in range(nb_sim):
    ## Standard BMS
    y = np.random.normal(loc=mu_real,scale=np.sqrt(var_AB),size=nb_obs)                  #Batch Observations
    fA_y = np.exp(np.sum(np.log(norm.pdf(y,loc=mu_A,scale=np.sqrt(var_mu+var_AB)))))     #joint likelihood of model A
    fB_y = np.exp(np.sum(np.log(norm.pdf(y,loc=mu_B,scale=np.sqrt(var_mu+var_AB)))))     #joint likelihood of model B
    prA_post[i] = fA_y / (fA_y+fB_y)                            #Posterior probability of model A

    ## Epistemic BMS
    for j in range(nb_obs):
        fAc_y = np.exp(np.log(norm.pdf(y[j],loc=mu_A,scale=np.sqrt(var_mu+var_AB))))
        fBc_y = np.exp(np.log(norm.pdf(y[j],loc=mu_B,scale=np.sqrt(var_mu+var_AB))))
        a = fAc_y / (fAc_y+fBc_y)  #Posterior probability of model A
        b = fBc_y / (fAc_y+fBc_y)  #Posterior probability of model B
        a_post[i] += a
        b_post[i] += b
        a_post[i] *= discount_coeff
        b_post[i] *= discount_coeff
    prAc_post[i] = a_post[i]/(a_post[i]+b_post[i])

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
axs[1].set_xlabel('Pr(A|y) -- BMS')
axs[1].set_ylabel('#')
axs[1].set_xlim([0,1.01])
axs[1].set_xticks([0,0.5,1])
if nb_sim>1:
    axs[2].hist(prAc_post, bins=100, range=(0,1))
    axs[2].set_ylabel('#')
else:
    p = np.linspace(0,1,200)
    fp = beta.pdf(p,a_post,b_post)
    axs[2].plot(p,fp)
    axs[2].vlines(a_post/(a_post+b_post),0,max(fp),color='red', label='$E[P|y_{1:i}]$')
    axs[2].set_ylabel('$f(p_1|y_{1:i})$')
    axs[2].legend()

axs[2].set_xlabel('Pr(A|y) -- EBMS')
axs[2].set_xlim([0,1.01])
axs[2].set_xticks([0,0.5,1])
fig.tight_layout()
plt.show()

