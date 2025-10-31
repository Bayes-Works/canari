import numpy as np
import matplotlib.pylab as plt
from scipy.stats import lognorm, norm
np.random.seed(300)
## Hidden states definition
nb_Z = 3          #Number of variables
s    = 1.5           #Prior std


mZ = [] #Initialize mean & std of hidden units
sZ = []
for i in range(nb_Z):
    mZ.append(np.random.normal(loc=-2,scale=2))
    sZ.append(s)
mZ[0] = 1
#sZ[0] = 5

mZ = np.array(mZ)
sZ = np.array(sZ)
s2Z = sZ**2

## MCS verification
nb_mcs = 1000000
Z_s = np.random.normal(loc=mZ,scale=sZ,size=(nb_mcs,nb_Z))
M_s = np.maximum(1E-6,Z_s)

Msum = sum(M_s.transpose())
Msum_s = Msum[:,np.newaxis]
Msum_inv_s = 1/Msum_s

A_s = np.divide(M_s,Msum_s)
lnM_s = np.log(M_s)
lnM_sum_s = np.log(Msum_s)
cA_s = lnM_s-lnM_sum_s
mA_s = np.mean(A_s,axis=0)
sA_s = np.std(A_s,axis=0)

cov_ZA_mcs = np.cov(Z_s,A_s,rowvar=False)
cov_Z0A_MCS = cov_ZA_mcs[0,nb_Z::]
cov_ZA_mcs = cov_ZA_mcs[0:nb_Z,nb_Z::]

mM_mcs = np.average(M_s,axis=0)
s2M_mcs = np.var(M_s,axis=0)
sM_mcs = np.sqrt(s2M_mcs)

mM_sum_mcs = np.average(Msum_s,axis=0)
s2M_sum_mcs = np.var(Msum_s,axis=0)

mM_sum_inv_mcs = np.average(1/Msum_s,axis=0)
s2M_sum_inv_mcs = np.var(1/Msum_s,axis=0)

mlnM_sum_mcs = np.average(lnM_sum_s,axis=0)
s2lnM_sum_mcs = np.var(lnM_sum_s,axis=0)

mlnM_mcs = np.average(lnM_s,axis=0)
s2lnM_mcs = np.var(lnM_s,axis=0)

mlnA_mcs = np.average(cA_s,axis=0)
s2lnA_mcs = np.var(cA_s,axis=0)

mA_mcs = np.average(A_s,axis=0)
s2A_mcs = np.var(A_s,axis=0)

cov_AM_mcs = np.cov(M_s,A_s,rowvar=False)
cov_AM_mcs = cov_AM_mcs[0:nb_Z,nb_Z::]

cov_cAlnM_mcs = np.cov(cA_s,lnM_s,rowvar=False)
cov_cAlnM_mcs = cov_cAlnM_mcs[0:nb_Z,nb_Z::]

cov_lnM_lnM_sum_mcs = np.cov(lnM_s,lnM_sum_s,rowvar=False)
cov_lnM_lnM_sum_mcs = cov_lnM_lnM_sum_mcs[nb_Z,0:nb_Z]

cov_M_M_sum_mcs = np.cov(M_s,Msum_s,rowvar=False)
cov_M_M_sum_mcs = cov_M_M_sum_mcs[0:nb_Z,0:nb_Z]

cov_M_M_sum_inv_mcs = np.cov(M_s,1/Msum_s,rowvar=False)
cov_M_M_sum_inv_mcs = cov_M_M_sum_inv_mcs[0:nb_Z,0:nb_Z]

## mReLU activation
cdfn = np.maximum(1E-20,norm.cdf(mZ/sZ))
pdfn = np.maximum(1E-20,norm.pdf(mZ/sZ))
mM = np.maximum(1E-20,sZ*pdfn + mZ * cdfn)
#s2M = mZ*sZ*pdfn + (mZ**2+s2Z) * cdfn - mM**2
s2M = np.maximum(1E-10,-mM**2 + 2*mM*mZ - mZ*sZ*pdfn + (s2Z - mZ**2) * cdfn)
sM = np.sqrt(s2M)
cov_ZM = s2Z * cdfn

## lnM = log(M_i)
s2lnM = np.log(1+np.minimum(10,s2M/mM**2))
slnM = np.sqrt(s2lnM)
mlnM = np.log(mM)-0.5*s2lnM
cov_M_lnM = s2lnM*mM

## \tilde{M} = sum(M_i)
mM_sum = sum(mM)
s2M_sum = sum(s2M)
sM_sum = np.sqrt(s2M_sum)
cov_M_M_sum = s2M #/ np.sqrt(2)#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! TEST

## ln\tilde{M} = log(\tilde{M}_i)
s2lnM_sum = np.log(1+s2M_sum/mM_sum**2)
slnM_sum = np.sqrt(s2lnM_sum)
mlnM_sum = np.log(mM_sum)-0.5*s2lnM_sum
cov_lnM_lnM_sum = np.log(1+cov_M_M_sum/mM/mM_sum)

## 1/\tilde{M} -> 1-ln\tilde{M}
s2lnM_sum_inv = s2lnM_sum
mlnM_sum_inv = 1 - mlnM_sum

mM_sum_inv = np.exp(mlnM_sum_inv+0.5*s2lnM_sum_inv)
s2M_sum_inv = mM_sum_inv**2*(np.exp(s2lnM_sum_inv)-1)
cov_M_M_sum_inv = (np.exp(cov_lnM_lnM_sum)-1) * mM_sum * mM_sum_inv

## \check{A}_i = lnM_i - ln\tilde{M}
mlnA = mlnM - mlnM_sum
s2lnA = s2lnM + s2lnM_sum - 2*cov_lnM_lnM_sum
slnA = np.sqrt(s2lnA)
cov_cAlnM = s2lnM - cov_lnM_lnM_sum
cov_cAlnM_sum = cov_lnM_lnM_sum - s2lnM_sum
cov_cAMj = cov_cAlnM_sum * mM_sum/s2M_sum * s2M[0]

## A_i = normal
mA = np.exp(mlnA+1/2*s2lnA)
################################################################################
mA = mA/np.sum(mA)
################################################################################

s2A = np.maximum(1E-10,mA**2*(np.exp(s2lnA)-1))
sA = np.sqrt(s2A)

cov_AM = np.maximum(0,(np.exp(cov_cAlnM)-1) * mA * mM)
print("cov_AM MCS: ",np.diag(cov_ZA_mcs))
print("cov_AM ANA: ",cov_AM)
print("\n")

cov_ZA_lin2 = cov_AM / s2M * cov_ZM

#cov_ZA = cov_AM / cdfn #1
#cov_ZA = cov_AM * s2Z*cdfn/s2M #2
cov_ZA = cov_AM * (1/cdfn + s2Z*cdfn/s2M) / 2 #(1+2)/2
################################################################################
cov_ZA = np.minimum(cov_ZA, sA*sZ)
################################################################################

cov_ZA_old = mA * (cov_cAlnM * mM) / s2M * cov_ZM
cov_ZjA = mA * cov_cAMj / s2M[0] * cov_ZM[0]
print("\n")

print("cov_AZ  MCS: ",np.diag(cov_ZA_mcs))
print ("cov_AZ   AN: ",cov_ZA)
print("cov_AZ LIN2: ",cov_ZA_lin2)
print("cov_AZ LIN1: ",cov_ZA_old)

print("\n")
print(cov_ZjA)
print(cov_ZA_mcs[0,:])

print("\n")
print("mM MCS: ",mM_mcs)
print("mM ANA: ",mM)
print("∆    %: ", np.average(100*np.abs((mM-mM_mcs)/mM_mcs)))
print("s2M MCS: ",s2M_mcs)
print("s2M ANA: ",s2M)
print("∆     %: ", np.average(100*np.abs((s2M-s2M_mcs)/s2M_mcs)))

print("\n")
print("mM_sum MCS: ",mM_sum_mcs)
print("mM_sum ANA: ",mM_sum)
print("∆    %: ", np.average(100*np.abs((mM_sum-mM_sum_mcs)/mM_sum_mcs)))
print("s2M_sum MCS: ",s2M_sum_mcs)
print("s2M_sum ANA: ",s2M_sum)
print("∆     %: ", np.average(100*np.abs((s2M_sum-s2M_sum_mcs)/s2M_sum_mcs)))

print("\n")
print("mM_sum_inv MCS: ",mM_sum_inv_mcs)
print("mM_sum_inv ANA: ",mM_sum_inv)
print("∆    %: ", np.average(100*np.abs((mM_sum_inv-mM_sum_inv_mcs)/mM_sum_inv_mcs)))
print("s2M_sum_inv MCS: ",s2M_sum_inv_mcs)
print("s2M_sum_inv ANA: ",s2M_sum_inv)
print("∆     %: ", np.average(100*np.abs((s2M_sum_inv-s2M_sum_inv_mcs)/s2M_sum_inv_mcs)))

print("\n")
print("mlnM_sum MCS: ",mlnM_sum_mcs)
print("mlnM_sum ANA: ",mlnM_sum)
print("∆    %: ", np.average(100*np.abs((mlnM_sum-mlnM_sum_mcs)/mlnM_sum_mcs)))
print("s2lnM_sum MCS: ",s2lnM_sum_mcs)
print("s2lnM_sum ANA: ",s2lnM_sum)
print("∆     %: ", np.average(100*np.abs((s2lnM_sum-s2lnM_sum_mcs)/s2lnM_sum_mcs)))

print("\n")
print("mlnM MCS: ",mlnM_mcs)
print("mlnM ANA: ",mlnM)
print("∆    %: ", np.average(100*np.abs((mlnM-mlnM_mcs)/mlnM_mcs)))
print("s2lnM MCS: ",s2lnM_mcs)
print("s2lnM ANA: ",s2lnM)
print("∆     %: ", np.average(100*np.abs((s2lnM-s2lnM_mcs)/s2lnM_mcs)))

print("\n")
print("mlnA MCS: ",mlnA_mcs)
print("mlnA ANA: ",mlnA)
print("∆    %: ", np.average(100*np.abs((mlnA-mlnA_mcs)/mlnA_mcs)))
print("s2lnA MCS: ",s2lnA_mcs)
print("s2lnA ANA: ",s2lnA)
print("∆     %: ", np.average(100*np.abs((s2lnA-s2lnA_mcs)/s2lnA_mcs)))

print("\n")
print("mA MCS: ",mA_mcs)
print("mA ANA: ",mA)
print("∆    %: ", np.average(100*np.abs((mA-mA_mcs)/mA_mcs)))
print("s2A MCS: ",s2A_mcs)
print("s2A ANA: ",s2A)
print("∆     %: ", np.average(100*np.abs((s2A-s2A_mcs)/s2A_mcs)))


print("\n")
print("cov_M_M_sum MCS: ",np.diag(cov_M_M_sum_mcs))
print("cov_M_M_sum ANA: ",cov_M_M_sum)
print("∆             %: ", np.average(100*np.abs((cov_M_M_sum-np.diag(cov_M_M_sum_mcs))/np.diag(cov_M_M_sum_mcs))))

print("\n")
print("cov_lnM_lnM_sum MCS: ",cov_lnM_lnM_sum_mcs)
print("cov_lnM_lnM_sum ANA: ",cov_lnM_lnM_sum)

print("\n")
print("cov_M_M_sum_inv MCS: ",np.diag(cov_M_M_sum_inv_mcs))
print("cov_M_M_sum_inv ANA: ",cov_M_M_sum_inv)
print("∆             %: ", np.average(100*np.abs((cov_M_M_sum_inv-np.diag(cov_M_M_sum_inv_mcs))/np.diag(cov_M_M_sum_inv_mcs))))

print("\n")
print("cov_cAlnM MCS: ",np.diag(cov_cAlnM_mcs))
print("cov_cAlnM ANA: ",cov_cAlnM)
print("\n")

## Plot results
fig = plt.figure( figsize=(10, 8))
subfig = fig.subfigures(1,3, width_ratios=[1.6,0.4,0.4])
fig.subplots_adjust(right=0.9)
ax0 = subfig[0].subplots(6, 1, sharex=True)
ax1 = subfig[1].subplots(6, 1, sharex=False)
ax2 = subfig[2].add_subplot(6,1,2)
subfig[0].subplots_adjust(hspace=0.4)
subfig[1].subplots_adjust(hspace=0.4)
subfig[2].subplots_adjust(hspace=0.4)

print('\nSum of mA = ', sum(mA),'\n')
delta = 0.3
lw = 1

for i in range(nb_Z):
    ax0[0].plot((i,i),(mZ[i]-sZ[i],mZ[i]+sZ[i]),'m',linewidth=lw)
    ax0[0].plot((i-delta,i+delta),(mZ[i]+sZ[i],mZ[i]+sZ[i]),'m',linewidth=lw)
    ax0[0].plot((i-delta,i+delta),(mZ[i]-sZ[i],mZ[i]-sZ[i]),'m',linewidth=lw)
ax0[0].set_ylabel('$Z_i$')
ax0[0].set_title('remax activation function')

for i in range(nb_Z):
    ax0[1].plot((i,i),(mM_mcs[i]-sM_mcs[i],mM_mcs[i]+sM_mcs[i]),'g',linewidth=lw)
    ax0[1].plot((i-1.5*delta,i+1.5*delta),(mM_mcs[i]+sM_mcs[i],mM_mcs[i]+sM_mcs[i]),'g',linewidth=lw)
    ax0[1].plot((i-1.5*delta,i+1.5*delta),(mM_mcs[i]-sM_mcs[i],mM_mcs[i]-sM_mcs[i]),'g',linewidth=lw)

    ax0[1].plot((i,i),(mM[i]-sM[i],mM[i]+sM[i]),'b',linewidth=lw)
    ax0[1].plot((i-delta,i+delta),(mM[i]+sM[i],mM[i]+sM[i]),'b',linewidth=lw)
    ax0[1].plot((i-delta,i+delta),(mM[i]-sM[i],mM[i]-sM[i]),'b',linewidth=lw)
ax0[1].set_ylabel('$M_i$')

for i in range(nb_Z):
    ax0[2].plot((i,i),(mA[i]-sA[i],mA[i]+sA[i]),'b',linewidth=lw,label='remax(${\mathbf{Z}}$): $\mu_A\pm \sigma_A$')
    ax0[2].plot((i-delta,i+delta),(mA[i]+sA[i],mA[i]+sA[i]),'b',linewidth=lw)
    ax0[2].plot((i-delta,i+delta),(mA[i]-sA[i],mA[i]-sA[i]),'b',linewidth=lw)

    ax0[2].plot((i,i),(mA_s[i]-sA_s[i],mA_s[i]+sA_s[i]),'g',linewidth=lw,label='remax(${\mathbf{Z}}$): MCS $\mu_A\pm \sigma_A$',alpha=0.75)
    ax0[2].plot((i-delta*1.5,i+delta*1.5),(mA_s[i]+sA_s[i],mA_s[i]+sA_s[i]),'g',linewidth=lw,alpha=0.75)
    ax0[2].plot((i-delta*1.5,i+delta*1.5),(mA_s[i]-sA_s[i],mA_s[i]-sA_s[i]),'g',linewidth=lw,alpha=0.75)

    ax0[2].scatter([i],np.maximum(0,mZ[i])/sum(np.maximum(0,mZ)),5,'r',marker='p',linewidth=lw,label='remax(${\mathbf{\mu_Z}}$): $a$')
    if i==0:
        ax0[2].legend()
ax0[2].set_ylabel('$A_i=\Pr(Class)$')
ax0[2].set_ylim(-0.1,1.1)
ax0[2].set_xlabel('Class #')

for i in range(nb_Z):
    ax0[3].plot((i,i),(cov_M_M_sum_mcs[i,i],cov_M_M_sum[i]),'--k',linewidth=lw/2)
    ax0[3].scatter(i,cov_M_M_sum_mcs[i,i],20,'g')
    ax0[3].scatter(i,cov_M_M_sum[i],5,'b')
ax0[3].set_ylabel('$cov(M,Msum)$')

# for i in range(nb_Z):
#     ax0[4].plot((i,i),(mM_sum_mcs[0],mM_sum),'--k',linewidth=lw/2)
#     ax0[4].scatter(i,mM_sum_mcs[0],10,'g')
#     ax0[4].scatter(i,mM_sum,5,'b')
# ax0[4].set_ylabel('$mM_sum$')
# for i in range(nb_Z):
#     ax0[4].plot((i,i),(mM_mcs[i],mM[i]),'--k',linewidth=lw/2)
#     ax0[4].scatter(i,mM_mcs[i],10,'g')
#     ax0[4].scatter(i,mM[i],5,'b')
# ax0[4].set_ylabel('$mM$')

for i in range(nb_Z):
    ax0[4].plot((i,i),(cov_lnM_lnM_sum_mcs[i],cov_lnM_lnM_sum[i]),'--k',linewidth=lw/2)
    ax0[4].scatter(i,cov_lnM_lnM_sum_mcs[i],20,'g')
    ax0[4].scatter(i,cov_lnM_lnM_sum[i],5,'b')
ax0[4].set_ylabel('$cov(lnM,lnMsum)$')

# for i in range(nb_Z):
#     ax0[4].plot((i,i),(cov_cAlnM_mcs[i,i],cov_cAlnM[i]),'--k',linewidth=lw/2)
#     ax0[4].scatter(i,cov_cAlnM_mcs[i,i],10,'g')
#     ax0[4].scatter(i,cov_cAlnM[i],5,'b')
# ax0[4].set_ylabel('$cov(lnA,lnM)$')

for i in range(nb_Z):
    ax0[5].plot((i,i),(cov_ZA_mcs[i,i],cov_ZA[i]),'--k',linewidth=lw/2)
    ax0[5].scatter(i,cov_ZA_mcs[i,i],20,'g')
    ax0[5].scatter(i,cov_ZA[i],5,'b')
ax0[5].set_ylabel('$cov(A,Z)$')
ax0[5].set_ylim(0,np.max(cov_ZA_mcs*1.3))


z=np.linspace(mZ[0]-4*sZ[0],mZ[0]+4*sZ[0],100)
ax1[0].plot(z,norm.pdf(z,loc=mZ[0],scale=sZ[0]),'m',label='$\mathcal{N}()$')
ax1[0].set_xlabel('$z_0$')
ax1[0].set_ylabel('$f(z_0)$')
ax1[0].legend(loc=1, prop={'size': 6})

m=np.linspace(mM[0]-4*sM[0],mM[0]+4*sM[0],100)
ax1[1].plot(m,norm.pdf(m,loc=mM[0],scale=sM[0]),'k',label='$\mathcal{N}()$')
ax1[1].plot(m,lognorm.pdf(m,slnM[0],scale=np.exp(mlnM[0])),'--',label='$\ln\mathcal{N}()$')
ax1[1].set_xlabel('$m_0$')
ax1[1].set_ylabel('$f(m_0)$')
ax1[1].legend(loc=1, prop={'size': 6})

m_sum=np.linspace(mM_sum-4*sM_sum,mM_sum+4*sM_sum,100)
ax2.plot(m_sum,norm.pdf(m_sum,loc=mM_sum,scale=sM_sum),'r',label='$\mathcal{N}()$')
ax2.plot(m_sum,lognorm.pdf(m_sum,slnM_sum,scale=np.exp(mlnM_sum)),'--',label='$\ln\mathcal{N}()$')
ax2.set_xlabel('$\sum m$')
ax2.set_ylabel('$f(\overline{m})$')
ax2.legend(loc=1, prop={'size': 6})

# a=np.linspace(min(-0.1,mA[0]-4*sA[0]),max(1.1,mA[0]+4*sA[0]),100)
a=np.linspace(-0.1,1.1,100)

ax1[2].plot(a,norm.pdf(a,loc=mA[0],scale=sA[0]),'b',label='$\mathcal{N}()$')
ax1[2].plot(a,lognorm.pdf(a,slnA[0],scale=np.exp(mlnA[0])),'--',label='$\ln\mathcal{N}()$')
ax1[2].set_xlabel('$a_0$')
ax1[2].set_ylabel('$f(a_0)$')
ax1[2].legend(loc=1, prop={'size': 6})

plt.draw()
plt.show()