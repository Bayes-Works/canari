import numpy as np
from scipy.stats import lognorm, norm

def log_remax(mZ, sZ):
    """
    """
    s2Z = sZ**2

    cdfn = np.maximum(1E-20,norm.cdf(mZ/sZ))
    pdfn = np.maximum(1E-20,norm.pdf(mZ/sZ))
    mM = np.maximum(1E-20,sZ*pdfn + mZ * cdfn)
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
    cov_M_M_sum = s2M

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

    return mA, sA

def hierachical_softmax(mZ, sZ):
    """
    """
    # First gate
    pr_gate1 = norm.cdf(mZ[0]/np.sqrt(1+sZ[0]**2))

    # Probabilities for three classes
    pr_all_classes = np.array([pr_gate1, 1 - pr_gate1])

    return pr_all_classes