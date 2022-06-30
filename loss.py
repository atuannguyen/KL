import torch
import torch.nn.functional as F
import numpy as np



def kl_upper(mu1, sigma1, coeff1, mu2, sigma2, coeff2):
    
    coeff1 = coeff1 / coeff1.sum()
    coeff2 = coeff2 / coeff2.sum()
    # KL[f_a||g_b] >>>> kl_matrix: [B,B]
    mu_p = mu1.unsqueeze(1)
    sigma_p = sigma1.unsqueeze(1)
    mu_q = mu2.unsqueeze(0)
    sigma_q = sigma2.unsqueeze(0)

    kl_matrix = torch.log(sigma_q) - torch.log(sigma_p) + 0.5*(sigma_p**2+(mu_p-mu_q)**2)/sigma_q**2 - 0.5
    kl_matrix = kl_matrix.sum(2)


    # H[f_a] >>>> H: [B]
    H = torch.log(sigma1*np.sqrt(2*np.pi*np.e)).sum(1)

    # Z_{a,alpha} >>>> Z: [B,B]
    mu_p = mu1.unsqueeze(1)
    sigma_p = sigma1.unsqueeze(1)
    mu_q = mu1.unsqueeze(0)
    sigma_q = sigma1.unsqueeze(0)

    log_Z = -1/2 * (np.log(2*np.pi) + torch.log(sigma_p**2 + sigma_q**2) + (mu_p-mu_q)**2/(sigma_p**2+sigma_q**2))
    log_Z = log_Z.sum(2)
    

    # Calculate upper bound
    coeff1_us0 = coeff1.unsqueeze(0)
    coeff2_us0 = coeff2.unsqueeze(0)

    log_Z_log_w = log_Z + torch.log(coeff1_us0)
    log_r_num = torch.logsumexp(log_Z_log_w,1)

    neq_kl = -kl_matrix
    neq_kl_log_w = neq_kl + torch.log(coeff2_us0)
    log_r_de = torch.logsumexp(neq_kl_log_w,1)

    r = log_r_num - log_r_de
    r = r + H
    r = (r*coeff1).sum(0) - mu1.shape[1]*(1-np.log(2))/2
    return r


def kl_lower(mu1, sigma1, coeff1, mu2, sigma2, coeff2):
    
    coeff1 = coeff1 / coeff1.sum()
    coeff2 = coeff2 / coeff2.sum()
    # KL[f_a||f_a] >>>> kl_matrix: [B,B]
    mu_p = mu1.unsqueeze(1)
    sigma_p = sigma1.unsqueeze(1)
    mu_q = mu1.unsqueeze(0)
    sigma_q = sigma1.unsqueeze(0)

    kl_matrix = torch.log(sigma_q) - torch.log(sigma_p) + 0.5*(sigma_p**2+(mu_p-mu_q)**2)/sigma_q**2 - 0.5
    kl_matrix = kl_matrix.sum(2)


    # H[f_a] >>>> H: [B]
    H = torch.log(sigma1*np.sqrt(2*np.pi*np.e)).sum(1)

    # T_{a,b} >>>> Z: [B,B]
    mu_p = mu1.unsqueeze(1)
    sigma_p = sigma1.unsqueeze(1)
    mu_q = mu2.unsqueeze(0)
    sigma_q = sigma2.unsqueeze(0)

    log_T = -1/2 * (np.log(2*np.pi) + torch.log(sigma_p**2 + sigma_q**2) + (mu_p-mu_q)**2/(sigma_p**2+sigma_q**2))
    log_T = log_T.sum(2)
    

    # Calculate upper bound
    coeff1_us0 = coeff1.unsqueeze(0)
    coeff2_us0 = coeff2.unsqueeze(0)

    log_T_log_w = log_T + torch.log(coeff2_us0)
    log_r_de = torch.logsumexp(log_T_log_w,1)

    neq_kl = -kl_matrix
    neq_kl_log_w = neq_kl + torch.log(coeff1_us0)
    log_r_num = torch.logsumexp(neq_kl_log_w,1)

    r = log_r_num - log_r_de
    r = r - H
    r = (r*coeff1).sum(0)
    return r

def kl_approx(mu1, sigma1, coeff1, mu2, sigma2, coeff2):
    return (kl_upper(mu1, sigma1, coeff1, mu2, sigma2, coeff2)+kl_lower(mu1, sigma1, coeff1, mu2, sigma2, coeff2))/2

if __name__=='__main__':
    B = 220
    K = 512
    mu1 = torch.zeros([B,K])
    sigma1 = torch.ones([B,K])
    coeff1 = torch.ones([B])

    mu2 = torch.zeros([B,K]) - 10
    sigma2 = torch.ones([B,K])
    coeff2 = torch.ones([B])
    k = kl_upper(mu1, sigma1, coeff1, mu2, sigma2,coeff2)
    import pdb; pdb.set_trace()
