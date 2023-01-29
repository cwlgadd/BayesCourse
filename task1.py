#     Modified from https://github.com/is0383kk/GMM-MetropolisHastings/blob/main/gmm_mh.ipynb

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from scipy.stats import  wishart, dirichlet 
from sklearn.metrics.cluster import adjusted_rand_score as ari
from tqdm import tqdm

def make_data(K=3, N=250, mixture_weights=None):
    """
    K:                    Number of clusters
    N:                    Number of samples
    mixture_weights:      Mixing ratio each mixture
    """

    if mixture_weights is None:
        mixture_weights = np.array([0.3333, 0.3333, 0.3333]) 

    ##################### Define data generating mixture model #####################
    # Specify mean vector parameters
    mixture_mus = np.array([[0, 5.0], 
                            [-10.0, -10.0],
                            [5.0, -20.0]]
                          )

    # Specify covariance matrix parameters
    mixture_sigmas = np.array([[[12.0, 0],
                                [0, 12.0]],
                               [[12.0, 0.0], 
                                [0.0, 12.0]],
                               [[12.0, 0.0], 
                                [0.0, 12.0]]]
                             )

    ############################## Make synthetic data ##############################
    # Allocate samples to a mixture component
    allocation_matrix = np.random.multinomial(n=1, pvals=mixture_weights, size=N)
    _, allocation = np.where(allocation_matrix == 1)
    # Sample from data generating distribution
    x = np.array([
        np.random.multivariate_normal(
            mean=mixture_mus[k], cov=mixture_sigmas[k], size=1).flatten() for k in allocation
    ])

    ############################### Plot ###############################
    x_1_line_1 = np.linspace(
        np.min(mixture_mus[:, 0] - 3 * np.sqrt(mixture_sigmas[:, 0, 0])), 
        np.max(mixture_mus[:, 0] + 3 * np.sqrt(mixture_sigmas[:, 0, 0])), 
        num=300
    )
    x_2_line_1 = np.linspace(
        np.min(mixture_mus[:, 1] - 3 * np.sqrt(mixture_sigmas[:, 1, 1])), 
        np.max(mixture_mus[:, 1] + 3 * np.sqrt(mixture_sigmas[:, 1, 1])), 
        num=300
    )
    x_1_grid_1, x_2_grid_1 = np.meshgrid(x_1_line_1, x_2_line_1)
    x_point_1 = np.stack([x_1_grid_1.flatten(), x_2_grid_1.flatten()], axis=1)
    x_dim_1 = x_1_grid_1.shape

    # Ovservation model
    true_model_1 = 0
    for k in range(K):
        tmp_density_1 = multivariate_normal.pdf(x=x_point_1, mean=mixture_mus[k], cov=mixture_sigmas[k])    
        true_model_1 += mixture_weights[k] * tmp_density_1


    # plot graph
    plt.figure(figsize=(12, 9))
    for k in range(K):
        k_idx, = np.where(allocation == k)
        plt.scatter(x=x[k_idx, 0], y=x[k_idx, 1], label='cluster:' + str(k + 1)) 
    plt.contour(x_1_grid_1, x_2_grid_1, true_model_1.reshape(x_dim_1), linestyles='--')
    plt.suptitle('Synthetic data for Multimodal-GMM:Observation1', fontsize=20)
    plt.title('Number of data:' + str(N) + ', Number of clusters:' + str(K), loc='left')
    plt.xlabel('')
    plt.ylabel('')
    plt.colorbar()
    plt.show()
    plt.close()
    
    return x, allocation

def GaussianMixtureModel(x, allocation, iterations=50, allocation_truth=None):
    """
    x: data
    allocation: mixture allocation
    """
    K = 3           # Number of clusters
    D = len(x)      # Number of data
    dim = len(x[0]) # Number of dimensions
    init_mus = np.array([[0, 5.0], 
                         [-10.0, -10.0],
                         [5.0, -20.0]]
                       )
    
    print(f"Number of clusters: {K}"); print(f"Number of data: {D} of dimension {dim}"); 
    
    
    ############################## Initializing parameters ##############################
    # Set hyperparameters
    alpha_k = np.repeat(2.0, K)             # Hyperparameters for \pi
    beta = 1.0; 
    m_d_1 = np.repeat(0.0, dim);            # Hyperparameters for \mu^A, \mu^B
    w_dd_1 = np.identity(dim) * 0.05;       # Hyperparameters for \Lambda^A, \Lambda^B
    nu = dim                                # Hyperparameters for \Lambda^A, \Lambda^B (nu > Number of dimension - 1)

    # Initializing \pi
    pi_k = dirichlet.rvs(alpha=alpha_k, size=1).flatten()
    alpha_hat_k = np.zeros(K)

    # Initializing z
    z_nk_new = np.random.multinomial(n=1, pvals=pi_k, size=D)   # Current iteration z
    z_nk_old = np.random.multinomial(n=1, pvals=pi_k, size=D)   # z before 1 iteration
    _, z_n = np.where(z_nk_new == 1)
    _, z_n = np.where(z_nk_old == 1)

    # Initializing unsampled \mu, \Lambda
#     mixture_means = np.zeros((K, dim))
#     mixture_lambdas = np.zeros((K, dim, dim))
    mixture_means = [np.random.multivariate_normal(mean=init_mus[k], cov= np.array([[4.0, 0],[0, 4.0]]), size=1) for k in range(K)]
    mixture_means = np.stack(mixture_means).squeeze(1)
    mixture_lambdas = np.stack([10 * np.eye(dim) for k in range(K)])

    # Initializing learning parameters
    eta_nk = np.zeros((D, K))
    tmp_eta_n = np.zeros((K, D))
    beta_hat_k_1 = np.zeros(K);
    m_hat_kd_1 = np.zeros((K, dim)); 
    w_hat_kdd_1 = np.zeros((K, dim, dim)); 
    nu_hat_k_1 = np.zeros(K); 
    cat_liks_prop = np.zeros(D)
    cat_liks_old = np.zeros(D)
    ARI = np.zeros((iterations)) 
    count_accept = np.zeros((iterations)) # number of accepted moves

    print("Metropolis-Hastings algorithm")
    for i in tqdm(range(iterations)):

        z_pred_n = [] # Labels estimated by the model
        count = 0

        z_nk = np.zeros((D, K));
        # Sampling the current iteration z 
        for k in range(K): 
            # Calculate the pdf for each sample
            tmp_eta_n[k] = np.diag(-0.5 * (x - mixture_means[k]).dot(mixture_lambdas[k]).dot((x - mixture_means[k]).T)).copy() 
            tmp_eta_n[k] += 0.5 * np.log(np.linalg.det(mixture_lambdas[k]) + 1e-7) 
            tmp_eta_n += np.log(pi_k[k] + 1e-7) 
            eta_nk[:, k] = np.exp(tmp_eta_n[k])     
#         print(eta_nk)
        eta_nk /= np.sum(eta_nk, axis=1, keepdims=True) + 1e-12


        for d in range(D):
            z_nk_new[d] = np.random.multinomial(n=1, pvals=eta_nk[d], size=1).flatten() # sampling z_nk_new
            z_prop = np.argmax(z_nk_new[d])
            z_old = np.argmax(z_nk_old[d])
            
            # The task is to replicate this Metropolis update in a function
            cat_liks_prop[d] = multivariate_normal.pdf(
                               x[d], 
                               mean=mixture_means[z_prop], 
                               cov=np.linalg.inv(mixture_lambdas[z_prop]),
                               )
            cat_liks_old[d] = multivariate_normal.pdf(
                              x[d], 
                              mean=mixture_means[z_old], 
                              cov=np.linalg.inv(mixture_lambdas[z_old]),
                              )
            judge_r = (cat_liks_prop[d] + 1e-4) / (cat_liks_old[d] + 1e-4)
            judge_r = min(1, judge_r) # acceptance rate
            if judge_r >= np.random.rand(): 
                # accept
                z_nk[d] = z_nk_new[d]
                count = count + 1
            else: 
                # reject
                z_nk[d] = z_nk_old[d]
            z_pred_n.append(z_prop)

        # Process on sampling \mu, \lambda using the updated z
        for k in range(K):
            # Calculate the parameters of the posterior distribution of \mu
            beta_hat_k_1[k] = np.sum(z_nk[:, k]) + beta; 
            m_hat_kd_1[k] = np.sum(z_nk[:, k] * x.T, axis=1); 
            m_hat_kd_1[k] += beta * m_d_1; 
            m_hat_kd_1[k] /= beta_hat_k_1[k]; 


            # Calculate the parameters of the posterior distribution of \Lambda
            tmp_w_dd_1 = np.dot((z_nk[:, k] * x.T), x); 
            tmp_w_dd_1 += beta * np.dot(m_d_1.reshape(dim, 1), m_d_1.reshape(1, dim)); 
            tmp_w_dd_1 -= beta_hat_k_1[k] * np.dot(m_hat_kd_1[k].reshape(dim, 1), m_hat_kd_1[k].reshape(1, dim))
            tmp_w_dd_1 += np.linalg.inv(w_dd_1); 
            w_hat_kdd_1[k] = np.linalg.inv(tmp_w_dd_1); 
            nu_hat_k_1[k] = np.sum(z_nk[:, k]) + nu

            # Sampling new \Lambda
            mixture_lambdas[k] = wishart.rvs(size=1, df=nu_hat_k_1[k], scale=w_hat_kdd_1[k])

            # Sampling new \mu
            mixture_means[k] = np.random.multivariate_normal(
                mean=m_hat_kd_1[k], 
                cov=np.linalg.inv(beta_hat_k_1[k] * mixture_lambdas[k]), size=1).flatten()
#             print(mixture_means)


        # Process on sampling \pi using the updated z
        # Calculate the parameters of the posterior distribution of \pi
        alpha_hat_k = np.sum(z_nk, axis=0) + alpha_k

        # Sampling \pi
        pi_k = dirichlet.rvs(size=1, alpha=alpha_hat_k).flatten()

        ARI[i] = np.round(ari(allocation, z_pred_n), 3)           # Calculate ARI
        count_accept[i] = count                                  # Number of times accepted during current iteration
#         print(f"ARI:{ARI[i]}, Accept_num:{count_accept[i]}")

        z_nk_old = z_nk_new
    
    
    # ARI
    plt.plot(range(0,iterations), ARI, marker="None")
    plt.xlabel('iteration')
    plt.ylabel('ARI')
    plt.ylim(0,1)
    #plt.savefig("./image/ari.png")
    plt.show()
    plt.close()

    # number of acceptation 
    plt.figure()
    plt.ylim(0,D)
    plt.plot(range(0,iterations), count_accept, marker="None", label="Accept_num")
    plt.xlabel('iteration')
    plt.ylabel('Number of acceptation')
    plt.legend()
    #plt.savefig('./image/accept.png')
    plt.show()
    plt.close()
