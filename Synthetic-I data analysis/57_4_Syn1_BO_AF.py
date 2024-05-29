
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt     
import os
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error 
from numpy import arange
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge, Lasso
from scipy import special
from sklearn.pipeline import Pipeline
from scipy import stats
from scipy.stats import gaussian_kde
from sklearn.model_selection import train_test_split

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import\
    DotProduct, RBF, ExpSineSquared, Matern, WhiteKernel,ConstantKernel
# from bayes_opt import BayesianOptimization, UtilityFunction
import time
import timeit
from scipy.spatial import distance
#%% Data synthesis
def data_synthesis():
    # np.random.seed(100) 
    n_samples = 300
    n_features = 30
    n_valid_features = 10
    percen_ = 0.20
    
    X = np.empty(shape=(n_samples, 0))   ### X1.shape
    for i in range(n_features):
        hyperparameter_= np.random.normal(loc = 0, scale = 1, size = 2)
        random_number = np.random.normal(loc = hyperparameter_[0], scale=abs(hyperparameter_[1]), size=n_samples).reshape(n_samples, 1)
        X = np.append(X, random_number, axis = 1)

    coef = np.zeros(n_features)
    coef[:n_valid_features] = np.random.normal(loc=0, scale=1, size= n_valid_features)
    y_wo_noise = X@coef
    SD_ = np.std(y_wo_noise)
    y = y_wo_noise + np.random.normal(0, percen_*SD_, n_samples)
    y_ = pd.Series(y)
    X_ = pd.DataFrame(X)  
    return y_, X_, coef, SD_, percen_
# target, in_variable, coef, SD_, percen_ = data_synthesis()

#%% bayesian optimization  (synthetic Data)
# 0.05/300 = 0.0001
def bayesian_opt(in_variable, target, path, SD_, delta=0.05):

    if not os.path.exists(path):
        os.mkdir(path) 
        
    K = len(target)    
    best_target = np.argmax(target)    
    simple_structure_molecule = 10
    T = K - simple_structure_molecule + 1
    training_indices = []
                                       
    train_linear_correlation = []
    test_linear_correlation = []
    all_train_rmse = []  
    all_test_rmse = [] 
    
    bo_reward_armwise = [[] for i in range(K)]
    bo_reward_store = []      
    pulled_arm_store = []    # len(pulled_arm_store)
    time_to_pick_best = []   
    all_gamma_ = []
    all_threshold = []
    bo_test_std = []
    beta_12 = []
    kernel_values = []
    elapsed_time1_ = []
    lenght_scale_bound = []
    
    _, numerofvariable = in_variable.shape   
    length_scale_ = [1.0]*(numerofvariable)
    length_scale_bounds_ = [(4, 12)]*(numerofvariable) 
    noise_bounds=(0.01, 2)
    # noise_bounds=(0.0001, 2)
    sigma_bound = (4, 12)
    
    for t in range(T):   
        if t == 0:
            molecules_except_best_target = [m for m in range(K) if m != best_target]
            pulled_arm = np.random.choice(molecules_except_best_target,\
                                            simple_structure_molecule, replace=False).tolist() 
            reward = target[pulled_arm].tolist()             
            for i, j in enumerate(pulled_arm):
                training_indices.append(j)
                bo_reward_armwise[j].append(reward[i])          
                bo_reward_store.append(reward[i]) 
                pulled_arm_store.append(j)                                                   
        else:
            # traning test spliting on raw data
            train_y = target[training_indices]
            test_y = target.drop(training_indices)              
            train_data_raw = in_variable.loc[training_indices]           
            test_data_raw = in_variable.drop(training_indices)

            # Standardization
            trans = StandardScaler()
            trans.fit(train_data_raw)           
            standardized_in_variable = trans.transform(in_variable.values)                               
            train_data = standardized_in_variable[training_indices,:]
            test_data  = np.delete(standardized_in_variable, training_indices, 0) 

            # train_data = train_data_raw
            # test_data  = test_data_raw

            # execution time count
            start_time = timeit.default_timer()
            
            train_train = []
            r1, c1 =train_data.shape
            for i in range(r1):
                for j in range(r1):
                    if i!=j:
                        dst = distance.euclidean(train_data[i,:], train_data[j,:]) 
                        train_train.append(dst)            
            
            L = max(min(train_train), 0.01)
            U = max(train_train)
            # L = max(np.percentile(train_train,5), 0.01)
            # U = np.percentile(train_train,95)           
            # lenght_scale_bound.append([L,U])
            
            # isotropic kernel
            kernel = ConstantKernel(constant_value=1.0, constant_value_bounds=(1e-5, 1e5))\
                *RBF(length_scale=1.0, length_scale_bounds=(L, U))\
                + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-2, 5))
                
            #Aniosotropic kernel
            # kernel = 1.0* RBF(length_scale=length_scale_, length_scale_bounds=length_scale_bounds_)\
            #     + WhiteKernel(noise_level=1.0, noise_level_bounds=noise_bounds)                                                  

            gpr = GaussianProcessRegressor(kernel=kernel, alpha=1e-5, optimizer='fmin_l_bfgs_b', n_restarts_optimizer=5, random_state=None) 
            gpr.fit(train_data, train_y)
            tuned_kernel=gpr.kernel_.get_params(deep=False)           # gpr.score(test_data, test_y) False
            kernel_values.append(tuned_kernel)
            # print(tuned_kernel)
            # print(gpr.score(train_data, train_y))
            # print(gpr.score(test_data, test_y)) 

            # Estimation of prediction using GP regression 
            train_pred_target, train_std = gpr.predict(train_data, return_std = True)
            test_pred_target, test_std = gpr.predict(test_data, return_std = True) 
            
            end_time = timeit.default_timer()
            elapsed_time1 = end_time - start_time
            elapsed_time1_.append(elapsed_time1) 
            #Execution time count end
            
            train_correlation = pearsonr(train_y, train_pred_target)[0]
            train_linear_correlation.append(train_correlation)

            train_rmse = np.sqrt(mean_squared_error(train_y, train_pred_target))
            right_critical_val = stats.chi2.ppf(delta/2, df=len(train_y) - 1)
            R = np.sqrt(len(train_y) / right_critical_val) * train_rmse
            all_train_rmse.append(R) 
                        
            bo_test_std.append(test_std.tolist())
            
            r, c = test_data.shape                                   
            if r >= 2:
                test_correlation = pearsonr(test_y, test_pred_target)[0]
                test_linear_correlation.append(test_correlation)  
                test_rmse = np.sqrt(mean_squared_error(test_y, test_pred_target))
                all_test_rmse.append(test_rmse)                   
            else:
                pass
            
            ## calculating confidence interval for test data
         
            beta_phalf = np.sqrt(2*np.log((len(test_pred_target)*np.pi**2*(t+1)**2)/(6*delta)))   # beta_phalf = Beta^1/2
            upperbound = test_pred_target + beta_phalf * test_std
            max_predicted_upper_bound = np.max(upperbound)
            beta_12.append(beta_phalf)
            
            best_in_training = np.max(train_y)
            gamma_ = R * np.sqrt(2) * special.erfinv(2*(1-delta) - 1) 
            threshold_ = best_in_training #- gamma_
            all_gamma_.append(gamma_)
            all_threshold.append(threshold_)


            # Plot of the acual and predicted rewards   
            yerr_train = (np.sqrt(2*np.log((len(train_pred_target)*np.pi**2*(t+1)**2)/(6*delta)))* train_std).tolist()[-1]
            yerr_test = (beta_phalf * test_std).tolist()
            
            plt.figure()
            plt.errorbar(train_y, train_pred_target, fmt=".", color="g", capsize=3, label="Training/ Corr = %s, & RMSE_ubound = %s"\
                    %(round(train_correlation, 2), round(R, 2)))
                
            plt.errorbar(test_y, test_pred_target, yerr=yerr_test, color="r", fmt=".", capsize=3, label="Test set/ Corr = %s, Test_RMSE =%s"\
                    %(round(test_correlation, 2), round(test_rmse, 2)))  
                
            plt.errorbar(train_y.values[-1], train_pred_target[-1],  yerr=yerr_train, fmt=".", color="b", capsize=3,\
                         label=" %s_th arm chosen and reawrd %s" % (pulled_arm, target[pulled_arm])) 
                
            plt.title("OFUL plots for iteration_%02d" % t)                  
            plt.plot([min(target), max(target)], [min(target), max(target)],\
                     color="0.70", linewidth=1, ls="dashed")
            plt.legend(loc="upper left")
            plt.axhline(y=threshold_, color="b", linestyle="-")
            plt.grid()
            plt.xlabel("Actual target") #    
            plt.ylabel("Predicted target")
            plt.savefig("%s/regplot%02d.png" % (path, t), dpi=300)
            # plt.show()
            plt.close()

            
            stopping_ = t
            if max_predicted_upper_bound <= threshold_:
                break
              

            # choosing next player and updates
            
            accusition_function = test_pred_target\
                +np.sqrt(2*np.log((len(test_pred_target)*np.pi**2*(t+1)**2)/(6*delta))) * test_std
            all_indices =np.arange(K).tolist()
            test_indices = [x for x in all_indices if (x not in training_indices)]
            pulled_arm = test_indices[np.argmax(accusition_function)]
            
            reward = target[pulled_arm]
            training_indices.append(pulled_arm)
            bo_reward_armwise[pulled_arm].append(reward)
            bo_reward_store.append(reward) 
            pulled_arm_store.append(pulled_arm)
            
    pd.DataFrame(training_indices).to_csv("%s/training_indices.csv" % path)
    best_candidate = np.max(target)
    if best_candidate in bo_reward_store:
        time_ = bo_reward_store.index(best_candidate) - simple_structure_molecule + 1 
        time_to_pick_best.append(time_)  # time_to_pick_best  time_to_pick_best
    else:
        time_to_pick_best.append(-2)
   
    return bo_reward_store, stopping_, time_to_pick_best, pulled_arm_store, all_test_rmse, all_train_rmse,\
        test_linear_correlation, all_gamma_, all_threshold, bo_test_std, beta_12, kernel_values,\
            elapsed_time1_, lenght_scale_bound

# bo_reward_store, stopping_, time_to_pick_best, pulled_arm_store, all_test_rmse, all_train_rmse, test_linear_correlation, all_gamma_, all_threshold = bayesian_opt(in_variable, target, path, SD_, delta=0.05)

#%% OFUL Calculaing many episodes

def episode_bo():   
    folder_path = "Manuscript Syn1 BO-AF 50 run"
    number_episode = 50
    
    reward_store_ = []     # reward_store.shape
    stopping_time = []

    time_to_get_best = []
    pulled_arm_ = []
    test_rmse_ = []
    train_rmse_ = []
    correlation_ = []
    elapsed_time_all = []
    

    for i in range(number_episode):  
        path = "%s/Episode_%s" % (folder_path, i)
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)  

        target, in_variable, coef, SD_, percen_ = data_synthesis()              
        bo_reward_store, stopping_, time_to_pick_best, pulled_arm_store, all_test_rmse,\
            all_train_rmse, test_linear_correlation, all_gamma_, all_threshold,\
                bo_test_std, beta_12, kernel_values, elapsed_time1_, lenght_scale_bound\
                    = bayesian_opt(in_variable, target, path, SD_, delta=0.05)

        pd.DataFrame(pd.concat([target,in_variable], axis=1)).to_csv("%s/data.csv" % path) 
        pd.DataFrame({"threshold": all_threshold}).to_csv("%s/threshold.csv" % path) 
        pd.DataFrame(pd.concat([pd.Series(beta_12),pd.DataFrame(bo_test_std)], axis=1)).to_csv("%s/beta and bo_test_std.csv" % path) 
        pd.DataFrame(kernel_values).to_csv("%s/kernel_values.csv" % path)  
        pd.DataFrame(elapsed_time1_).to_csv("%s/elapsed_time1.csv" % path) 
        pd.DataFrame(lenght_scale_bound).to_csv("%s/lenght_scale_bound.csv" % path)

        pd.DataFrame(all_test_rmse).to_csv("%s/episode_test_rmse.csv" % path)        
        pd.DataFrame(all_train_rmse).to_csv("%s/episode_train_rmse.csv" % path)        
      
        reward_store_.append(bo_reward_store)          
        stopping_time.append(stopping_)     
        time_to_get_best.append(time_to_pick_best[0])
        pulled_arm_.append(pulled_arm_store)

        train_rmse_.append(all_train_rmse)
        test_rmse_.append(all_test_rmse)
        correlation_.append(test_linear_correlation)    
        elapsed_time_all.append(elapsed_time1_)

    pd.DataFrame(time_to_get_best).to_csv("%s/time_to_get_best.csv" % path)
    pd.DataFrame(reward_store_).to_csv("%s/reward_store_.csv" % path)
    pd.DataFrame(stopping_time).to_csv("%s/stopping_time.csv" % path)   
    pd.DataFrame(pulled_arm_).to_csv("%s/pulled_arm.csv" % path)
    pd.DataFrame(train_rmse_).to_csv("%s/rmse_tarining.csv" % path)
    pd.DataFrame(test_rmse_).to_csv("%s/rmse_test.csv" % path)
    pd.DataFrame(correlation_).to_csv("%s/correlation.csv" % path)    
    pd.DataFrame(elapsed_time_all).to_csv("%s/elapsed_time_all.csv" % path) 
    
    plt.figure()
    plt.bar(np.arange(number_episode).tolist(), time_to_get_best)
    plt.title("Plot of Time at best reward arrived")
    plt.xlabel("Episodes")
    plt.ylabel("Time at best reward arrived")
    plt.savefig("%s/Time at best reward achieved.png" % path, dpi=300) 
    plt.close()
    
    return stopping_time, reward_store_, time_to_get_best, pulled_arm_, folder_path,\
        test_rmse_, correlation_, path

stopping_time, reward_store_, time_to_get_best,  pulled_arm_, folder_path,\
    test_rmse_, correlation_, path = episode_bo()


def mse_plot(path):
    alpha___ = pd.read_csv(r"%s/rmse_test.csv" % path)   
    ffff = pd.read_csv(r"%s/time_to_get_best.csv" % path)["0"]       
    alpha___.drop(alpha___.columns[alpha___.columns.str.contains("unnamed", case=False)], axis=1, inplace=True)
    path2 = "%s/mse_plot" % path
    if not os.path.exists(path2):
        os.mkdir(path2) 
    for i in range(len(alpha___)):
        oth = alpha___.iloc[i].dropna()
        plt.figure(figsize=(10, 15))
        plt.figure()
        # plt.ylim([-3, 20])
        plt.plot(oth, label = "Time to get best reward = %s" % ffff.iloc[i])
        plt.title("Plot of MSE calculated each episode")
        plt.xlabel("Iteration/experiment in a episode")
        plt.ylabel("Magnitute of mse")
        plt.legend(loc="upper left")
        # plt.xticks(oth, minor=False)
        plt.axvline(x=ffff.iloc[i], color="r", linestyle="-")
        plt.savefig("%s/mse_episode%02d.png" % (path2, i), dpi=300)
        plt.close("all")        
mse_plot(path)

def correlation_plot(path):
    alpha___ = pd.read_csv(r"%s/correlation.csv" % path)   
    ffff = pd.read_csv(r"%s/time_to_get_best.csv" % path)["0"]       
    alpha___.drop(alpha___.columns[alpha___.columns.str.contains("unnamed", case=False)], axis=1, inplace=True)
    path2 = "%s/correlation_plot" % path
    if not os.path.exists(path2):
        os.mkdir(path2) 
    for i in range(len(alpha___)):
        oth = alpha___.iloc[i].dropna()
        plt.figure(figsize=(10, 15))
        plt.figure()
        plt.plot(oth, label = "Time to get best reward = %s" % ffff.iloc[i])
        plt.title("Plot of correlation")
        plt.xlabel("Iteration/experiment in a episode")
        plt.ylabel("Magnitute of correlation")
        plt.legend(loc="upper left")
        plt.axvline(x=ffff.iloc[i], color="r", linestyle="-")
        plt.savefig("%s/corr_episode%02d.png" % (path2, i), dpi=300)
        plt.close("all")
correlation_plot(path)


