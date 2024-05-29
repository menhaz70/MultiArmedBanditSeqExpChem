
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
import timeit
#%% Data synthesis
def data_synthesis():
    n_samples = 300
    n_features = 500
    n_valid_features = 30
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

#%% OFUL bandit dynamic feature  (Syn2 Data)
# path = "synthetic data1"

def oful_bandit(in_variable, target, path, delta=0.05):

    if not os.path.exists(path):
        os.mkdir(path) 
        
    K = len(target)    
    best_target = np.argmax(target)
    simple_structure_molecule = 10
    T = K - simple_structure_molecule + 1
    training_indices = []
                                       
    train_linear_correlation = []
    
    oful_reward_armwise = [[] for i in range(K)]
    oful_reward_store = []      
    test_linear_correlation =[]
    all_train_rmse = []  
    all_test_rmse = [] 

    lambda_ = []    
    theta_hat_store = []
    aa = []
    bb = []      
    S_store = []
    C_t_store = []

    pulled_arm_store = []    # len(pulled_arm_store)
    time_to_pick_best = []   
    all_gamma_ = []
    all_threshold = []
    elapsed_time1_ = []
    
    for t in range(T):   
        if t == 0:
            # t=0
            molecules_except_best_target = [m for m in range(K) if m != best_target]
            pulled_arm = np.random.choice(molecules_except_best_target,\
                                            simple_structure_molecule, replace=False).tolist() 
            reward = target[pulled_arm].tolist()             
            for i, j in enumerate(pulled_arm):
                training_indices.append(j)
                oful_reward_armwise[j].append(reward[i])          
                oful_reward_store.append(reward[i]) 
                pulled_arm_store.append(j)                                       
           
        else:
            # traning test spliting on raw data
            train_y = target[training_indices]
            train_data_raw = in_variable.values[training_indices,:]           
            test_y = target.drop(training_indices)             

            # Standardization
            trans  = StandardScaler()
            trans.fit(train_data_raw)           
            standardized_in_variable = trans.transform(in_variable.values)
            
            # Time counting
            start_time = timeit.default_timer()
            
            # traning test spliting after standardization on full feature space                              
            train_data = np.insert(standardized_in_variable[training_indices,:], 0,\
                                   np.ones(len(train_y)), axis=1)                
            test_data  =  np.insert(np.delete(standardized_in_variable, training_indices, 0),\
                                    0, np.ones(len(test_y)), axis=1)                                                       
            pipeline_ = Pipeline([("scaler", StandardScaler()), ("model", Ridge())])
            search_ = GridSearchCV(pipeline_, {"model__alpha": np.logspace(-3, 3, num=500)},\
                                    cv=5, scoring="neg_mean_squared_error", verbose=0)            
            search_.fit(train_data, train_y)
            lamda = (search_.best_params_)["model__alpha"]
            
            # parameter estimation for OFUL            
            d = len(train_data[1,:])    # x.shape
            I = np.identity(d)
            
            V = train_data.T @ train_data + lamda * I
            det_ = np.linalg.det(V/lamda) 
            
            if det_ == np.inf:                
                while det_ == np.inf:
                    lamda = 2*lamda
                    V = train_data.T @ train_data + lamda * I
                    det_ = np.linalg.det(V/lamda)
            elif det_ == 0: 
                while det_ == 0:
                    lamda = lamda/2
                    V = train_data.T @ train_data + lamda * I
                    det_ = np.linalg.det(V/lamda)                
            else:
                pass
            
            lambda_.append(lamda)                   ## this lamda is not from tuning step, but from while loop
            V_inverse = np.linalg.inv(V)            ## V = x'x + lamda*I
            theta_hat = np.array((V_inverse @ (train_data.T @ oful_reward_store)).T)
            shape_ = V.shape[0]

            # Estimation of prediction using ridge regression 
            train_pred_target = train_data @ theta_hat
            test_pred_target = test_data @ theta_hat        ## test_data.shape              

            end_time = timeit.default_timer()
            elapsed_time1 = end_time - start_time
            elapsed_time1_.append(elapsed_time1) 
            # Time count end 
            
            train_correlation = pearsonr(train_y, train_pred_target)[0]
            train_linear_correlation.append(train_correlation)
            
            train_rmse = np.sqrt(mean_squared_error(train_y, train_pred_target))
            right_critical_val = stats.chi2.ppf(delta/2, df=len(train_y) - 1)
            R = np.sqrt(len(train_y) / right_critical_val) * train_rmse
            all_train_rmse.append(R) 
               
            r, c = test_data.shape                                   
            if r >= 2:
                test_correlation = pearsonr(test_y, test_pred_target)[0]
                test_linear_correlation.append(test_correlation)  
                test_rmse = np.sqrt(mean_squared_error(test_y, test_pred_target))
                all_test_rmse.append(test_rmse)                   
            else:
                pass
            
            # Calculation of S from SE of the theta  
            critical_value = stats.norm.ppf(1-(delta/2), loc=0, scale=1)              
            SE_theta_hat = np.sqrt(np.diag(V_inverse @ (train_data.T @ train_data) @ V_inverse)) * R
            upper_bound_of_theta = []
            for i in arange(len(theta_hat)):
                if theta_hat[0] < 0:
                    upper_bound_of_theta.append(theta_hat[i] - critical_value * SE_theta_hat[i])
                else:
                    upper_bound_of_theta.append(theta_hat[i] + critical_value * SE_theta_hat[i])

            S = np.linalg.norm(upper_bound_of_theta)                    

            a = R * np.sqrt(np.log(det_) - 2 * np.log(delta))      
            b = (lamda ** 0.5) * S               
            C_t = a + b

            aa.append(a)
            bb.append(b)
            theta_hat_store.append(theta_hat)
            S_store.append(S)
            C_t_store.append(C_t)
            
            x = np.insert(standardized_in_variable, 0, np.ones(K), axis=1) # x.shape             
            theta_set = np.empty(shape=(0, d))  
            for k in range(K):  
                lagrange = (x[k,:] @ V_inverse @ x[k,:].T) ** 0.5 / C_t
                theta1 = (((x[k,:] @ V_inverse) / lagrange) + theta_hat).reshape(1, d)     # theta.shape 
                theta2 = (-((x[k,:] @ V_inverse) / lagrange) + theta_hat).reshape(1, d)     # theta.shape 
                theta_set = np.append(theta_set, theta1, axis=0)
                theta_set = np.append(theta_set, theta2, axis=0)
                    
            train_bound = np.empty(shape=(2, 0))        
            test_bound = np.empty(shape=(2, 0))  

            for i in range(len(train_data)):
                scaler_ = training_indices[i] * 2
                ubound = train_data[i,:] @ theta_set[scaler_].flatten() - train_data[i,:] @ theta_hat   # theta_set.shape
                lbound =  train_data[i,:] @ theta_hat - train_data[i,:] @ theta_set[scaler_ + 1]
                interval = np.array([lbound, ubound]).reshape(2, 1)                   
                train_bound = np.append(train_bound, interval, axis=1)
                
            whole_indices = set(np.arange(K)) 
            test_indices = [x for x in whole_indices if x not in training_indices]   # len(test_indices)
            
            for i in range(len(test_data)):
                scaler_ = test_indices[i] * 2    
                ubound = test_data[i,:] @ theta_set[scaler_].flatten() - test_data[i,:] @ theta_hat 
                lbound =  test_data[i,:] @ theta_hat - test_data[i,:] @ theta_set[scaler_ + 1]
                interval = np.array([lbound, ubound]).reshape(2, 1)                   
                test_bound = np.append(test_bound, interval, axis=1)
            
            # Stopping condition       
            t_est_bound2 = np.empty(shape=(2, 0))               
            for j in range(len(test_pred_target)):
                l_bound = test_pred_target.tolist()[j] - test_bound[1, j]
                u_bound = test_pred_target.tolist()[j] + test_bound[1, j]
                i_nterval = np.array([l_bound, u_bound]).reshape(2, 1) 
                t_est_bound2 = np.append(t_est_bound2, i_nterval, axis=1)

            max_predicted_upper_bound = np.max(t_est_bound2)
            best_in_training = np.max(train_y)
            gamma_ = R * np.sqrt(2) * special.erfinv(2*(1-delta) - 1) 
            threshold_ = best_in_training #- gamma_
            all_gamma_.append(gamma_)
            all_threshold.append(threshold_)

            # Plot of the acual and predicted rewards   
            plt.figure()
            plt.errorbar(train_y, train_pred_target, \
                fmt=".", color="g", capsize=3, label="Training/ Corr = %s, & RMSE_ubound = %s"\
                    %(round(train_correlation, 2), round(R, 2)))  # yerr = train_bound
            plt.errorbar(test_y, test_pred_target, yerr=test_bound, color="r", \
                fmt=".", capsize=3, label="Test set/ Corr = %s, Test_RMSE =%s"\
                    %(round(test_correlation, 2), round(test_rmse, 2)))
            plt.errorbar(train_y.values[-1], train_pred_target[-1],  yerr=train_bound[:,-1].reshape(2, 1), \
                fmt=".", color="b", capsize=3, label=" %s_th arm chosen and reawrd %s"\
                    % (pulled_arm, target[pulled_arm])) 
            plt.title("OFUL plots for iteration_%02d" % t)                  
            plt.plot([min(target), max(target)], [min(target), max(target)],\
                     color="0.70", linewidth=1, ls="dashed")
            plt.legend(loc="upper left")
            plt.axhline(y=threshold_, color="b", linestyle="-")
            plt.grid()
            plt.xlabel("Actual target") #    
            plt.ylabel("Predicted target")
            plt.savefig("%s/regplot%02d.png" % (path, t), dpi=300)
            plt.close()
            
            stopping_ = t
            if max_predicted_upper_bound < threshold_:
                break
            
            # choosing next player and updates
            max_ = None               
            for i in range(K):         
                if i in pulled_arm_store:
                    pass
                else:                    
                    for j in range(len(theta_set)):   ## for each theta
                        v = x[i,:] @ theta_set[j,:]
                        if max_ is None or max_ < v:
                            max_ = v
                            pulled_arm = i
            
            reward = target[pulled_arm]
            training_indices.append(pulled_arm)
            oful_reward_armwise[pulled_arm].append(reward)
            oful_reward_store.append(reward) 
            pulled_arm_store.append(pulled_arm)

    best_candidate = np.max(target)
    if best_candidate in oful_reward_store:
        time_ = oful_reward_store.index(best_candidate) - simple_structure_molecule + 1 
        time_to_pick_best.append(time_)  # time_to_pick_best  time_to_pick_best
    else:
        time_to_pick_best.append(-2)

    pd.DataFrame({"lambda": lambda_, "R": all_train_rmse, "S": S_store, "1st part": aa, "2nd part": bb}).to_csv("%s/parts of ellipse.csv" % path)                
   
    return oful_reward_store, path, stopping_, lambda_, time_to_pick_best, theta_hat_store,\
        S_store, C_t_store, pulled_arm_store, all_test_rmse, all_train_rmse,\
            test_linear_correlation, all_gamma_, all_threshold, elapsed_time1_ 

#%% OFUL Calculaing many episodes

def episode_oful():
    folder_path = "Corrected Manuscript Syn2 OFUL-AF 50 run"
    number_episode = 50
    
    reward_store_ = []     # reward_store.shape
    stopping_time = []
    lambda_store = []
    time_to_get_best = []
    theta_hat_store_ = []
    S_store_ = []
    C_t_store_ = []
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
        oful_reward_store, path, stopping_, lambda_, time_to_pick_best, theta_hat_store, S_store, C_t_store, pulled_arm_store, all_test_rmse, all_train_rmse, test_linear_correlation, all_gamma_, all_threshold, elapsed_time1_ =\
            oful_bandit(in_variable, target, path, delta=0.05)

        pd.DataFrame(pd.concat([target,in_variable], axis=1)).to_csv("%s/data.csv" % path)
        pd.DataFrame(lambda_).to_csv("%s/hypara_oful.csv" % path)
        pd.DataFrame({"gamma": all_gamma_, "threshold": all_threshold}).to_csv("%s/gamma and threshold.csv" % path) 
        pd.DataFrame(elapsed_time1_).to_csv("%s/elapsed_time1.csv" % path) 

        pd.DataFrame(all_test_rmse).to_csv("%s/episode_test_rmse.csv" % path)        
        pd.DataFrame(all_train_rmse).to_csv("%s/episode_train_rmse.csv" % path)        
        pd.DataFrame(oful_reward_store).to_csv("%s/episode_oful_reward_store.csv" % path) 
        pd.DataFrame(oful_reward_store).to_csv("%s/episode_test_linear_correlation.csv" % path) 
        
        reward_store_.append(oful_reward_store)          
        stopping_time.append(stopping_)
        lambda_store.append(lambda_)      
        time_to_get_best.append(time_to_pick_best[0])
        pulled_arm_.append(pulled_arm_store)

        theta_hat_store_.append(theta_hat_store)
        S_store_.append(S_store)
        C_t_store_.append(C_t_store) 
        train_rmse_.append(all_train_rmse)
        test_rmse_.append(all_test_rmse)
        correlation_.append(test_linear_correlation)   
        elapsed_time_all.append(elapsed_time1_)
        
    pd.DataFrame(lambda_store).to_csv("%s/oful_hypara_all_exp.csv" % path) 
    pd.DataFrame(time_to_get_best).to_csv("%s/time_to_get_best.csv" % path)
    pd.DataFrame(theta_hat_store_).to_csv("%s/theta_hat_store_.csv" % path)
    pd.DataFrame(S_store_).to_csv("%s/S_store_.csv" % path)
    pd.DataFrame(C_t_store_).to_csv("%s/C_t_store_.csv" % path)
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
    
    return stopping_time, lambda_store, path, reward_store_, time_to_get_best, theta_hat_store_, S_store_, C_t_store_, pulled_arm_, folder_path, test_rmse_, correlation_

stopping_time, lambda_store, path, reward_store_, time_to_get_best, theta_hat_store_,\
    S_store_, C_t_store_, pulled_arm_, folder_path, test_rmse_,\
        correlation_ = episode_oful()

def plots__(path):
    alpha___ = pd.read_csv(r"%s/oful_hypara_all_exp.csv" % path)   
    ffff = pd.read_csv(r"%s/time_to_get_best.csv" % path)["0"]       
    alpha___.drop(alpha___.columns[alpha___.columns.str.contains("unnamed", case=False)], axis=1, inplace=True)
    path2 = "%s/lambda_oful" % path
    if not os.path.exists(path2):
        os.mkdir(path2) 
    for i in range(len(alpha___)):
        oth = alpha___.iloc[i].dropna()
        plt.figure(figsize=(10, 15))
        plt.figure()
        plt.ylim([-3, 20])
        plt.plot(oth, label="Time to get best reward = %s" % ffff.iloc[i])
        plt.title("Plot of Lambdas used in each episode")
        plt.xlabel("Iteration/experiment in a episode")
        plt.ylabel("Magnitute of Lambda for OFUL")
        plt.legend(loc="upper left")
        # plt.xticks(oth, minor=False)
        plt.axvline(x=ffff.iloc[i], color="r", linestyle="-")
        plt.savefig("%s/alphas%02d.png" % (path2, i), dpi=300)
        plt.close("all")
plots__(path)

def S_and_C_t_plots__(path):
    ffff = pd.read_csv(r"%s/time_to_get_best.csv" % path)["0"] 
    sss= pd.read_csv(r"%s/S_store_.csv" % path)  
    ct = pd.read_csv(r"%s/C_t_store_.csv" % path)
    sss.drop(sss.columns[sss.columns.str.contains("unnamed", case=False)], axis=1, inplace=True)
    ct.drop(ct.columns[ct.columns.str.contains("unnamed", case=False)], axis=1, inplace=True)
    path2 = "%s/SandQ" % path
    if not os.path.exists(path2):
        os.mkdir(path2) 
        
    for i in range(len(sss)):
        sss_ = sss.iloc[i].dropna()
        ct_ = ct.iloc[i].dropna()
        plt.figure(figsize=(10, 15))
        plt.figure()
        plt.ylim([-3, 50])
        plt.plot(sss_, label="Value of S")
        plt.plot(ct_, label="Value of Q")
        plt.title("Plot of S and Q")
        plt.xlabel("Iteration/experiment in a episode")
        plt.ylabel("Magnitute")
        plt.axvline(x=ffff.iloc[i], color="r", linestyle="-", label="Time to get best reward = %s" % ffff.iloc[i])
        plt.legend(loc="upper left")
        plt.savefig("%s/QandS_%02d.png" % (path2, i), dpi=300)    
        plt.close("all")
S_and_C_t_plots__(path)

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
