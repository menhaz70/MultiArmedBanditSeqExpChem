
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

def data_preparation():
    sheet = pd.read_csv(r"FreeSolv.csv")
    target = sheet["expt"]
    feature = pd.read_excel(r"freesolved_feature.xlsx")       
    names = feature.columns
    num_feature = arange(len(names)).tolist()
    delete_box = []    # len(delete_box)

    for i in num_feature:
        
        if i in delete_box:
            pass
        else:          
            for j in num_feature:
                if i < j:
                    bool_ = feature[names[i]].equals(feature[names[j]])
                    if bool_ == True:
                        delete_box.append(j)
                    else:
                        pass

    for element in delete_box:
        if element in num_feature:
            num_feature.remove(element)

    unique_features = names[num_feature]    # len(unique_features)
    in_variable1 = feature[unique_features]
    
    return delete_box, num_feature, in_variable1, target   

delete_box, num_feature, in_variable, target = data_preparation()

s = np.sum(in_variable1) > 5
in_variable = in_variable1[s[s].index]

#%% OFUL bandit dynamic feature  (Freesolve Data)

def rdm_bandit(in_variable, target, path, delta=0.05):

    if not os.path.exists(path):
        os.mkdir(path) 
     
    K = len(target) 
    best_target = np.argmin(target)
    simple_structure_molecule = 10 
    T = K - simple_structure_molecule + 1
    training_indices = [] 
    
    train_linear_correlation = []
    test_linear_correlation =[]
    all_train_rmse = []  
    all_test_rmse = []  

    lambda_ = []   
    theta_hat_store = []
    aa = []
    bb = []  
    S_store = []
    C_t_store = []

    oful_reward_armwise = [[] for i in range(K)]    ### len(oful_reward_armwise)
    oful_reward_store = []    
    pulled_arm_store = []    ### len(selected_arm_store)
    time_to_pick_best = []
    all_gamma_ = []
    all_threshold = []
    elapsed_time1_ = []   
    
    for t in range(T):   
        if t == 0:
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

            # traning test spliting after standardization on full feature space                              
            train_data = np.insert(standardized_in_variable[training_indices,:], 0, np.ones(len(train_y)), axis=1)                
            test_data  =  np.insert(np.delete(standardized_in_variable, training_indices, 0), 0, np.ones(len(test_y)), axis=1)                                                       

            # Time counting
            start_time = timeit.default_timer()
            
            
            pipeline_ = Pipeline([("scaler", StandardScaler()), ("model", Ridge())])
            search_ = GridSearchCV(pipeline_, {"model__alpha": np.logspace(-3, 3, num=200)},\
                                    cv=5, scoring="neg_mean_squared_error", verbose=3)            
            search_.fit(train_data, train_y)
            lamda = max((search_.best_params_)["model__alpha"], 0.05)

            #parameter estimation for OFUL                
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
            # S = np.linalg.norm(theta_hat) + 0.50  
            # S = 1.10 * np.linalg.norm(theta_hat)            

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

            min_predicted_lower_bound = np.min(t_est_bound2)
            best_in_training = np.min(train_y)
            gamma_ = R*np.sqrt(2) * special.erfinv(2 * (1-delta) - 1) 
            threshold_ = best_in_training #+ gamma_
            all_gamma_.append(gamma_)
            all_threshold.append(threshold_)

            # Plot of the acual and predicted rewards   
            plt.figure()
            plt.errorbar(train_y, train_pred_target, \
                fmt=".", color="g", capsize=3, label="Training/ Corr = %s, & RMSE_ubound = %s"\
                    %(round(train_correlation, 2), round(R, 2)))  # yerr = train_bound,
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
            if min_predicted_lower_bound >= threshold_:
                break
            
            # choosing next player and updates
            all_indices = list(np.arange(K))
            test_indices = list((set(all_indices) - set(training_indices)))
            pulled_arm = np.random.choice(test_indices, 1, replace=False)[0]
          
            reward = target[pulled_arm]
            training_indices.append(pulled_arm)
            oful_reward_armwise[pulled_arm].append(reward)
            oful_reward_store.append(reward) 
            pulled_arm_store.append(pulled_arm)
                
    best_candidate = np.min(target)
    if best_candidate in oful_reward_store:
        time_ = oful_reward_store.index(best_candidate) - simple_structure_molecule + 1 
        time_to_pick_best.append(time_)
    else:
        time_to_pick_best.append(-2) # since "time_to_pick_best" will be within (1-642)
    
    pd.DataFrame({"lambda": lambda_, "R": all_train_rmse, "S": S_store, "1st part": aa, "2nd part": bb}).to_csv("%s/parts of ellipse.csv" % path) 
                            
    return oful_reward_store, path, stopping_, lambda_,\
        time_to_pick_best, theta_hat_store, S_store, C_t_store,\
            pulled_arm_store, all_test_rmse, all_train_rmse, test_linear_correlation,\
                all_gamma_, all_threshold, elapsed_time1_ 

#%% OFUL Calculaing many episodes

def episode_rdm_bandit():
    folder_path = "Corrected Manuscript freesolv RS 50 run"
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
        
        oful_reward_store, path, stopping_, lambda_,\
            time_to_pick_best, theta_hat_store, S_store, C_t_store,\
                pulled_arm_store, all_test_rmse, all_train_rmse, test_linear_correlation,\
                    all_gamma_, all_threshold, elapsed_time1_\
                        = rdm_bandit(in_variable, target, path, delta=0.05)
                        
        pd.DataFrame(lambda_).to_csv("%s/hypara_oful.csv" % path) 
        pd.DataFrame({"gamma": all_gamma_, "threshold": all_threshold}).to_csv("%s/gamma and threshold.csv" % path) 
        pd.DataFrame(elapsed_time1_).to_csv("%s/elapsed_time1.csv" % path)

        pd.DataFrame(all_test_rmse).to_csv("%s/episode_test_rmse.csv" % path)        
        pd.DataFrame(all_train_rmse).to_csv("%s/episode_train_rmse.csv" % path)            
     
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
    pd.DataFrame(pulled_arm_).to_csv("%s/selected_arm.csv" % path) 
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
    # plt.show() 
    plt.close("all")
    
    return stopping_time, lambda_store, path, reward_store_, time_to_get_best,\
        theta_hat_store_, S_store_, C_t_store_, pulled_arm_,\
            folder_path, test_rmse_, correlation_

stopping_time, lambda_store, path, reward_store_, time_to_get_best,\
    theta_hat_store_, S_store_, C_t_store_, pulled_arm_,\
        folder_path, test_rmse_, correlation_  = episode_rdm_bandit()                

def best_reward(reward_store_):
    best_reward = []
    for i in range(len(reward_store_)):
        r = np.nanmin(reward_store_[i])
        best_reward.append(r)
    return best_reward
best_reward = best_reward(reward_store_)

def kde_plot(target, stopping_time, best_reward, path):
    path2 = "%s/density" % path
    if not os.path.exists(path2):
        os.mkdir(path2)     
    data = pd.DataFrame.from_dict({"stopping_time": stopping_time,"reward": best_reward})
    min_ = np.min(target)
    d1 = data.loc[data["reward"] <= min_]   #success
    d2 = data.loc[data["reward"] > min_]    #Unsuccess
    k =len(data)
    success_percentage = (len(d1) / k) * 100
    unsuccess_percentage = (len(d2) / k) * 100  
    
    kde1 = gaussian_kde(d1["stopping_time"])
    plt.scatter(d1["stopping_time"].values, np.zeros(d1["reward"].values.size), marker="x", c="blue", zorder=5, label="Success cases (%s percent)" % round(success_percentage, 2))
    xss1 = np.linspace(0, max(stopping_time), 100)
    plt.fill_between(xss1, kde1(xss1) * (len(d1)/k), np.zeros(xss1.size), facecolor="blue", alpha=0.3)
    if len(d2) <=1:
        pass
    else:
        kde2 = gaussian_kde(d2["stopping_time"].values)
        plt.scatter(d2["stopping_time"].values, np.zeros(d2["reward"].values.size), marker="x", c="red", zorder=5, label="Failure cases (%s percent)" % round(unsuccess_percentage, 2))
        xss2 = np.linspace(0, max(stopping_time), 100)
        plt.fill_between(xss2, kde2(xss2) * (len(d2)/k), np.zeros(xss2.size), facecolor="red", alpha=0.3)
        
    plt.title("Plot of stopping time Vs Reward achived")
    plt.xlabel("Stopping time")
    plt.ylabel("Desnsity of the reward achived")  
    plt.text(70, 0.007, "Maximum stopping time success case = %s" % np.max(d1["stopping_time"]))
    plt.text(70, 0.006, "Max stopping time failure case = %s" % np.max(d2["stopping_time"]))
    plt.grid()
    plt.legend()
    plt.savefig("%s/max_rewards_achived.png" % path2, dpi=300)
    plt.close("all")    
kde_plot(target, stopping_time, best_reward, path)


def rdm_hypara_plot(path):
    oful_hypara_csv = pd.read_csv(r"%s/oful_hypara_all_exp.csv" % path)   
    best_time = pd.read_csv(r"%s/time_to_get_best.csv" % path)["0"]       
    oful_hypara_csv.drop(oful_hypara_csv.columns[oful_hypara_csv.columns.str.contains("unnamed", case=False)], axis=1, inplace=True)
    path2 = "%s/lambda_oful" % path
    if not os.path.exists(path2):
        os.mkdir(path2) 
    for i in range(len(oful_hypara_csv)):
        each_epi_oful_hypara = oful_hypara_csv.iloc[i].dropna()
        plt.figure(figsize=(10, 15))
        plt.figure()
        plt.ylim([-3, 20])
        plt.plot(each_epi_oful_hypara, label="Time to get best reward = %s" % best_time.iloc[i])
        plt.title("Plot of Lambdas used in each episode")
        plt.xlabel("Iteration/experiment in a episode")
        plt.ylabel("Magnitute of Lambda for OFUL")
        plt.legend(loc="upper left")
        plt.axvline(x=best_time.iloc[i], color="r", linestyle="-")
        plt.savefig("%s/alphas%02d.png" % (path2, i), dpi=300)
        plt.close("all")
rdm_hypara_plot(path)

def S_and_Ct_plot(path):
    best_time = pd.read_csv(r"%s/time_to_get_best.csv" % path)["0"] 
    S_csv= pd.read_csv(r"%s/S_store_.csv" % path)  
    ct_csv = pd.read_csv(r"%s/C_t_store_.csv" % path) # ct_csv
    S_csv.drop(S_csv.columns[S_csv.columns.str.contains("unnamed",case=False)], axis=1, inplace=True)
    ct_csv.drop(ct_csv.columns[ct_csv.columns.str.contains("unnamed", case=False)], axis=1, inplace=True)
    path2 = "%s/SandQ" % path
    if not os.path.exists(path2):
        os.mkdir(path2) 
        
    for i in range(len(S_csv)):
        each_epi_S = S_csv.iloc[i].dropna() #each_epi_S
        each_epi_ct = ct_csv.iloc[i].dropna() #each_epi_ct
        plt.figure(figsize=(10, 15))
        plt.figure()
        plt.ylim([-3, 50])
        plt.plot(each_epi_S, label="Value of S")
        plt.plot(each_epi_ct, label="Value of Q")
        plt.title("Plot of S and Q")
        plt.xlabel("Iteration/experiment in a episode")
        plt.ylabel("Magnitute")
        plt.axvline(x=best_time.iloc[i], color="r", linestyle="-", label="Time to get best reward = %s" % best_time.iloc[i])
        plt.legend(loc="upper left")
        plt.savefig("%s/QandS_%02d.png" % (path2, i), dpi=300)    
        plt.close("all")
S_and_Ct_plot(path)

def rmse_test_plot(path):
    rmse_csv = pd.read_csv(r"%s/rmse_test.csv" % path)   
    best_time = pd.read_csv(r"%s/time_to_get_best.csv" % path)['0']       
    rmse_csv.drop(rmse_csv.columns[rmse_csv.columns.str.contains('unnamed', case=False)], axis=1, inplace=True)
    path2 = "%s/rmse_test_plot" % path
    if not os.path.exists(path2):
        os.mkdir(path2) 
    for i in range(len(rmse_csv)):
        each_epi_rmse = rmse_csv.iloc[1].dropna()
        plt.figure(figsize=(10, 15))
        plt.figure()
        # plt.ylim([-3, 20])
        plt.plot(each_epi_rmse, label="Time to get best reward = %s" % best_time.iloc[i])
        plt.title("Plot of RMSE calculated each episode")
        plt.xlabel("Iteration/experiment in a episode")
        plt.ylabel("Magnitute of RMSE")
        plt.legend(loc="upper left")
        # plt.xticks(oth, minor=False)
        plt.axvline(x=best_time.iloc[i], color="r", linestyle="-")
        plt.savefig("%s/rmse_epi_%02d.png" % (path2, i), dpi=300)
        # x = np.arange(1, len(each_episode_rmse)+1)
        # plt.xticks(x, rotation=90)
        plt.close("all")        
rmse_test_plot(path)


def correlation_plot(path):
    correlation_csv = pd.read_csv(r"%s/correlation.csv" % path)   
    best_time = pd.read_csv(r"%s/time_to_get_best.csv" % path)["0"]       
    correlation_csv.drop(correlation_csv.columns[correlation_csv.columns.str.contains("unnamed", case=False)], axis=1, inplace=True)
    path2 = "%s/correlation_plot" % path
    if not os.path.exists(path2):
        os.mkdir(path2) 
    for i in range(len(correlation_csv)):
        each_epi_corr = correlation_csv.iloc[i].dropna()
        plt.figure(figsize=(10, 15))
        plt.figure()
        plt.plot(each_epi_corr, label = "Time to get best reward = %s" % best_time.iloc[i])
        plt.title("Plot of correlation")
        plt.xlabel("Iteration/experiment in a episode")
        plt.ylabel("Magnitute of correlation")
        plt.legend(loc="upper left")
        plt.axvline(x=best_time.iloc[i], color="r", linestyle="-")
        plt.savefig("%s/corr_episode%02d.png" % (path2, i), dpi=300)
        plt.close("all")
correlation_plot(path)

#%%
