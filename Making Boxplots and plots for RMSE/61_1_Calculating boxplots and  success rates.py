
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import matplotlib.ticker as ticker
from scipy import stats 
# Manuscript freesolv BO-AF min-p95 noise-1-1
# Manuscript freesolv BO-SF min-p95 noise-1-1
# Manuscript freesolv BO-AF min max compare
# Manuscript freesolv BO-SF min max compare
# Manuscript Syn1 BO-AF min-p95 noise-1-1
# Manuscript Syn1 BO-SF min-p95 noise-1-1
# Manuscript Syn1 BO-AF min max compare
# Manuscript Syn1 BO-SF min max compare

# all_paths = ["C:/Users/menha/Desktop/corrected_ex/Manuscript Syn1 RS 50 run/Episode_49",
#               "C:/Users/menha/Desktop/corrected_ex/Manuscript Syn1 BO-AF 50 run/Episode_49",
#               "C:/Users/menha/Desktop/corrected_ex/Manuscript Syn1 BO-SF 50 run/Episode_49",
#               "C:/Users/menha/Desktop/corrected_ex/Corrected Manuscript Syn1 OFUL-AF 50 run/Episode_49",
#               "C:/Users/menha/Desktop/corrected_ex/Corrected Manuscript Syn1 OFUL-SF 50 run/Episode_49"]  
# figure_path = "C:/Users/menha/OneDrive - 国立大学法人 北海道大学/PhD work/Manuscript working/figure_raw_python/corrected/alljpg/box_syn1.jpg"


all_paths = ["C:/Users/menha/Desktop/corrected_ex/Manuscript Syn2 RS 50 run/Episode_49",
              "C:/Users/menha/Desktop/corrected_ex/Manuscript Syn2 BO-AF 50 run/Episode_49",
              "C:/Users/menha/Desktop/corrected_ex/Manuscript Syn2 BO-SF 50 run/Episode_49",
              "C:/Users/menha/Desktop/corrected_ex/Corrected Manuscript Syn2 OFUL-AF 50 run/Episode_49",
              "C:/Users/menha/Desktop/corrected_ex/Corrected Manuscript Syn2 OFUL-SF 50 run/Episode_49"]  
figure_path = "C:/Users/menha/OneDrive - 国立大学法人 北海道大学/PhD work/Manuscript working/figure_raw_python/corrected/alljpg/box_syn2.jpg"

# all_paths = ["C:/Users/menha/Desktop/corrected_ex/Manuscript freesolv RS 50 run/Episode_49",
#               "C:/Users/menha/Desktop/corrected_ex/Manuscript freesolv BO-AF 50 run/Episode_49",
#               "C:/Users/menha/Desktop/corrected_ex/Manuscript freesolv BO-SF 50 run/Episode_49",
#               "C:/Users/menha/Desktop/corrected_ex/Corrected Manuscript Freesolv OFUL-AF 50 run/Episode_49",
#               "C:/Users/menha/Desktop/corrected_ex/Corrected Manuscript freesolv OFUL-SF 50 run 1/Episode_49"]   
# figure_path = "C:/Users/menha/OneDrive - 国立大学法人 北海道大学/PhD work/Manuscript working/figure_raw_python/corrected/alljpg/box_freesolv.jpg"



# all_paths = ["C:/Users/menha/Desktop/corrected_ex/Manuscript enan RS 50 run/Episode_49",
#               "C:/Users/menha/Desktop/corrected_ex/Manuscript enan BO-AF 50 run/Episode_49",
#               "C:/Users/menha/Desktop/corrected_ex/Manuscript enan BO-SF 50 run/Episode_49",
#               "C:/Users/menha/Desktop/corrected_ex/Corrected Manuscript enan OFUL-AF 50 run/Episode_49",
#               "C:/Users/menha/Desktop/corrected_ex/Corrected Manuscript enan OFUL-SF 50 run/Episode_49"]
# figure_path = "C:/Users/menha/OneDrive - 国立大学法人 北海道大学/PhD work/Manuscript working/figure_raw_python/corrected/alljpg/box_freesolv.jpg"


# all_paths = ["C:/Users/menha/Desktop/Review/Review photoswitch RS 50 run/Episode_49",
#               "C:/Users/menha/Desktop/Review/Review photoswitch BO-AF 50 run/Episode_49",
#               "C:/Users/menha/Desktop/Review/Review photoswitch BO-SF 50 run/Episode_49",
#               "C:/Users/menha/Desktop/Review/Review photoswitch OFUL-AF 50 run/Episode_49",
#               "C:/Users/menha/Desktop/Review/Review photoswitch OFUL-SF 50 run/Episode_49"]
# figure_path = "C:/Users/menha/OneDrive - 国立大学法人 北海道大学/PhD work/Manuscript working/figure_raw_python/corrected/alljpg/box_photoswitch.jpg"




arg_max_percentage = []    # len(arg_percentage_std_down)   hoefding
# arg_hoefding_up = []
# arg_hoefding_down = []
for p in range(0,len(all_paths)):
    path = all_paths[p]
    best_time_ = pd.read_csv(r"%s/time_to_get_best.csv" %path, index_col=0)["0"].tolist()

    percentage = []    
    hoefding_up = []
    hoefding_down = []
    
    for j in range(np.max(best_time_)+1):
        
        p = 100 * (([i <= j and i > 0 for i in best_time_].count(True))/len(best_time_))
        percentage.append(p) 
        
        # epsilon_ = np.sqrt((1/(2*len(best_time_))) * np.log(2/0.05))
        # hoefding_up.append(p + 100 * epsilon_ if p + 100 * epsilon_ <= 100 else 100)
        # hoefding_down.append(p - 100 * epsilon_ if p - 100 * epsilon_ > 0 else 0)
    arg_max_percentage.append(percentage)   
    # arg_hoefding_up.append(hoefding_up)
    # arg_hoefding_down.append(hoefding_down)




linewidth = 2
labels = ["RS",  "BO-AF", "BO-SF", "OFUL-AF", "OFUL-SF"]

# labels = ["OFUL-AF", "OFUL-SF" ]
best_time = []
never_best = []
stop = []
intial_sample = 10



for p in range(5):
    path = all_paths[p]
    
    time_to_best_without_initial_sample = pd.read_csv(r"%s/time_to_get_best.csv" %path, index_col=0) 
    
    count_never_best = time_to_best_without_initial_sample.squeeze().tolist().count(-2)
    never_best.append(count_never_best)
    index_never_best = time_to_best_without_initial_sample[time_to_best_without_initial_sample["0"] ==-2].index
    time_to_best = time_to_best_without_initial_sample.drop(index_never_best) #+ intial_sample

    time_to_best.drop(time_to_best.columns[time_to_best.columns.str.contains("unnamed", case=False)], axis=1, inplace=True)
    best_time.append(time_to_best.squeeze().tolist())
    
    time_to_stopped = pd.read_csv(r"%s/stopping_time.csv" %path, index_col=0).drop(index_never_best)  #+ intial_sample
    time_to_stopped.drop(time_to_stopped.columns[time_to_stopped.columns.str.contains("unnamed", case=False)], axis=1, inplace=True)
    stop.append(time_to_stopped.squeeze().tolist())


fig, (ax1,ax2) = plt.subplots(2, 1, figsize=(8,10))    
ax1.boxplot(best_time, vert=True, showmeans=True)
# ax1.set_xticklabels(labels=labels, rotation = 0, fontsize=25)
ax1.xaxis.set_major_locator(ticker.NullLocator())
ax1.set_ylabel("Best finding time", rotation = 90, fontsize=26)
ax1.tick_params(axis='y', labelsize=17)
ax1.grid(axis='y')

ax2.boxplot(stop, vert=True, showmeans=True)
ax2.set_xticklabels(labels=labels, fontsize=20)
ax2.set_ylabel("Stopping time", rotation = 90, fontsize=26)
ax2.tick_params(axis='y', labelsize=17)
ax2.grid(axis='y')
fig.tight_layout()       # without this functiontitle and xticklabel will be overlap
plt.savefig(figure_path) 
plt.show()


num_of_episode = len(pd.read_csv(r"%s/time_to_get_best.csv" %all_paths[0], index_col=0))


print("Percentage of success of RS = %s" %((1-(never_best[0]/num_of_episode))*100))
print("Percentage of success of BO-AF = %s" %((1-(never_best[1]/num_of_episode))*100))
print("Percentage of success of BO-SF = %s" %((1-(never_best[2]/num_of_episode))*100))
print("Percentage of success of OFUL-AF = %s" %((1-(never_best[3]/num_of_episode))*100))
print("Percentage of success of OFUL-SF = %s" %((1-(never_best[4]/num_of_episode))*100))

print("Median of best finding time of RS = %s" %np.median(best_time[0]))
print("Median of best finding time of BO-AF = %s" %np.median(best_time[1]))
print("Median of best finding time of BO-SF = %s" %np.median(best_time[2]))
print("Median of best finding time of OFUL-AF = %s" %np.median(best_time[3]))
print("Median of best finding time of OFUL-SF = %s" %np.median(best_time[4]))

print("Median of stopping time of RS = %s" %np.median(stop[0]))
print("Median of stopping time of BO-AF = %s" %np.median(stop[1]))
print("Median of stopping time of BO-SF = %s" %np.median(stop[2]))
print("Median of stopping time of OFUL-AF = %s" %np.median(stop[3]))
print("Median of stopping time of OFUL-SF = %s" %np.median(stop[4]))

# print("Mean of best finding time of RS = %s" %np.mean(best_time[0]))
# print("Mean of best finding time of BO-AF = %s" %np.mean(best_time[1]))
# print("Mean of best finding time of BO-SF = %s" %np.mean(best_time[2]))
# print("Mean of best finding time of OFUL-AF = %s" %np.mean(best_time[3]))
# print("Mean of best finding time of OFUL-SF = %s" %np.mean(best_time[4]))

# print("Mean of stopping time of RS = %s" %np.mean(stop[0]))
# print("Mean of stopping time of BO-AF = %s" %np.mean(stop[1]))
# print("Mean of stopping time of BO-SF = %s" %np.mean(stop[2]))
# print("Mean of stopping time of OFUL-AF = %s" %np.mean(stop[3]))
# print("Mean of stopping time of OFUL-SF = %s" %np.mean(stop[4]))

#%% Independent T test of the mean
all_best_time = []  
all_stopping_time = []  
for p in range(0,4):
    path = all_paths[p]
    best_time_ = pd.read_csv(r"%s/time_to_get_best.csv" %path, index_col=0)["0"].tolist()
    stopping_time_ = pd.read_csv(r"%s/stopping_time.csv" %path, index_col=0)["0"].tolist()    
    all_best_time.append(best_time_)
    all_stopping_time.append(stopping_time_)

print("Test of the Mean of best finding time")
print("Random search Vs BO = %s" %stats.ttest_ind(all_best_time[0], all_best_time[1], equal_var = False)[1])
print("Random search Vs OFUL-AF = %s" %stats.ttest_ind(all_best_time[0], all_best_time[2], equal_var = False)[1])
print("Random search Vs OFUL-SF = %s" %stats.ttest_ind(all_best_time[0], all_best_time[3], equal_var = False)[1])
print("BO Vs OFUL-AF = %s" %stats.ttest_ind(all_best_time[1], all_best_time[2], equal_var = False)[1])
print("BO Vs OFUL-SF = %s" %stats.ttest_ind(all_best_time[1], all_best_time[3], equal_var = False)[1])
print("OFUL-AF Vs OFUL-SF = %s" %stats.ttest_ind(all_best_time[2], all_best_time[3], equal_var = False)[1])

print("Test of the Mean of Stopping time")
print("Random search Vs BO = %s" %stats.ttest_ind(all_stopping_time[0], all_stopping_time[1], equal_var = False)[1])
print("Random search Vs OFUL-AF = %s" %stats.ttest_ind(all_stopping_time[0], all_stopping_time[2], equal_var = False)[1])
print("Random search Vs OFUL-SF = %s" %stats.ttest_ind(all_stopping_time[0], all_stopping_time[3], equal_var = False)[1])
print("BO Vs OFUL-AF = %s" %stats.ttest_ind(all_stopping_time[1], all_stopping_time[2], equal_var = False)[1])
print("BO Vs OFUL-SF = %s" %stats.ttest_ind(all_stopping_time[1], all_stopping_time[3], equal_var = False)[1])
print("OFUL-AF Vs OFUL-SF = %s" %stats.ttest_ind(all_stopping_time[2], all_stopping_time[3], equal_var = False)[1])

#%%

# all_paths = ["C:/Users/menha/Desktop/Manuscript result/Manuscript Syn1 RS 50 run/Episode_49",
#               "C:/Users/menha/Desktop/Manuscript result/Manuscript Syn1 BO-AF 50 run/Episode_49",
#               "C:/Users/menha/Desktop/Manuscript result/Manuscript Syn1 BO-SF 50 run/Episode_49",
#               "C:/Users/menha/Desktop/Manuscript result/Manuscript Syn1 OFUL-AF 50 run/Episode_49",
#               "C:/Users/menha/Desktop/Manuscript result/Manuscript Syn1 OFUL-SF 50 run/Episode_49"]  

# all_paths = ["C:/Users/menha/Desktop/Manuscript result/Manuscript Syn2 RS 50 run/Episode_49",
#               "C:/Users/menha/Desktop/Manuscript result/Manuscript Syn2 BO-AF 50 run/Episode_49",
#               "C:/Users/menha/Desktop/Manuscript result/Manuscript Syn2 BO-SF 50 run/Episode_49",
#               "C:/Users/menha/Desktop/Manuscript result/Manuscript Syn2 OFUL-AF 50 run/Episode_49",
#               "C:/Users/menha/Desktop/Manuscript result/Manuscript Syn2 OFUL-SF 50 run/Episode_49"]  

# all_paths = ["C:/Users/menha/Desktop/Manuscript result/Manuscript freesolv RS 50 run/Episode_49",
#               "C:/Users/menha/Desktop/Manuscript result/Manuscript freesolv BO-AF 50 run/Episode_49",
#               "C:/Users/menha/Desktop/Manuscript result/Manuscript freesolv BO-SF 50 run/Episode_49",
#               "C:/Users/menha/Desktop/Manuscript result/Manuscript freesolv OFUL-AF 50 run/Episode_49",
#               "C:/Users/menha/Desktop/Manuscript result/Manuscript freesolv OFUL-SF 50 run/Episode_49"]   

# all_paths = ["C:/Users/menha/Desktop/Manuscript result/Manuscript enan RS 50 run/Episode_49",
#               "C:/Users/menha/Desktop/Manuscript result/Manuscript enan BO-AF 50 run/Episode_49",
#               "C:/Users/menha/Desktop/Manuscript result/Manuscript enan BO-SF 50 run/Episode_49",
#               "C:/Users/menha/Desktop/Manuscript result/Manuscript enan OFUL-AF 50 run/Episode_49",
#               "C:/Users/menha/Desktop/Manuscript result/Manuscript enan OFUL-SF 50 run/Episode_49"]


# all_paths = ["C:/Users/menha/Desktop/Manuscript result/Manuscript Syn1 RS/Episode_19",
#               "C:/Users/menha/Desktop/Manuscript result/Manuscript Syn1 BO-AF default/Episode_19",
#               "C:/Users/menha/Desktop/Manuscript result/Manuscript Syn1 BO-SF default/Episode_19",
#               "C:/Users/menha/Desktop/Manuscript result/Manuscript Syn1 OFUL-AF/Episode_19",
#               "C:/Users/menha/Desktop/Manuscript result/Manuscript Syn1 OFUL-SF/Episode_19"]

# all_paths = ["C:/Users/menha/Desktop/Manuscript result/Manuscript Syn2 RS/Episode_19",
#               "C:/Users/menha/Desktop/Manuscript result/Manuscript Syn2 BO-AF default/Episode_19",
#               "C:/Users/menha/Desktop/Manuscript result/Manuscript Syn2 BO-SF default/Episode_19",
#               "C:/Users/menha/Desktop/Manuscript result/Manuscript Syn2 OFUL-AF/Episode_19",
#               "C:/Users/menha/Desktop/Manuscript result/Manuscript Syn2 OFUL-SF/Episode_19"]

# all_paths = ["C:/Users/menha/Desktop/Manuscript result/Manuscript freesolv RS 50 run/Episode_49",
#               "C:/Users/menha/Desktop/Manuscript result/Manuscript freesolv BO-AF 50 run/Episode_49",
#               "C:/Users/menha/Desktop/Manuscript result/Manuscript freesolv BO-SF 50 run/Episode_49",
#               "C:/Users/menha/Desktop/Manuscript result/Manuscript freesolv OFUL-AF 50 run/Episode_49",
#               "C:/Users/menha/Desktop/Manuscript result/Manuscript freesolv OFUL-SF 50 run/Episode_49"]   

# all_paths = ["C:/Users/menha/Desktop/Manuscript result/Manuscript enan RS 50 run/Episode_49",
#               "C:/Users/menha/Desktop/Manuscript result/Manuscript enan BO-AF 50 run/Episode_49",
#               "C:/Users/menha/Desktop/Manuscript result/Manuscript enan BO-SF 50 run/Episode_49",
#               "C:/Users/menha/Desktop/Manuscript result/Manuscript enan OFUL-AF 50 run/Episode_49",
#               "C:/Users/menha/Desktop/Manuscript result/Manuscript enan OFUL-SF 50 run/Episode_49"]

# all_paths = ["C:/Users/menha/Desktop/Manuscript result/Manuscript Syn1 RS 50 run/Episode_49",
#               "C:/Users/menha/Desktop/Manuscript result/Manuscript Syn1 BO-AF 50 run/Episode_49",
#               "C:/Users/menha/Desktop/Manuscript result/Manuscript Syn1 BO-SF 50 run/Episode_49",
#               "C:/Users/menha/Desktop/Manuscript result/Manuscript Syn1 OFUL-AF 50 run old/Episode_49",
#               "C:/Users/menha/Desktop/Manuscript result/Manuscript Syn1 OFUL-SF 50 run old/Episode_49"]  

# all_paths = ["C:/Users/menha/Desktop/Manuscript result/Manuscript Syn2 RS 50 run/Episode_49",
#               "C:/Users/menha/Desktop/Manuscript result/Manuscript Syn2 BO-AF 50 run/Episode_49",
#               "C:/Users/menha/Desktop/Manuscript result/Manuscript Syn2 BO-SF 50 run/Episode_49",
#               "C:/Users/menha/Desktop/Manuscript result/Manuscript Syn2 OFUL-AF 50 run/Episode_49",
#               "C:/Users/menha/Desktop/Manuscript result/Manuscript Syn2 OFUL-SF 50 run old/Episode_49"]  

# all_paths = ["C:/Users/menha/Desktop/Manuscript result/Manuscript freesolv RS 50 run/Episode_49",
#               "C:/Users/menha/Desktop/Manuscript result/Manuscript freesolv BO-AF 50 run/Episode_49",
#               "C:/Users/menha/Desktop/Manuscript result/Manuscript freesolv BO-SF 50 run/Episode_49",
#               "C:/Users/menha/Desktop/Manuscript result/Manuscript freesolv OFUL-AF 50 run/Episode_49",
#               "C:/Users/menha/Desktop/Manuscript result/Manuscript freesolv OFUL-SF 50 run/Episode_49"]   

# all_paths = ["C:/Users/menha/Desktop/Manuscript result/Manuscript enan RS 50 run/Episode_49",
#               "C:/Users/menha/Desktop/Manuscript result/Manuscript enan BO-AF 50 run/Episode_49",
#               "C:/Users/menha/Desktop/Manuscript result/Manuscript enan BO-SF 50 run/Episode_49",
#               "C:/Users/menha/Desktop/Manuscript result/Manuscript enan OFUL-AF 50 run/Episode_49",
#               "C:/Users/menha/Desktop/Manuscript result/Manuscript enan OFUL-SF 50 run/Episode_49"]


# all_paths = ["C:/Users/menha/Desktop/Review/Review photoswitch RS 50 run/Episode_49",
#               "C:/Users/menha/Desktop/Review/Review photoswitch BO-AF 50 run/Episode_49",
#               "C:/Users/menha/Desktop/Review/Review photoswitch BO-SF 50 run/Episode_49",
#               "C:/Users/menha/Desktop/Review/Review photoswitch OFUL-AF 50 run/Episode_49",
#               "C:/Users/menha/Desktop/Review/Review photoswitch OFUL-SF 50 run/Episode_49"]# all_paths = ["C:/Users/menha/Desktop/Manuscript result/Manuscript Syn1 RS 50 run/Episode_49",
#               #               "C:/Users/menha/Desktop/Manuscript result/Manuscript Syn1 BO-AF 50 run/Episode_49",
#               #               "C:/Users/menha/Desktop/Manuscript result/Manuscript Syn1 BO-SF 50 run/Episode_49",
#               #               "C:/Users/menha/Desktop/Manuscript result/Manuscript Syn1 OFUL-AF 50 run old/Episode_49",
#               #               "C:/Users/menha/Desktop/Manuscript result/Manuscript Syn1 OFUL-SF 50 run old/Episode_49"]  

#               # all_paths = ["C:/Users/menha/Desktop/Manuscript result/Manuscript Syn2 RS 50 run/Episode_49",
#               #               "C:/Users/menha/Desktop/Manuscript result/Manuscript Syn2 BO-AF 50 run/Episode_49",
#               #               "C:/Users/menha/Desktop/Manuscript result/Manuscript Syn2 BO-SF 50 run/Episode_49",
#               #               "C:/Users/menha/Desktop/Manuscript result/Manuscript Syn2 OFUL-AF 50 run/Episode_49",
#               #               "C:/Users/menha/Desktop/Manuscript result/Manuscript Syn2 OFUL-SF 50 run old/Episode_49"]  

#               # all_paths = ["C:/Users/menha/Desktop/Manuscript result/Manuscript freesolv RS 50 run/Episode_49",
#               #               "C:/Users/menha/Desktop/Manuscript result/Manuscript freesolv BO-AF 50 run/Episode_49",
#               #               "C:/Users/menha/Desktop/Manuscript result/Manuscript freesolv BO-SF 50 run/Episode_49",
#               #               "C:/Users/menha/Desktop/Manuscript result/Manuscript freesolv OFUL-AF 50 run/Episode_49",
#               #               "C:/Users/menha/Desktop/Manuscript result/Manuscript freesolv OFUL-SF 50 run/Episode_49"]   

#               # all_paths = ["C:/Users/menha/Desktop/Manuscript result/Manuscript enan RS 50 run/Episode_49",
#               #               "C:/Users/menha/Desktop/Manuscript result/Manuscript enan BO-AF 50 run/Episode_49",
#               #               "C:/Users/menha/Desktop/Manuscript result/Manuscript enan BO-SF 50 run/Episode_49",
#               #               "C:/Users/menha/Desktop/Manuscript result/Manuscript enan OFUL-AF 50 run/Episode_49",
#               #               "C:/Users/menha/Desktop/Manuscript result/Manuscript enan OFUL-SF 50 run/Episode_49"]


#               all_paths = ["C:/Users/menha/Desktop/Review/Review photoswitch RS 50 run/Episode_49",
#                             "C:/Users/menha/Desktop/Review/Review photoswitch BO-AF 50 run/Episode_49",
#                             "C:/Users/menha/Desktop/Review/Review photoswitch BO-SF 50 run/Episode_49",
#                             "C:/Users/menha/Desktop/Review/Review photoswitch OFUL-AF 50 run/Episode_49",
#                             "C:/Users/menha/Desktop/Review/Review photoswitch OFUL-SF 50 run/Episode_49"]







