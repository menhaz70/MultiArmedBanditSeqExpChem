
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


all_paths = ["C:/Users/menha/Desktop/corrected_ex/Review photoswitch BO-SF 50 run/Episode_49",
              "C:/Users/menha/Desktop/corrected_ex/Corrected Review photoswitch OFUL-SF 50 run/Episode_49"] 
labels = ["BO-SF", "OFUL-SF"]
figure_path = "C:/Users/menha/OneDrive - 国立大学法人 北海道大学/PhD work/Manuscript working/figure_raw_python/corrected/alljpg/box_photoswitch.jpg"


# all_paths = ["C:/Users/menha/Desktop/corrected_ex/Review freesolv BO-AF 50 run matern kernel/Episode_49",
#               "C:/Users/menha/Desktop/corrected_ex/Manuscript freesolv BO-AF 50 run/Episode_49",
#               "C:/Users/menha/Desktop/corrected_ex/Review freesolv BO-SF 50 run matern kernel/Episode_49",
#               "C:/Users/menha/Desktop/corrected_ex/Manuscript freesolv BO-SF 50 run/Episode_49"] 
# labels = ["BO-AF(Matérn)", "BO-AF(RBF)", "BO-SF(Matérn)", "BO-SF(RBF)"]
# figure_path = "C:/Users/menha/OneDrive - 国立大学法人 北海道大学/PhD work/Manuscript working/figure_raw_python/corrected/alljpg/boxplot_matern_freesolv.jpg"


# all_paths = ["C:/Users/menha/Desktop/corrected_ex/Review enan BO-AF 50 run matern kernel/Episode_49",
#               "C:/Users/menha/Desktop/corrected_ex/Manuscript enan BO-AF 50 run/Episode_49",
#               "C:/Users/menha/Desktop/corrected_ex/Review enan BO-SF 50 run matern kernel/Episode_49",
#               "C:/Users/menha/Desktop/corrected_ex/Manuscript enan BO-SF 50 run/Episode_49"] 
# labels = ["BO-AF(Matérn)", "BO-AF(RBF)", "BO-SF(Matérn)", "BO-SF(RBF)"]
# figure_path = "C:/Users/menha/OneDrive - 国立大学法人 北海道大学/PhD work/Manuscript working/figure_raw_python/corrected/alljpg/boxplot_matern_enan.jpg"


# all_paths = ["C:/Users/menha/Desktop/corrected_ex/Review freesolv_MF BO-SF 50 run/Episode_49",
#               "C:/Users/menha/Desktop/corrected_ex/Manuscript freesolv BO-SF 50 run/Episode_49",
#               "C:/Users/menha/Desktop/corrected_ex/Corrected Review freesolv_MF OFUL-SF 50 run/Episode_49",
#               "C:/Users/menha/Desktop/corrected_ex/Corrected Manuscript freesolv OFUL-SF 50 run 1/Episode_49"] 
# labels = ["BO-SF(MF)", "BO-SF(Frag.)", "OFUL-SF(MF)", "OFUL-SF(Frag.)"]
# figure_path = "C:/Users/menha/OneDrive - 国立大学法人 北海道大学/PhD work/Manuscript working/figure_raw_python/corrected/alljpg/box_morgan_freesolv.jpg"


# all_paths = ["C:/Users/menha/Desktop/corrected_ex/Review enan_MF BO-SF 50 run/Episode_49",
#               "C:/Users/menha/Desktop/corrected_ex/Manuscript enan BO-SF 50 run/Episode_49",
#               "C:/Users/menha/Desktop/corrected_ex/Corrected Review enan_MF OFUL-SF 50 run/Episode_49",
#               "C:/Users/menha/Desktop/corrected_ex/Corrected Manuscript enan OFUL-SF 50 run/Episode_49"] 
# labels = ["BO-SF(MF)", "BO-SF(Frag.)", "OFUL-SF(MF)", "OFUL-SF(Frag.)"]
# figure_path = "C:/Users/menha/OneDrive - 国立大学法人 北海道大学/PhD work/Manuscript working/figure_raw_python/corrected/alljpg/box_morgan_enan.jpg"


len(all_paths)
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
# labels = ["BO-SF", "OFUL-SF"]
# labels = ["BO-SF(Matern, FN)", "BO-SF(Matern, Frag.)"]


# labels = ["OFUL-AF", "OFUL-SF" ]
best_time = []
never_best = []
stop = []
intial_sample = 10



for p in range(len(all_paths)):
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


fig, (ax1,ax2) = plt.subplots(2, 1, figsize=(10,12))     # 10,12
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


#%% Photoswictch data success rate, median best finfing time and stopping time

print("Percentage of success of BO-SF = %s" %((1-(never_best[0]/num_of_episode))*100))
print("Percentage of success of OFUL-SF = %s" %((1-(never_best[1]/num_of_episode))*100))

print("Median of best finding time of BO-SF = %s" %np.median(best_time[0]))
print("Median of best finding time of OFUL-SF = %s" %np.median(best_time[1]))

print("Median of stopping time of BO-SF  = %s" %np.median(stop[0]))
print("Median of stopping time of OFUL-SF = %s" %np.median(stop[1]))

print("Mean of best finding time of BO-SF  = %s" %np.mean(best_time[0]))
print("Mean of best finding time of OFUL-SF = %s" %np.mean(best_time[1]))

print("Mean of stopping time of BO-SF  = %s" %np.mean(stop[0]))
print("Mean of stopping time of OFUL-SF = %s" %np.mean(stop[1]))


#%% kernel comparison

# print("Percentage of success of BO-AF(Matern) = %s" %((1-(never_best[0]/num_of_episode))*100))
# print("Percentage of success of BO-SF(RBF) = %s" %((1-(never_best[1]/num_of_episode))*100))
# print("Percentage of success of OFUL-AF(Matern) = %s" %((1-(never_best[2]/num_of_episode))*100))
# print("Percentage of success of OFUL-SF(RBF) = %s" %((1-(never_best[3]/num_of_episode))*100))


# print("Median of best finding time of BO-AF(Matern) = %s" %np.median(best_time[0]))
# print("Median of best finding time of BO-SF(RBF) = %s" %np.median(best_time[1]))
# print("Median of best finding time of OFUL-AF(Matern) = %s" %np.median(best_time[2]))
# print("Median of best finding time of OFUL-SF(RBF) = %s" %np.median(best_time[3]))


# print("Median of stopping time of BO-AF(Matern)  = %s" %np.median(stop[0]))
# print("Median of stopping time of BO-SF(RBF) = %s" %np.median(stop[1]))
# print("Median of stopping time of OFUL-AF(Matern) = %s" %np.median(stop[2]))
# print("Median of stopping time of OFUL-SF(RBF) = %s" %np.median(stop[3]))

#%% Morgan finger print and Fragment feature compararison

# print("Percentage of success of BO-SF(MF) = %s" %((1-(never_best[0]/num_of_episode))*100))
# print("Percentage of success of BO-SF(Frag.) = %s" %((1-(never_best[1]/num_of_episode))*100))
# print("Percentage of success of OFUL-SF(MF) = %s" %((1-(never_best[2]/num_of_episode))*100))
# print("Percentage of success of OFUL-SF(Frag.) = %s" %((1-(never_best[3]/num_of_episode))*100))


# print("Median of best finding time of BO-SF(MF) = %s" %np.median(best_time[0]))
# print("Median of best finding time of BO-SF(Frag.) = %s" %np.median(best_time[1]))
# print("Median of best finding time of OFUL-SF(MF) = %s" %np.median(best_time[2]))
# print("Median of best finding time of OFUL-SF(Frag.) = %s" %np.median(best_time[3]))


# print("Median of stopping time of BO-SF(MF)  = %s" %np.median(stop[0]))
# print("Median of stopping time of BO-SF(Frag.) = %s" %np.median(stop[1]))
# print("Median of stopping time of OFUL-SF(MF) = %s" %np.median(stop[2]))
# print("Median of stopping time of OFUL-SF(Frag.) = %s" %np.median(stop[3]))

