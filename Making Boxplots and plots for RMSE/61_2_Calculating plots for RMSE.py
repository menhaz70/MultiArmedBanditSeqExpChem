
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

all_paths = ["C:/Users/menha/Desktop/corrected_ex/Manuscript Syn1 RS 50 run/Episode_49",
              "C:/Users/menha/Desktop/corrected_ex/Manuscript Syn1 BO-AF 50 run/Episode_49",
              "C:/Users/menha/Desktop/corrected_ex/Manuscript Syn1 BO-SF 50 run/Episode_49",
              "C:/Users/menha/Desktop/corrected_ex/Corrected Manuscript Syn1 OFUL-AF 50 run/Episode_49",
              "C:/Users/menha/Desktop/corrected_ex/Corrected Manuscript Syn1 OFUL-SF 50 run/Episode_49"] 
figure_path = "C:/Users/menha/OneDrive - 国立大学法人 北海道大学/PhD work/Manuscript working/figure_raw_python/corrected/alljpg/rmse_syn1.jpg"


all_paths = ["C:/Users/menha/Desktop/corrected_ex/Manuscript Syn2 RS 50 run/Episode_49",
              "C:/Users/menha/Desktop/corrected_ex/Manuscript Syn2 BO-AF 50 run/Episode_49",
              "C:/Users/menha/Desktop/corrected_ex/Manuscript Syn2 BO-SF 50 run/Episode_49",
              "C:/Users/menha/Desktop/corrected_ex/Corrected Manuscript Syn2 OFUL-AF 50 run/Episode_49",
              "C:/Users/menha/Desktop/corrected_ex/Corrected Manuscript Syn2 OFUL-SF 50 run/Episode_49"]  
figure_path = "C:/Users/menha/OneDrive - 国立大学法人 北海道大学/PhD work/Manuscript working/figure_raw_python/corrected/alljpg/rmse_syn2.jpg"


# all_paths = ["C:/Users/menha/Desktop/corrected_ex/Manuscript freesolv RS 50 run/Episode_49",
#               "C:/Users/menha/Desktop/corrected_ex/Manuscript freesolv BO-AF 50 run/Episode_49",
#               "C:/Users/menha/Desktop/corrected_ex/Manuscript freesolv BO-SF 50 run/Episode_49",
#               "C:/Users/menha/Desktop/corrected_ex/Corrected Manuscript Freesolv OFUL-AF 50 run/Episode_49",
#               "C:/Users/menha/Desktop/corrected_ex/Corrected Manuscript freesolv OFUL-SF 50 run 1/Episode_49"]   
# figure_path = "C:/Users/menha/OneDrive - 国立大学法人 北海道大学/PhD work/Manuscript working/figure_raw_python/corrected/alljpg/rmse_freesolv.jpg"


# all_paths = ["C:/Users/menha/Desktop/corrected_ex/Manuscript enan RS 50 run/Episode_49",
#               "C:/Users/menha/Desktop/corrected_ex/Manuscript enan BO-AF 50 run/Episode_49",
#               "C:/Users/menha/Desktop/corrected_ex/Manuscript enan BO-SF 50 run/Episode_49",
#               "C:/Users/menha/Desktop/corrected_ex/Corrected Manuscript enan OFUL-AF 50 run/Episode_49",
#               "C:/Users/menha/Desktop/corrected_ex/Corrected Manuscript enan OFUL-SF 50 run/Episode_49"]
# figure_path = "C:/Users/menha/OneDrive - 国立大学法人 北海道大学/PhD work/Manuscript working/figure_raw_python/corrected/alljpg/rmse_enan.jpg"



correlation_average = []
rmse_average = []        #len(rmse_average)
average_find_from = []
standard_error = []

for p in range(0,len(all_paths)):
    path = all_paths[p]
    rmse_ = pd.read_csv(r"%s/rmse_test.csv" %path, index_col=0)
    average_of_each_method = []
    len_of_each_method = []
    standard_error_of_each_method = []
    for j in range(len(rmse_.columns)):
        col_ = (rmse_["%s"%j].tolist())
        
        new_list = []
        for item in col_:
          if str(item) != 'nan' and item !=-2.0:
            new_list.append(item)
        average_of_each_method.append(np.mean(new_list))
        standard_error_of_each_method.append(np.std(new_list)/np.sqrt(len(new_list)))
        len_of_each_method.append(len(new_list))
        
    rmse_average.append(average_of_each_method)
    standard_error.append(standard_error_of_each_method)    
    average_find_from.append(len_of_each_method)


min_stopping_time = []
for p in range(0,5):
    path = all_paths[p]
    stopping_time = pd.read_csv(r"%s/stopping_time.csv" %path, index_col=0)
    min_ = np.nanmin(stopping_time)
    min_stopping_time.append(min_)


# for p in range(0,4):
#     path = all_paths[p]
#     corr_ = pd.read_csv(r"%s/correlation.csv" %path, index_col=0)
#     average_of_each_method = []
#     for j in range(len(corr_.columns)):
#         col_ = (corr_["%s"%j].tolist())
#         average_ = np.nanmean([k for k in col_ if k != -2.0])
#         average_of_each_method.append(average_)
#     correlation_average.append(average_of_each_method)


linewidth = 1.5

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8,6), gridspec_kw={'height_ratios': [3, 2]})  

ax1.plot(range(len(rmse_average[0])), rmse_average[0], label = "RS", color = "black", linewidth=linewidth)
ax1.fill_between(range(len(rmse_average[0])), np.array(rmse_average[0])+np.array(standard_error[0]),\
                 np.array(rmse_average[0])-np.array(standard_error[0]), facecolor="black", alpha=0.3)

ax1.plot(range(len(rmse_average[1])), rmse_average[1], label = "BO-AF", color = "blue", linewidth=linewidth)
ax1.fill_between(range(len(rmse_average[1])), np.array(rmse_average[1])+np.array(standard_error[1]),\
    np.array(rmse_average[1])-np.array(standard_error[1]), facecolor="blue", alpha=0.3)    

ax1.plot(range(len(rmse_average[2])), rmse_average[2], label = "BO-SF", color = "green", linewidth=linewidth)
ax1.fill_between(range(len(rmse_average[2])), np.array(rmse_average[2])+np.array(standard_error[2]),\
    np.array(rmse_average[2])-np.array(standard_error[2]), facecolor="green", alpha=0.3)
    
ax1.plot(range(len(rmse_average[3])), rmse_average[3], label = "OFUL-AF", color = "orange", linewidth=linewidth)
ax1.fill_between(range(len(rmse_average[3])), np.array(rmse_average[3])+np.array(standard_error[3]),\
    np.array(rmse_average[3])-np.array(standard_error[3]), facecolor="orange", alpha=0.3)

ax1.plot(range(len(rmse_average[4])), rmse_average[4], label = "OFUL-SF", color = "red", linewidth=linewidth)
ax1.fill_between(range(len(rmse_average[4])), np.array(rmse_average[4])+np.array(standard_error[4]),\
    np.array(rmse_average[4])-np.array(standard_error[4]), facecolor="red", alpha=0.3)



x_lim = 150    
# ax1.set_xlabel("Time steps", fontsize=20)
ax1.set_ylabel("Average of RMSE", fontsize=20)
ax1.tick_params(axis='x', labelsize=18)
ax1.tick_params(axis='y', labelsize=18)
# ax.set_title("Reward trajectory of Random search", fontsize=20)
ax1.legend(loc='lower left', fontsize=14)
# ax1.legend(loc='upper right', fontsize=14)
# plt.tick_params(left = False, bottom = False)
# plt.xticks([0,10,15,20,30,40,50,60,70])
ax1.set_xlim([0, x_lim])
ax1.grid()

ax2.plot(average_find_from[0], color = "black", linewidth=linewidth)
ax2.plot(average_find_from[1], color = "blue", linewidth=linewidth)
ax2.plot(average_find_from[2], color = "green", linewidth=linewidth)
ax2.plot(average_find_from[3], color = "orange", linewidth=linewidth)
ax2.plot(average_find_from[4], color = "red", linewidth=linewidth)

ax2.set_xlabel("Time steps", fontsize=20)
ax2.set_ylabel("# runs for avg.", fontsize=20)
# ax2.set_ylabel("# runs used to average", fontsize=16)
ax2.tick_params(axis='x', labelsize=18)
ax2.tick_params(axis='y', labelsize=18)
ax2.set_xlim([0, x_lim])
ax2.grid()
plt.tight_layout()
plt.savefig(figure_path)
plt.show()
