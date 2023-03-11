
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
from sklearn.linear_model import LinearRegression

def DFA(pp_values, lower_scale_limit, upper_scale_limit):
    scaleDensity = 30 # scales DFA is conducted between lower_scale_limit and upper_scale_limit
    m = 1 # order of polynomial fit (linear = 1, quadratic m = 2, cubic m = 3, etc...)

    # initialize, we use logarithmic scales
    start = np.log(lower_scale_limit) / np.log(10)
    stop = np.log(upper_scale_limit) / np.log(10)
    scales = np.floor(np.logspace(np.log10(math.pow(10, start)), np.log10(math.pow(10, stop)), scaleDensity))
    F = np.zeros(len(scales))
    count = 0

    for s in scales:
        rms = []
        # Step 1: Determine the "profile" (integrated signal with subtracted offset)
        x = pp_values
        y_n = np.cumsum(x - np.mean(x))
        # Step 2: Divide the profile into N non-overlapping segments of equal length s
        L = len(x)
        shape = [int(s), int(np.floor(L/s))]
        nwSize = int(shape[0]) * int(shape[1])
        # beginning to end, here we reshape so that we have a number of segments based on the scale used at this cycle
        Y_n1 = np.reshape(y_n[0:nwSize], shape, order="F")
        Y_n1 = Y_n1.T
        # end to beginning
        Y_n2 = np.reshape(y_n[len(y_n) - (nwSize):len(y_n)], shape, order="F")
        Y_n2 = Y_n2.T
        # concatenate
        Y_n = np.vstack((Y_n1, Y_n2))

        # Step 3: Calculate the local trend for each 2Ns segments by a least squares fit of the series
        for cut in np.arange(0, 2 * shape[1]):
            xcut = np.arange(0, shape[0])
            pl = np.polyfit(xcut, Y_n[cut,:], m)
            Yfit = np.polyval(pl, xcut)
            arr = Yfit - Y_n[cut,:]
            rms.append(np.sqrt(np.mean(arr * arr)))

        if (len(rms) > 0):
            F[count] = np.power((1 / (shape[1] * 2)) * np.sum(np.power(rms, 2)), 1/2)
        count = count + 1

    pl2 = np.polyfit(np.log2(scales), np.log2(F), 1)
    alpha = pl2[0]
    return alpha

def computeFeatures(df):
  features = []
  step = 120
  for index in range(0, int(round(np.max(x)/step))):
      
      array_rr = df.loc[(df['timestamp'] >= (index*step)) & (df['timestamp'] <= (index+1)*step), 'RR']*1000
      # compute heart rate
      heartrate = round(60000/np.mean(array_rr), 2)
      # compute rmssd
      NNdiff = np.abs(np.diff(array_rr))
      rmssd = round(np.sqrt(np.sum((NNdiff * NNdiff) / len(NNdiff))), 2)
      # compute sdnn 
      sdnn = round(np.std(array_rr), 2)
      #dfa, alpha 1
      alpha1 = DFA(array_rr.to_list(), 4, 16)

      curr_features = {
          'timestamp': index,
          'heartrate': heartrate,
          'rmssd': rmssd,
          'sdnn': sdnn,
          'alpha1': alpha1,
      }

      features.append(curr_features)

  features_df = pd.DataFrame(features)
  return features_df

#-------------------------------------------------------------------------------------------------------------------------

# Read the data
df = pd.read_csv('ACR_HRV_Data_3.csv')# Data from 2021/01/01 till present

# create an array of all unique gc_activity_id values
gc_activity_id_list = df['gc_activity_id'].unique()

# find activity_id 7138066139 (start of keto diet) in the list and print index
keto_start_index = int(np.where(gc_activity_id_list == 7138066139)[0])
# find activity_id 8604626257 (end of keto diet) in the list and print index
keto_end_index = int(np.where(gc_activity_id_list == 8604626257)[0])

results = []

# Iterate through the list of gc_activity_id values. 
for gc_activity_id in gc_activity_id_list:
    print(gc_activity_id)
    # for each gc_activity_id in the list create a list of corresponding values in the hrv column
    RRs = df.loc[df['gc_activity_id'] == gc_activity_id, 'hrv'].tolist()
    # convert the list of strings to a list of floats
    RRs = [float(i) for i in RRs]

    # remove values > 1 std from the mean
    mean = np.mean(RRs)
    std = np.std(RRs)
    print(std)

    RRs = [x for x in RRs if (x > mean - std)]
    RRs = [x for x in RRs if (x < mean + std)]
    
    # Correct artifacts - Create a 'filteredRRs' list of values that are within 5% beat to beat difference and ignore the rest
    artifact_correction_threshold = 0.05
    filtered_RRs = []
    for i in range(len(RRs)):
      if RRs[(i-1)]*(1-artifact_correction_threshold) < RRs[i] < RRs[(i-1)]*(1+artifact_correction_threshold):
            filtered_RRs.append(RRs[i])
    
    # Compute the cumulative sum of the filtered RRs to get the timestamps
    x = np.cumsum(filtered_RRs)
    # Create a dataframe with the timestamps and the filtered RRs
    df_1 = pd.DataFrame()
    df_1['timestamp'] = x
    df_1['RR'] = filtered_RRs

    try:
        # Compute features
        features_df = computeFeatures(df_1)
        print(features_df.head())
        length = len(features_df['alpha1'])
        # define linear regression model and fit values
        reg = LinearRegression().fit(features_df['alpha1'].values.reshape(length, 1), features_df['heartrate'].values.reshape(length, 1))
        # predict heartrate value at DFA alpha1 0.75
        prediction = reg.predict(np.array(0.75).reshape(1, 1))
        print(math.floor(prediction))
        # append result to list
        results.append(math.floor(prediction))
    except:
        continue

# Plot the results
plt.scatter(range(len(results)), results, c=results, cmap='viridis', alpha=0.5)
plt.colorbar()
# compute and plot a curved line of best fit
z = np.polyfit(range(len(results)), results, 3)
p = np.poly1d(z)
plt.plot(range(len(results)), p(range(len(results))), "r--")
#plot span of keto diet
plt.axvspan(keto_start_index, keto_end_index, alpha=0.1, color='grey')
#add text to span
plt.text(keto_start_index, 110, 'Keto diet', alpha=0.3, color='grey',rotation=90)
#labels
plt.xlabel('Activity')
plt.ylabel('Predicted aerobic threshold heartrate')
#title
plt.title('Aerobic threshold heartrate derived from DFA alpha1')
plt.show()

