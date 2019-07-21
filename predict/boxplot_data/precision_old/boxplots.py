import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def color_box(bp, color):

    # Define the elements to color. You can also add medians, fliers and means
    elements = ['boxes','caps','whiskers']

    # Iterate over each of the elements changing the color
    for elem in elements:
        [plt.setp(bp[elem][idx], color=color) for idx in xrange(len(bp[elem]))]
    return

files = ['crashes_precision.csv','scandariatost_precision.csv','shinst_precision.csv','zimmermannst_precision.csv','scandariato_crashesst_precision.csv','softwaremetrics_crashes_precision.csv','textmining_softwaremetricsst_precision.csv','everythingst_precision.csv']

init = 0

for file in files:
  data = pd.read_csv(file, sep=',',header=None)
  if init == 0:
    total_data = data.values
    init = 1
  else:
    total_data = np.concatenate((total_data, data.values), axis=1)

#a = np.random.uniform(0,10,[100,1])    
#print type(a)

plt.gcf().subplots_adjust(bottom=0.25)

plt.ylabel('Precision')
plt.xlabel('Models')
labels = ['Crashes','Text Mining','SM - Churn/Complexity','SM - Broad','Crashes+Text Mining','Crashes+Software Metrics','Text Mining+Software Metrics','All']
plt.xticks(range(-7, 0, 1), labels, rotation=30, fontsize=10, ha='right')

bp = plt.boxplot(total_data)
color_box(bp, 'black')
plt.savefig('figure.pdf')



