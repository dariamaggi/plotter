import numpy as np
import matplotlib.pyplot as plt

# data to plot
n_groups = 4
means_accuracy = (0.756, 0.744, 0.756, 0.734)
means_precision = (0.756, 0.744, 0.756, 0.734)
means_recall = (0.756, 0.744, 0.756, 0.734)
means_feature =(0.756, 0.744, 0.756, 0.734)
# create plot
fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.10
opacity = 0.8

rects1 = plt.bar(index, means_accuracy, bar_width,
alpha=opacity,
color='#9ef949',
label='Accuracy')

rects2 = plt.bar(index + 1.5*bar_width, means_recall, bar_width,
alpha=opacity,
color='#f96849',
label='Precision')

rects3 = plt.bar(index + 3.5*bar_width, means_precision, bar_width,
alpha=opacity,
color='#49bef9',
label='Recall')

rects4 = plt.bar(index + 5.5*bar_width, means_feature, bar_width,
alpha=opacity,
color='#de49f9',
label='F-measure')

plt.xlabel('Classifier')
plt.ylabel('Scores')
plt.title('Scores by classifier')
plt.xticks(index + bar_width, ('J48', 'SMO', 'RainForest', 'Naive Bayes'))
plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')


plt.tight_layout()
plt.show()