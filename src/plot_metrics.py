import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')

#%matplotlib inline

#data from best model performances
x = ['LogisticRegression', 'MultinomialNB', 'RandomForestClassifer', 'GradientBoostingClassifier']

accuracy = [0.776, .780, .782, .770]
auc = [.946, .950, .950, .950]

weighted_avg_precision = [.776, .780, .782, .779]
phish_precision = [.74, .74, .74, .67]
gd_precision = [.75, .76, .75, .74]
pf_precision = [.80, .82, .81, .85]
beatles_precision = [.81, .81, .83, .86]

weighted_avg_recall = [.776, .780, .782, .770]
phish_recall = [.78, .80, .79, .82]
gd_recall = [.73, .71, .73, .72]
pf_recall = [.79, .78, .80, .76]
beatles_recall = [.80, .82, .81, .77]

#accuracy
plt.style.use('ggplot')
fig, ax = plt.subplots(figsize=(12, 8))
ax.scatter(x, accuracy, color='black', marker= 'x', linewidth=3, label='Model Accuracy')
ax.set_xlabel('Model', fontsize=14)
ax.set_ylabel('Accuracy Score', fontsize=14)
ax.set_title('Accuracy Comparison Across Models', fontsize=16)
ax.legend(loc='upper left')
ax.set_ylim(0.7, 0.8)
plt.xticks(fontsize=14)
# plt.savefig('images/accuracy_plot.png', bbox_inches = 'tight')


#roc auc
fig, ax = plt.subplots(figsize=(12, 8))
ax.scatter(x, auc, color='black', marker= 'x', linewidth=3, label='Model ROC AUC')
ax.set_xlabel('Model', fontsize=14)
ax.set_ylabel('ROC AUC Score', fontsize=14)
ax.set_title('ROC Area Under Curve Comparison Across Models', fontsize=16)
ax.legend(loc='upper left')
ax.set_ylim(.92, .96)
plt.xticks(fontsize=14)
# plt.savefig('images/roc_plot.png', bbox_inches = 'tight')

#precision
fig, ax = plt.subplots(figsize=(12, 8))
ax.scatter(x, weighted_avg_precision, color='black', marker='x', linewidth=3, label='Weighted Average Precision')
ax.scatter(x, phish_precision, color='red', ls='--', linewidth=2, label='Phish Precision')
ax.scatter(x, gd_precision, color='blue', ls='--', linewidth=2, label='Grateful Dead Precision')
ax.scatter(x, pf_precision, color='green', ls='--', linewidth=2, label='Pink Floyd Precision')
ax.scatter(x, beatles_precision, color='purple', ls='--', linewidth=2, label='Beatles Precision')
ax.set_xlabel('Model', fontsize=14)
ax.set_ylabel('Precision Score', fontsize=14)
ax.set_title('Precision Comparison Across Models', fontsize=16)
ax.legend(loc='upper left', fontsize=12)
plt.xticks(fontsize=14)
plt.ylim(.6, .9)
# plt.savefig('images/precision_plot.png', bbox_inches = 'tight')

#recall
fig, ax = plt.subplots(figsize=(12, 8))
ax.scatter(x, phish_recall, color='red', ls='--', linewidth=2, label='Phish Recall')
ax.scatter(x, gd_recall, color='blue', ls='--',  linewidth=2, label='Grateful Dead Recall')
ax.scatter(x, pf_recall, color='green', ls='--',linewidth=2, label='Pink Floyd Recall')
ax.scatter(x, beatles_recall, color='purple', ls='--', linewidth=2, label='Beatles Recall')
ax.scatter(x, weighted_avg_recall, color='black', linewidth=2, marker='x', label='Weighted Average Recall')
ax.set_xlabel('Model', fontsize=14)
ax.set_ylabel('Recall Score', fontsize=14)
ax.set_title('Recall Comparison Across Models', fontsize=16)
ax.legend(loc='upper left', fontsize=12)
plt.xticks(fontsize=14)
plt.ylim(.6, .9)
# plt.savefig('images/recall_plot.png', bbox_inches = 'tight')
