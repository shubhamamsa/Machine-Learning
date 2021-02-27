import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix 
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn import tree
from matplotlib import pyplot as plt

car_eval_data = pd.read_csv(r'https://raw.githubusercontent.com/shubhamamsa/'+\
							'Machine-Learning-Datasets/main/car_eval.csv')

indp_ftr = car_eval_data.drop('Status', axis=1)
dpnd_ftr = car_eval_data.Status

indp_train, indp_test, dpnd_train, dpnd_test = train_test_split(indp_ftr, \
																dpnd_ftr, \
																test_size = 0.28, \
																random_state = 40)

dt_model = DecisionTreeClassifier(criterion='entropy', max_depth=10)
dt_model.fit(indp_train, dpnd_train)

pred = dt_model.predict(indp_test)

accuracy = metrics.accuracy_score(dpnd_test, pred)*100
cf_matrix = confusion_matrix(dpnd_test, pred)
depth = dt_model.tree_.max_depth
fig = plt.figure(figsize=(25,30))
_ = tree.plot_tree(dt_model, filled=True)
fig.savefig("decistion_tree.png")

print("\nConfusion Matrix: \n")
for i in cf_matrix:
	print(i)

print("\nAccuracy:", end=" ")
print("{percentage}%\n".format(percentage = round(accuracy, 3)))
print("Decision Tree Depth:", end=" ")
print(depth)
print()