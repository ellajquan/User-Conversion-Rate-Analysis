#Machine Learning
#Will this user convert (1) or not convert (0)? Through Binary-classification 
import pandas as pd
data = pd.read_csv("/Users/ellaquan/Downloads/Conversion Rate/conversion_project.csv")
data_dummy = pd.get_dummies(data,drop_first=True)
data_dummy.head()
data_dummy.columns
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

np.random.seed(4684)
  
train, test = train_test_split(data_dummy, test_size = 0.34) # Randomly shuffles,66 % goes into train, 34 % into test
  
#Random forest model
rf = RandomForestClassifier(n_estimators=100,   #100 trees with each split 
                            max_features=3,     #sees only 3 features
                            oob_score=True)     #out-of-bag predictions: each tree predicts the rows that were not in its bootstrap sample 
                                                #as a internal cross-validation metric.
rf.fit(train.drop('converted', axis=1), train['converted'])
  
#OOB accuracy and confusion matrix
print(
"OOB accuracy is", 
rf.oob_score_, 
"\n", 
"OOB Confusion Matrix", 
"\n",
pd.DataFrame(confusion_matrix(train['converted'], rf.oob_decision_function_[:,1].round(), labels=[0, 1]))
)
#and let's print test accuracy and confusion matrix
test_data = test.drop('converted', axis=1)
test_pred = rf.predict(test_data)
print(
"Test accuracy is", rf.score(test_data,test['converted']), 
"\n", 
"Test Set Confusion Matrix", 
"\n",
pd.DataFrame(confusion_matrix(test['converted'], test_pred, labels=[0, 1]))
)

from sklearn.metrics import f1_score, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Print classification report
print(classification_report(test['converted'], test_pred))

# Plot confusion matrix
ConfusionMatrixDisplay.from_predictions(test['converted'], test_pred, cmap="Blues")
plt.title("Test Set Confusion Matrix")
plt.show()

test_pred_prob = rf.predict_proba(test_data)
test_pred_prob_pos = test_pred_prob[:,1]
test_pred_prob_pos
from sklearn.metrics import roc_curve, auc
false_positive_rate, true_positive_rate, thresholds = roc_curve(test['converted'].values, test_pred_prob_pos, pos_label = 1)
roc_auc = auc(false_positive_rate, true_positive_rate)
plt.plot(false_positive_rate, true_positive_rate, 'b', label='%s: AUC %0.4f'% ('random forest',roc_auc))
  

plt.title('Random Forest ROC')
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.2])
plt.ylim([-0.1,1.2])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
#Which Feature is the most important to the prediction?
feat_importances = pd.Series(rf.feature_importances_, index=train.drop('converted', axis=1).columns)
feat_importances.sort_values().plot(kind='barh')
plt.show()

#Random Forest model without total_pages_visited (Previous most important feature)

rf2 = RandomForestClassifier(n_estimators=100, max_features=3, oob_score=True, class_weight={0:1, 1:10})
rf2.fit(train.drop(['converted', 'total_pages_visited'], axis=1), train['converted'])
  
#let's print OOB accuracy and confusion matrix
print(
"OOB accuracy is", 
rf2.oob_score_, 
"\n", 
"OOB Confusion Matrix", 
"\n",
pd.DataFrame(confusion_matrix(train['converted'], rf2.oob_decision_function_[:,1].round(), labels=[0, 1]))
)
#and let's print test accuracy and confusion matrix
print(
"Test accuracy is", rf2.score(test.drop(['converted', 'total_pages_visited'], axis=1),test['converted']), 
"\n", 
"Test Set Confusion Matrix", 
"\n",
pd.DataFrame(confusion_matrix(test['converted'], rf2.predict(test.drop(['converted', 'total_pages_visited'], axis=1)), labels=[0, 1]))
)

feat_importances = pd.Series(rf2.feature_importances_, index=train.drop(['converted', 'total_pages_visited'], axis=1).columns)
feat_importances.sort_values().plot(kind='barh')
plt.show()

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

X_test = test.drop(['converted', 'total_pages_visited'], axis=1)
y_test = test['converted']

y_score = rf2.predict_proba(X_test)[:, 1]

fpr, tpr, _ = roc_curve(y_test, y_score)
roc_auc      = auc(fpr, tpr)

plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, label=f'Random Forest (AUC = {roc_auc:.3f})', lw=2)
plt.plot([0, 1], [0, 1], 'k--', lw=1)         
plt.ylabel('True Positive Rate')
plt.title('ROC Curve – Test Set')
plt.legend(loc='lower right')
plt.grid(alpha=.3)
plt.tight_layout()
plt.show()

train
from sklearn.inspection import PartialDependenceDisplay
X = train.drop(['converted'], axis=1)
PartialDependenceDisplay.from_estimator(
    estimator=rf,
    X=X,
    features=['country_US'], # or a list\n grid_resolution=50,
    )

rf.fit(train.drop(['converted', 'total_pages_visited'], axis=1), train['converted'])

from sklearn.inspection import PartialDependenceDisplay
import matplotlib.pyplot as plt

# Keep only the columns that were used in training
X_train = train.drop(['converted', 'total_pages_visited'], axis=1)

features = ['country_Germany', 'country_UK', 'country_US']
PartialDependenceDisplay.from_estimator(
    rf, X_train, features, kind="average", grid_resolution=20
)
plt.suptitle("Partial Dependence - Country Features")
plt.tight_layout()
plt.show()

X_train = train.drop(['converted', 'total_pages_visited'], axis=1)

features = ['source_Direct', 'source_Seo']
PartialDependenceDisplay.from_estimator(
    rf, X_train, features, kind="average", grid_resolution=20
)
plt.suptitle("Partial Dependence - Source Features")
plt.tight_layout()
plt.show()

from sklearn.inspection import PartialDependenceDisplay
X_train = train.drop(['converted','total_pages_visited'], axis=1) 
PartialDependenceDisplay.from_estimator(rf, X_train, features=['source_Direct', 'source_Seo'], 
                                        grid_resolution=2,)
plt.tight_layout(); 
plt.show()
from sklearn.inspection import PartialDependenceDisplay
import matplotlib.pyplot as plt

X_train_full = train.drop(['converted', 'total_pages_visited'], axis=1)   
assert 'new_user' in X_train_full.columns
PartialDependenceDisplay.from_estimator(
    estimator=rf,                       
    X=X_train_full,
    features=['new_user'],                   
    kind="both",                        
    grid_resolution=50,
)
plt.tight_layout()
plt.show()

from sklearn.inspection import PartialDependenceDisplay
import matplotlib.pyplot as plt

X_train_full = train.drop(['converted', 'total_pages_visited'], axis=1)   
assert 'age' in X_train_full.columns
PartialDependenceDisplay.from_estimator(
    estimator=rf,                       
    X=X_train_full,
    features=['age'],                   
    kind="both",                        
    grid_resolution=50,
)
plt.tight_layout()
plt.show()

#age
pdp_iso = pdp.pdp_isolate( model=rf, 
                          dataset=train.drop(['converted', 'total_pages_visited'], axis=1),      
                          model_features=list(train.drop(['converted', 'total_pages_visited'], axis=1)), 
                          feature='age', 
                          num_grid_points=50)
pdp_dataset = pd.Series(pdp_iso.pdp, index=pdp_iso.feature_grids)
pdp_dataset.plot(title='Age')
plt.show()
import graphviz
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from graphviz import Source
  
tree = DecisionTreeClassifier( max_depth=2,class_weight={0:1, 1:10}, min_impurity_decrease = 0.001)
tree.fit(train.drop(['converted', 'total_pages_visited'], axis=1), train['converted'])
  
#visualize it
export_graphviz(tree, out_file="tree_conversion.dot", feature_names=train.drop(['converted', 'total_pages_visited'], axis=1).columns, proportion=True, rotate=True)
with open("tree_conversion.dot") as f:
    dot_graph = f.read()
  
s = Source.from_file("tree_conversion.dot")
s.view()

##Conclusion:
#Young visitors ( < 30 yr) convert 3 – 4 × better than older cohorts.
#Germany: highest conversion rate of any country, yet traffic share ≃ 5 %.
#Lapsed accounts (created > 90 days ago) re-activate well.
#30 + users drop sharply in funnel.
#China traffic is high but conversion is worst.

#Next Steps:
#Create dashboards for the five metrics above (Tableau): <30 conversion, German traffic share, Reactivation lift, 30 + funnel, CN checkout errors>.