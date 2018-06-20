'''
Created on 05/06/2017

@author: Jaime Sacramento Perez Gutierrez

Clasifier of shoes based on genders using three learning models: 
Tree, Stochastic Gradient Decent and Naive Bayes
'''

from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model.stochastic_gradient import SGDClassifier

#List of list x. Attributes: height, weight, shoe size

Listx= [[188,57,30], [167,32,22],[193,65,29],[185,53,27],[164,45,22],[157,38,24],[179,52,27],[175,68,26],
[167,39,24],[178,62,27],[158,46,26]]

#List of labels Y. Gender female or male

Listy=['male','female','male','male','female','female','male','male','female',
       'male','female']

#Model to store the decision tree model: Clasifier

Clasifier_tree = tree.DecisionTreeClassifier()
Clasifier_Sgradient = SGDClassifier()
Clasifier_naive = GaussianNB()
#Training stage 
Clasifier_tree = Clasifier_tree.fit(Listx,Listy)
Clasifier_Sgradient = Clasifier_Sgradient.fit(Listx,Listy)
Clasifier_naive = Clasifier_naive.fit(Listx,Listy)
 
#Test stage
Listz = [[150,35,21]]
Prediction_tree = Clasifier_tree.predict(Listz)
Prediction_Gradient = Clasifier_Sgradient.predict(Listz)
Prediction_naive = Clasifier_naive.predict(Listz) 

print(Prediction_tree)
print(Prediction_Gradient)
print(Prediction_naive)
