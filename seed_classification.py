import numpy as np
import matplotlib.pyplot as plt
with open('seeds_dataset.txt','r') as f:
    data = [[float(x) for x in i.strip().split()] for i in f.readlines()]
    data = np.array(data)

X = data[:,:-1]
Y = data[:,-1]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)


y_pred = classifier.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))


from sklearn.metrics import confusion_matrix, accuracy_score
confusion = confusion_matrix(y_test, y_pred)
print(confusion)
print(accuracy_score(y_test, y_pred))