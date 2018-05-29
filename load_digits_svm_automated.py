from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

digits=load_digits()
train_data,test_data,train_target,test_target=train_test_split(digits.data,digits.target,test_size=0.001)


clf=SVC()

trained=clf.fit(train_data,train_target)

output=trained.predict(test_data)
print(test_target)
print(output)
# plt.imshow(digits.images[1]
# plt.show()
accuracy=accuracy_score(test_target,output)
print(accuracy)
