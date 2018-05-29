from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC

digits=load_digits()
x=[1,100,200]
train_data=np.delete(digits.data,x,axis=0)
train_target=np.delete(digits.target,x,axis=0)

test_data=digits.data[x]
test_target=digits.target[x]

clf=SVC()

trained=clf.fit(train_data,train_target)

output=trained.predict(digits.data[x].reshape(3,64))
print(output)
plt.imshow(digits.images[1]
plt.show()
