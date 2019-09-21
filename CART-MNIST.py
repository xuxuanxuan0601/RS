# MNIST手写数字分类(CART)
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


# 加载数据
digits = load_digits()
data = digits.data

# 分割数据
train_x, test_x, train_y, test_y = train_test_split(data, digits.target, test_size=0.25, random_state=33)

# 规范化
ss = preprocessing.StandardScaler()
train_ss_x = ss.fit_transform(train_x)
test_ss_x = ss.transform(test_x)

# 创建分类器
dt_classifier = DecisionTreeClassifier()
dt_classifier.fit(train_ss_x,train_y)
predict_y = dt_classifier.predict(test_ss_x)
print('决策树(CART)准确率: %0.4lf' % accuracy_score(predict_y, test_y))
