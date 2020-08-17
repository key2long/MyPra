import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score


class Model:

    def __init__(self, feature_file):
        self.feature_file = feature_file
        self.features = []
        self.labels = []
        self.coef = []  # coef_和intercept_都是模型参数，即为w coef_为w1到w4 intercept_为w0
        self.test_result = []
        self.path_num = 0
        self.data_process()
        self.path_ids = [n for n in range(1620)]
        self.test_accuracy = 0

    def data_process(self):
        with open(self.feature_file, 'r') as f:
            datas = f.readlines()
            for data in datas:
                data = eval(data.strip().split('\t')[1])
                for n, d in enumerate(data):
                    data[n] = float(data[n])
                if len(data) == 1621:
                    self.labels.append(data[0])
                    self.features.append(data[1:])

    def train(self, stop_loss=0.01, max_iter=10000):
        self.model = LogisticRegression(C=0.5, random_state=0, penalty='l1', class_weight='balanced',
                                        solver='saga', tol=stop_loss, max_iter=max_iter, verbose=1)
        X_train, X_test, y_train, y_test = train_test_split(self.features, self.labels, test_size=0.3, random_state=0)
        self.model.fit(X_train, y_train)
        self.model.predict_proba([X_test[0]])
        self.test_result = precision_recall_fscore_support(y_test, self.model.predict(X_test))
        self.coef = self.model.coef_[0]
        self.test_accuracy = accuracy_score(y_test, self.model.predict(X_test))
        return

    def save(self, model_file, result_file='result.txt'):
        with open(model_file, 'wb') as f:
            pickle.dump(self.model, f)

        with open(result_file, 'w') as f:
            for result in self.test_result:
                f.write(str(result) + '\n')
            f.write('\n\n\n')
            for n, c in enumerate(self.coef):
                f.write(str(self.path_ids[n]) + '\t' + str(c) + '\n')
        return




if __name__=='__main__':
    path = './pro/'
    model = Model(feature_file='./train_data.txt')
    model.train(stop_loss=0.0001, max_iter=100000)
    model.save(path+'model_1.pkl', path+'result_1.txt')
    print(model.test_accuracy)
    '''model.path_selection(threshold=0.01)
    model.retrain()
    model.save(path+'model_2.pkl', path+'result_2.txt')
    model.path_selection(threshold=0.1)
    model.retrain()
    model.save(path+'model_3.pkl', path+'result_3.txt')
    model.path_selection(threshold=0.1)
    model.retrain()
    model.save(path+'model_4.pkl', path+'result_4.txt')'''