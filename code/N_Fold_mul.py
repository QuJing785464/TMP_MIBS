from GetSamples_mul import GetSamples
import random
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from prettytable import PrettyTable
from sklearn.model_selection import ShuffleSplit


class Validation():
    
    def main(self,win,rate,N,n_estimators,classifier,ion,Undersample,threshold):

        data = GetSamples()
        X,Y = data.main(path_id, mul_label, binary_label, path_fea, feature, win, rate, Undersample)
        stdsc = StandardScaler()
        X = stdsc.fit_transform(X)
        ACC, Spe, Sen, MCC, AUC = self.run_threshold_model(X, Y, N, n_estimators, classifier, threshold)
        return ACC, Spe, Sen, MCC, AUC

    def run_threshold_model(self, X, y, N, n_estimators, classifier, threshold):

        method = Classifier()
        ss = ShuffleSplit(n_splits=N, test_size=0.2, random_state=0)
        i = 0
        evaluation = []
        print("threshold:{}".format(threshold))
        for train_index, test_index in ss.split(X):
            i += 1
            print(str(i) + ' Done')
            x_train, y_train, x_test, y_test = X[train_index], y[train_index], X[test_index], y[test_index]
            if classifier == 'rf':
                y_pred = method.RF(x_train, y_train, x_test, n_estimators, threshold)
            elif classifier == 'svm':
                y_pred = method.SVM(x_train, y_train, x_test)
            elif classifier == 'nb':
                y_pred = method.NB(x_train, y_train, x_test)
            elif classifier == 'ada':
                y_pred = method.Ada(x_train, y_train, x_test)
            elif classifier == 'XGBoost':
                y_pred = method.XGBoost(x_train, y_train, x_test)
            else:
                print('classifier type error')
            evaluation.append(self.evaluate(y[test_index], y_pred))

        tn, fp, fn, tp, acc, spe, sen, mcc, auc = zip(*evaluation)
        TN, FP, FN, TP = sum(tn), sum(fp), sum(fn), sum(tp)
        ACC, Spe, Sen, MCC, AUC = sum(acc) / N, sum(spe) / N, sum(sen) / N, sum(mcc) / N, sum(auc) / N

        print('---Evaluation---')
        print('TP={}  FP={}  TN={}  FN={}'.format(TP, FP, TN, FN))
        print('ACC={0}  Spe={1}  Sen={2}  MCC={3}  AUC={4}'.format(ACC, Spe, Sen, MCC, AUC))

        return ACC, Spe, Sen, MCC, AUC

    def evaluate(self,y_true, y_pred): 
        
        TN,FP,FN,TP = confusion_matrix(y_true, y_pred, labels = [0, 1]).ravel()
        Spe = TN/(TN+FP)
        Sen = TP/(TP+FN)
        ACC = accuracy_score(y_true, y_pred)
        MCC = matthews_corrcoef(y_true, y_pred)
        AUC = roc_auc_score(y_true, y_pred)
        #print(ACC,Spe,Sen,MCC)
        return TN, FP, FN, TP, ACC, Spe, Sen, MCC, AUC
        
    
class Classifier():
    
    def SVM(self, fea_train, label_train, fea_test):
        clf = svm.SVC(C=2, gamma=0.01, kernel='rbf', probability=True)
        clf.fit(fea_train, label_train)
        label_predict = list(clf.predict(fea_test))
        return label_predict
    
    def RF(self, fea_train, label_train, fea_test, n_estimator, threshold):
        clf = RandomForestClassifier(random_state=39, n_estimators=n_estimator, n_jobs=-1)
        clf.fit(fea_train, label_train)
        pred_array = np.array(clf.predict_proba(fea_test))
        pred_y_new = np.zeros([len(fea_test), 1])
        pred_y_new[pred_array[:, 1] > threshold] = 1
        return pred_y_new
    
    def NB(self, fea_train, label_train, fea_test):
        clf = GaussianNB()      
        clf.fit(fea_train, label_train)
        label_predict = list(clf.predict(fea_test))
        return label_predict
    
    def Ada(self, fea_train, label_train, fea_test):
        clf = AdaBoostClassifier(algorithm='SAMME.R', n_estimators=1000, learning_rate=0.8)
        clf.fit(fea_train, label_train)
        label_predict = list(clf.predict(fea_test))
        return label_predict 
    
    def XGBoost(self, fea_train, label_train, fea_test):
        clf = GradientBoostingClassifier(n_estimators=1000, learning_rate=0.8, max_depth=6, random_state=0)
        clf.fit(fea_train, label_train)
        label_predict = list(clf.predict(fea_test))
        return label_predict         
                                
        
if __name__ == '__main__':

    path_id = '../data/train/id_train.pic'
    binary_label = '../data/train/label_binary_train.pic'
    path_fea = '../feature/train/'
    feature = 'PSSM,PCP553,RASA,Zcoord,Topo'
    N = 10
    n_estimators = [1800, 2700, 2200, 500, 700, 800, 200]
    n_estimators = 200
    threshold = 0.52
    ions = ['K', 'Ca', 'Na', 'Zn', 'Mg', 'Hg', 'Others']  # 3,15,3,7,13,7,3,13
    list_rate =[2, 1, 3, 5, 4, 3, 4]
    w = [3, 15, 3, 7, 13, 7, 3]
    x = PrettyTable(['ION', 'ACC', 'Spe', 'Sen', 'MCC', 'AUC'])
    Undersample = 'True'
    classifier = 'rf'
    for i in range(0, len(ions)):
        ion = ions[i]
        mul_label = '../data/train/'+ion+'_label_train.pic'
        print('-------', ion, 'ION Done', '-------')
        test = Validation()
        win = w[i]
        rate = list_rate[i]
        ACC, Spe, Sen, MCC, AUC = test.main(win, rate, N, n_estimators, classifier, ion, Undersample, threshold)
        # x.add_row([ion[i], ACC, Spe, Sen, MCC, AUC])

    # print('-------------------Evaluation-------------------')
    # print(feature)
    # print(x)
