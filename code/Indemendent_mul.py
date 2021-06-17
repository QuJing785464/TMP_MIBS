from GetSamples_mul import GetSamples
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
import numpy as np

class Validation():
    
    def main(self, win, rate, n_estimators, classifier, threshold):
        data = GetSamples()
        method = Classifier()                
        x_train, y_train = data.main(path_id_train, mul_label_train, path_label_train, path_fea_train, feature, win, rate, Undersample='True')
        x_test, y_test = data.main(path_id_test, mul_label_test, path_label_test, path_fea_test, feature, win, rate, Undersample='False')
        if classifier == 'rf':
            y_pre = method.RF(x_train, y_train, x_test, n_estimators, threshold)
        elif classifier == 'svm':
            y_pre = method.SVM(x_train, y_train, x_test)
        elif classifier == 'nb':
            y_pre = method.NB(x_train, y_train, x_test)
        elif classifier == 'ada':
            y_pre = method.Ada(x_train, y_train, x_test)
        elif classifier == 'XGBoost':
            y_pre = method.XGBoost(x_train, y_train, x_test)
        else:
            print('classifier type error')        
        ACC, Spe, Sen, MCC, AUC = self.evaluate(y_test, y_pre)
        return ACC, Spe, Sen, MCC, AUC

    def evaluate(self, y_true, y_pred):
        
        TN,FP,FN,TP = confusion_matrix(y_true, y_pred, labels = [0,1]).ravel()
        Spe = round(TN/(TN+FP), 3)
        Sen = round(TP/(TP+FN), 3)
        ACC = round((TP+TN)/(TP+TN+FP+FN), 3)
        #ACC = accuracy_score(y_true,y_pred)
        MCC = round(matthews_corrcoef(y_true, y_pred), 3)
        AUC = round(roc_auc_score(y_true, y_pred), 3)
        print('---Evaluation---')
        print('TP='+str(TP)+'  FP='+str(FP)+'  TN='+str(TN)+'  FN='+str(FN))
        print('ACC='+str(ACC)+'  Spe='+str(Spe)+'  Sen='+str(Sen)+'  MCC='+str(MCC)+'  AUC='+str(AUC))
        return ACC, Spe, Sen, MCC, AUC


class Classifier():

    def SVM(self, fea_train, label_train, fea_test):
        print('classifier: SVM')
        clf = svm.SVC(C=10, gamma=0.1, kernel='poly', probability=True)
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
        print('classifier: Naive Bayes')
        clf = GaussianNB()
        clf.fit(fea_train, label_train)
        label_predict = list(clf.predict(fea_test))
        return label_predict
    
    def Ada(self, fea_train, label_train, fea_test):
        print('classifier: AdaBoost')
        clf = AdaBoostClassifier(algorithm='SAMME.R', n_estimators=50, learning_rate=0.8)
        clf.fit(fea_train, label_train)
        label_predict = list(clf.predict(fea_test))
        return label_predict
    
    def XGBoost(self, fea_train, label_train, fea_test):
        print('classifier: XGBoost')
        clf = GradientBoostingClassifier(n_estimators=140, learning_rate=0.8, max_depth=6, random_state=0)
        clf.fit(fea_train, label_train)
        label_predict = list(clf.predict(fea_test))
        return label_predict     
    
                               
        
if __name__ == '__main__':

    path_id_train = '../data/train/id_train.pic'
    path_label_train = '../data/train/label_binary_train.pic'
    path_fea_train = '../feature/train/'
    path_id_test = '../data/test/id_test.pic'
    path_label_test = '../data/test/label_binary_test.pic'
    path_fea_test = '../feature/test/'
    feature = 'PSSM,PCP553,RASA,Topo,Zcoord'
    N = 10
    classifier = 'rf'
    n_estimators = [1800, 2700, 2200, 500, 700, 800, 200]
    threshold = 0.52
    ions = ['K', 'Ca', 'Na', 'Zn', 'Mg', 'Hg', 'Others']  # 3,15,3,7,13,7,3,13
    list_rate = [2, 1, 3, 5, 4, 3, 4]
    w = [3, 15, 3, 7, 13, 7, 3]
    for i in range(0, len(ions)):
        ion = ions[i]
        win = w[i]
        rate = list_rate[i]
        mul_label_train = '../data/train/'+ion+'_label_train.pic'
        mul_label_test = '../data/test/'+ion+'_label_test.pic'
        print('-------', ion, 'ION Done', '-------')

        test = Validation()
        ACC, Spe, Sen, MCC, AUC = test.main(win, rate,  n_estimators[i], classifier, threshold)
