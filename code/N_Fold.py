from GetSamples import GetSamples
from sklearn import svm
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import make_scorer
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_validate
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn.model_selection import ShuffleSplit

class Validation():

    def main(self,win,rate,N,classifier,Undersample,threshold):
        data = GetSamples()
        X,Y = data.main(path_id,path_label,path_fea,feature,win,rate,Undersample)

        ACC,Spe,Sen,MCC,AUC = self.run_threshold_model(X, Y, N, classifier, threshold)

        return ACC,Spe,Sen,MCC,AUC

    def deal_data(self,features,data_new):
        '''处理经过特征选择后的数据，格式为 DataFrame'''
        features.append('target')
        data_to_keep = data_new.loc[:, features]
        label = data_to_keep.target
        Y = list(label.values)
        data = data_to_keep.drop(labels=['target'], axis=1)
        X = data.values
        return X, Y

    def Predict(self, x, y, N, classifier, threshold):
        pos_all, neg_all = [], []
        pos_test, pos_train, neg_test, neg_train = [], [], [], []
        method = Classifier()
        TN, FP, FN, TP = 0, 0, 0, 0
        ACC, Spe, Sen, MCC, AUC = 0.0, 0.0, 0.0, 0.0, 0.0
        list_mcc, list_acc, list_spe, list_sen, list_auc = [], [], [], [], []
        for i in range(0, len(y)):
            if y[i] == 1 or y[i] == '1':
                pos_all.append(i)
            else:
                neg_all.append(i)
        num_pos_test = len(pos_all) // N
        num_neg_test = len(neg_all) // N

        for i in range(0, num_pos_test * N, num_pos_test):
            pos_test.append(pos_all[i:i + num_pos_test])
            pos_train.append(pos_all[0:i + 1] + pos_all[i + num_pos_test + 1:len(pos_all) + 1])
        for i in range(0, num_neg_test * N, num_neg_test):
            neg_test.append(neg_all[i:i + num_neg_test])
            neg_train.append(neg_all[0:i + 1] + neg_all[i + num_neg_test + 1:len(neg_all) + 1])


        for n in range(0, N):
            print(str(n) + ' Done')
            x_train, y_train, x_test, y_test = [], [], [], []

            for i in range(0, len(pos_train[n])):
                index = pos_train[n][i]
                x_train.append(x[index])
                y_train.append(1)
            for i in range(0, len(neg_train[n])):
                index = neg_train[n][i]
                x_train.append(x[index])
                y_train.append(0)
            for i in range(0, len(pos_test[n])):

                index = pos_test[n][i]
                x_test.append(x[index])
                y_test.append(1)
            for i in range(0, len(neg_test[n])):
                index = neg_test[n][i]
                x_test.append(x[index])
                y_test.append(0)

            if classifier == 'rf':
                y_pred = method.RF(x_train, y_train, x_test,n_estimators)
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
            tn, fp, fn, tp, acc, spe, sen, mcc, auc = self.evaluate(y_test, y_pred)
            TP += tp
            FP += fp
            TN += tn
            FN += fn
            ACC += acc
            Spe += spe
            Sen += sen
            MCC += mcc
            AUC += auc

        ACC = round(ACC / N, 3)
        Spe = round(TN / (TN + FP), 3)
        Sen = round(TP / (TP + FN), 3)
        MCC = round(MCC / N, 3)
        AUC = round(AUC / N, 3)
        list_acc.append(ACC)
        list_sen.append(Sen)
        list_spe.append(Spe)
        list_mcc.append(MCC)
        list_auc.append(AUC)
        print('---Evaluation---')
        print('TP=' + str(TP) + '  FP=' + str(FP) + '  TN=' + str(TN) + '  FN=' + str(FN))
        print('ACC=' + str(ACC) + '  Spe=' + str(Spe) + '  Sen=' + str(Sen) + '  MCC=' + str(MCC) + '  AUC=' + str(AUC))
        return ACC, Spe, Sen, MCC, AUC

    def run_threshold_model(self, X, y, N, classifier, threshold):

        method = Classifier()
        ss = ShuffleSplit(n_splits=N, test_size=0.2, random_state=0)
        threshold = 0.52
        i = 0
        evaluation = []
        print("threshold:{} Done".format(threshold))
        for train_index, test_index in ss.split(X):
            i += 1
            print(str(i) + ' Done')
            x_train, y_train, x_test, y_test = X[train_index], y[train_index], X[test_index], y[test_index]
            if classifier == 'rf':
                y_pred = method.RF(x_train, y_train, x_test, threshold)
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
                
    def evaluate(self,y_true, y_pred): 
        
        TN, FP, FN, TP = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        Spe = TN/(TN+FP)
        Sen = TP/(TP+FN)
        ACC = accuracy_score(y_true, y_pred)
        MCC = matthews_corrcoef(y_true, y_pred)
        AUC = roc_auc_score(y_true, y_pred)
        print(ACC,Spe,Sen,MCC,AUC)
        return TN,  FP, FN, TP, ACC, Spe, Sen, MCC, AUC
        
    
class Classifier():
    
    def SVM(self,fea_train,label_train,fea_test):
        clf = svm.SVC(C=10, gamma=0.1, kernel='rbf',probability=True)
        clf.fit(fea_train,label_train)
        label_predict = list(clf.predict(fea_test))
        return label_predict
    
    def RF(self, fea_train, label_train, fea_test, threshold):
        clf = RandomForestClassifier(random_state=39, n_estimators=200, n_jobs=-1)
        clf.fit(fea_train, label_train)
        pred_array = np.array(clf.predict_proba(fea_test))
        pred_y_new = np.zeros([len(fea_test), 1])
        pred_y_new[pred_array[:, 1] > threshold] = 1
        return pred_y_new
    
    def NB(self,fea_train,label_train,fea_test):
        clf = GaussianNB()      
        clf.fit(fea_train,label_train)
        label_predict = list(clf.predict(fea_test))
        return label_predict
    
    def Ada(self,fea_train,label_train,fea_test):
        clf = AdaBoostClassifier(algorithm='SAMME.R',n_estimators=200,learning_rate=0.8)
        clf.fit(fea_train,label_train)
        label_predict = list(clf.predict(fea_test))
        return label_predict 
    
    def XGBoost(self,fea_train,label_train,fea_test):
        clf = GradientBoostingClassifier(n_estimators=200, learning_rate=0.8, max_depth=6, random_state=0)
        clf.fit(fea_train,label_train)
        label_predict = list(clf.predict(fea_test))
        return label_predict         
                                
        
if __name__ == '__main__':

    path_id = '../data/train/id_train.pic'
    path_label = '../data/train/label_binary_train.pic'
    path_fea = '../feature/train/'
    feature = 'PSSM,PCP553,RASA'#,Topo,Zcoord
    N = 10
    win = 3
    rate = 5
    threshold = 0.52
    Undersample = 'True'
    classifier = 'rf'

    test = Validation()
    ACC, Spe, Sen, MCC, AUC = test.main(win, rate, N, classifier, Undersample, threshold)

