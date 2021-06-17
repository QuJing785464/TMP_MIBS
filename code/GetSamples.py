import pickle
import random
import numpy as np
from Bio import SeqIO
import pandas as pd
from sklearn.model_selection import train_test_split
# from Feature_Selection_Filter import Filter_Methonds
# from Feature_selection_Embedded import Embedded_Methods

class GetSamples():
    
    def main(self,path_id,path_label,path_fea,feature,win,rate,Undersample):
        data = Data_Method()
        ID = data.Get_ID(path_id)
        Label = data.Get_Label(path_label)
        Fea,Fea_dim = data.get_feature(Label,path_fea,feature)
        x,y = data.build_fea_space(Label,Fea,Fea_dim,win,rate,Undersample)
        self.output(ID, x, y)
                
        return x,y
    
    def output(self,ID,x,y):        
        pos,neg = 0,0 
        for i in range(0,len(y)):
            if y[i] == 1 or y[i] == '1':
                pos += 1
            else:
                neg += 1
                
        print('---Data---')
        print('Prot Num: '+str(len(ID)))
        print('Site Num: '+str(pos)+'  None_Site Num: '+str(neg))
        print('Feature Dimension: '+str(len(x[0])))
        

class Data_Method():
    def Get_ID(self, path):
        pic_id = open(path, 'rb')
        ID = pickle.load(pic_id)
        pic_id.close()
        no_find = ['6G72_R', '5Z62_F', '6HU9_d', '6J8M_C', '5YI2_A', '4XKT_K', '5KC1_F', '2WCD_K',
                   '6ADQ_K', '2YBB_7', '6YJ4_R', '6IGZ_3', '6TJV_P', '2IWV_A', '5ZLG_A', '5EC5_G','1KQG_A']
        ID_list = []
        for key in ID:
            if key in no_find:
                continue
            else:
                ID_list.append(ID)
        return ID_list

    def Get_Label(self, path):
        pic_label = open(path, 'rb')
        label = pickle.load(pic_label)
        pic_label.close()
        no_find = ['6G72_R', '5Z62_F', '6HU9_d', '6J8M_C', '5YI2_A', '4XKT_K', '5KC1_F', '2WCD_K',
                   '6ADQ_K', '2YBB_7', '6YJ4_R', '6IGZ_3', '6TJV_P', '2IWV_A', '5ZLG_A', '5EC5_G','1KQG_A']
        label_dict = {}
        for key in label:
            if key in no_find:
                continue
            else:
                label_dict[key] = label[key]
        return label_dict

    def get_feature(self,Label,path_fea,feature):
        Fea_original = {}
        Fea = {}
        fea_list = feature.split(',')
        for each in fea_list:
            path = path_fea+each+'.pic'
            pic = open(path,'rb')
            fea = pickle.load(pic)
            pic.close()
            Fea_original[each] = fea

        for ID in Label.keys():
            fea_prot = []
            for i in range(0,len(Label[ID])):
                fea_aa = []
                for each in Fea_original:
                    fea_aa.extend(Fea_original[each][ID][i])
                fea_prot.append(fea_aa)
            Fea[ID] = fea_prot
        Fea_dim = len(fea_aa)
        return Fea,Fea_dim

    def pre(self,Label,Fea,Fea_dim,win,rate,Undersample):
        x_pos, x_neg = [], []
        X, Y= [], []
        for ID in Label:
            for i in range(0,len(Label[ID])):
                x = Fea[ID][i]
                if Label[ID][i] == 1 or Label[ID][i] == '1':
                    x_pos.append(x)
                else:
                    x_neg.append(x)
        neg_num = rate * len(x_pos)
        if Undersample == 'True':
            x_neg = random.sample(x_neg, neg_num)
        for i in range(0, len(x_pos)):
            X.append(x_pos[i])
            Y.append([1])
        for i in range(0, len(x_neg)):
            X.append(x_neg[i])
            Y.append([0])
        X = np.array(X, dtype='float_')
        Y = np.array(Y, dtype='int_')
        X_Y = np.concatenate((X, Y), axis=1)
        selected_feat = self.Feature_selection(X_Y)
        return selected_feat

    def update_fea(self,selected_feature,Fea,Label):
        new_fea = {}
        for ID in Fea:
            fea_prot = []
            for residue in Fea[ID]:
                fea_aa = []
                for i in selected_feature:
                    fea_aa.append(float(residue[int(i)-1]))
                fea_prot.append(fea_aa)
            if len(fea_prot) != len(Label[ID]):
                print(len(fea_prot),len(Label[ID]))
            new_fea[ID] = fea_prot
        Fea_dim = len(fea_aa)
        return new_fea, Fea_dim

    def build_fea_space(self,Label,Fea,Fea_dim,win,rate,Undersample):  
        x_pos,x_neg = [],[]      
        X,Y,y = [],[],[]
        zero = []
        for i in range (0,Fea_dim):
            zero.append(0)        
        for ID in Label:
            for center in range(0,len(Label[ID])):
                x = []
                start = center - win
                end = center + win+1
                seq_length = len(Fea[ID])               
                if start >= 0 and end <= seq_length:
                    for aa in range(start,end):
                        x.extend(Fea[ID][aa])      
                elif start < 0:
                    for i in range(start,0):
                        x.extend(zero)
                    for aa in range(0,end):
                        x.extend(Fea[ID][aa])
                else:
                    for aa in range(start,seq_length):
                        x.extend(Fea[ID][aa])
                    for i in range(seq_length,end):
                        x.extend(zero)
                if Label[ID][center] == 1 or Label[ID][center] == '1':
                    x_pos.append(x)
                else:
                    x_neg.append(x)
        neg_num = rate*len(x_pos)
        if Undersample == 'True':
            x_neg = random.sample(x_neg,neg_num)
        for i in range(0,len(x_pos)):
            X.append(x_pos[i])
            y.append(1)
        for i in range(0,len(x_neg)):
            X.append(x_neg[i])
            y.append(0)
        X = np.array(X,dtype='float_')
        y = np.array(y, dtype = 'int')
        # dict = {}
        # dict['data'] = X
        # dict['target'] = Y.reshape(-1, 1)
        # dict['feature_name'] = [i for i in range(1, 554)]
        # savefilepath = "/home/qjnenu/PDBTM/data/Fea_train_data/binary_rate16_data.pic"
        # dictfile = open(savefilepath, 'wb')
        # pickle.dump(X_Y, dictfile)
        # dictfile.close()
        # print('OK')
        return X,y

    def Feature_selection(self,data):
        df = pd.DataFrame(data)
        df.rename(columns={df.columns[-1]: 'target'}, inplace=True)
        X = df.drop(labels=['target'], axis=1)
        y = df.target
        y = np.array(y)
        # split data set
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        print(y_test,y_test.shape)

        filter = Filter_Methonds()
        X_train_basic_filter,X_test_basic_filter = filter.Filter_Basics(X_train, X_test)
        embedded = Embedded_Methods()
        selected_feat = embedded.Gradient_Boosted_Trees_Importance(X_train_basic_filter,X_test_basic_filter,y_train)

        return selected_feat

            
'''         
if __name__ == '__main__':
    
    path_id = '/home/luchang/MPPI/Data/Train_ID.pic'
    path_label = '/home/luchang/MPPI/Data/Train_label.pic'
    path_fea = '/home/luchang/MPPI/Fea/Train/'
    feature = 'PSSM,Topo'
    win = 3
    rate = 1 
    Undersample = 'True'
    
    sample = GetSamples()
    train_x,train_y = sample.main(path_id,path_label,path_fea,feature,win,rate,Undersample)
'''  