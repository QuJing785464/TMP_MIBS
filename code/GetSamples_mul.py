import pickle
import random
import numpy as np

class GetSamples():
    
    def main(self, path_id, mul_label, binary_label, path_fea, feature, win, rate, Undersample):
        data = Data_Method()
        ID = data.Get_ID(path_id)
        bin_Label = data.Get_Label(binary_label)
        mul_Label= data.Get_Label(mul_label)
        Fea, Fea_dim = data.get_feature(mul_Label, path_fea, feature)
        x, y = data.build_fea_space(bin_Label, mul_Label, Fea, Fea_dim, win, rate, Undersample)
        self.output(ID, x, y)
                
        return x, y
    
    def output(self, ID, x, y):
        pos, neg = 0, 0
        for i in range(0,len(y)):
            if y[i] == 1 or y[i] == '1':
                pos += 1
            else:
                neg += 1

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
        
    def get_feature(self, Label, path_fea, feature):
        Fea_original = {}
        Fea = {}
        fea_list = feature.split(',')
        for each in fea_list:
            path = path_fea+each+'.pic'
            pic = open(path, 'rb')
            fea = pickle.load(pic)
            pic.close()
            Fea_original[each] = fea
        for ID in Label.keys():
            fea_prot = []
            for i in range(0, len(Label[ID])):
                fea_aa, fea_bb = [], []
                for each in Fea_original:
                    fea_aa.extend(Fea_original[each][ID][i])
                fea_prot.append(fea_aa)
            Fea[ID] = fea_prot
        Fea_dim = len(fea_aa)
        return Fea, Fea_dim

    def build_fea_space(self, bin_Label, mul_Lable, Fea, Fea_dim, win, rate, Undersample):
        x_pos, x_neg = [], []
        X, Y = [], []
        zero = []
        for i in range(0, Fea_dim):
            zero.append(0)        
        for ID in bin_Label:
            for center in range(0, len(bin_Label[ID])):

                x = []
                start = center - win
                end = center + win+1
                seq_length = len(Fea[ID])
                if start >= 0 and end <= seq_length:
                    for aa in range(start, end):
                        x.extend(Fea[ID][aa])
                elif start < 0:
                    for i in range(start, 0):
                        x.extend(zero)
                    for aa in range(0, end):
                        x.extend(Fea[ID][aa])
                else:
                    for aa in range(start, seq_length):
                        x.extend(Fea[ID][aa])
                    for i in range(seq_length, end):
                        x.extend(zero)
                if bin_Label[ID][center] == 1 or bin_Label[ID][center] == '1':
                    if mul_Lable[ID][center] == 1 or mul_Lable[ID][center] == '1':
                        x_pos.append(x)
                    else:
                        x_neg.append(x)
                else:
                    continue

        print(len(x_pos), len(x_neg))
        neg_num = rate*len(x_pos)
        if Undersample == 'True':
            x_neg = random.sample(x_neg, neg_num)
        for i in range(0, len(x_pos)):
            X.append(x_pos[i])
            Y.append(1)
        for i in range(0, len(x_neg)):
            X.append(x_neg[i])
            Y.append(0)
        X = np.array(X, dtype='float_')
        Y = np.array(Y, dtype='int_')
        return X, Y


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