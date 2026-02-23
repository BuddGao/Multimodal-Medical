import pandas as pd
import numpy as np
import pickle
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='qin_kmeans')
    parser.add_argument('--K', type=int, default=4)
    return parser.parse_known_args()
args, unknown = parse_args()
K = args.K
data = pd.read_excel('/content/drive/MyDrive/MML/medical_dataset/Duke_GIRI/Clinical_and_Other_Features.xlsx', header=1)
data = data.drop([0])
data.columns = data.columns.str.rstrip()

data = data.reset_index(drop=True)


data['Sum grade'] = data['Sum grade'].astype('Int64')
data['Nottingham grade']=data['Nottingham grade'].fillna(data['Sum grade'])
group1 = [
    "Menopause (at diagnosis)", "ER", "PR", "Surgery", 
    "Adjuvant Radiation Therapy", "Adjuvant Endocrine Therapy Medications", 
    "Pec/Chest Involvement"
]

group2 = [
    "HER2", "Multicentric/Multifocal", "Lymphadenopathy or Suspicious Nodes", 
    "Definitive Surgery Type", "Neoadjuvant Chemotherapy", "Adjuvant Chemotherapy", 
    "Neoadjuvant Anti-Her2 Neu Therapy", "Adjuvant Anti-Her2 Neu Therapy"
]

group3 = [
    "Metastatic at Presentation (Outside of Lymph Nodes)", 
    "Contralateral Breast Involvement", "Staging(Metastasis)#(Mx -replaced by -1)[M]", 
    "Skin/Nipple Invovlement", "Neoadjuvant Radiation Therapy", "Recurrence event(s)", 
    "Known Ovarian Status", 
    "Therapeutic or Prophylactic Oophorectomy as part of Endocrine Therapy", 
    "Neoadjuvant Endocrine Therapy Medications"
]

group4 = [
    "Staging(Nodes)#(Nx replaced by -1)[N]"
]

targetv = ['Nottingham grade' , 'Patient ID']
all_columns = group1 + group2 + group3 + group4 +targetv


data = data[all_columns]

# na_drop_list = []
# for col in data.keys():
#     if sum(data[col].isna())/len(data) > 0.1:
        
#         na_drop_list.append(col)
   
# #na_drop_list  = ['Staging(Tumor Size)# [T]','Mol Subtype','Number of Ovaries In Situ \n','Race and Ethnicity'] + na_drop_list +['Days to last local recurrence free assessment (from the date of diagnosis) ', 'Days to last distant recurrence free assemssment(from the date of diagnosis) ','Days to Surgery (from the date of diagnosis)','Days to MRI (From the Date of Diagnosis)']
# data.drop(columns = na_drop_list, inplace=True)
for col in data.keys():
    data[col].fillna(data[col].mode()[0])

#numerical = ['Date of Birth (Days)']
data_non_dmgi = data.copy()

#for col in numerical:
#    data_non_dmgi[col] = pd.qcut(data_non_dmgi[col],5, labels=False)
sub = data_non_dmgi['Patient ID']
cdr = data_non_dmgi['Nottingham grade']
data_non_dmgi_drop = data_non_dmgi.drop(columns=['Patient ID','Nottingham grade'])

#data_non_dmgi_drop = data_non_dmgi.drop(columns=['Patient ID','Tumor Grade','Unnamed: 32','Unnamed: 32','Summation','Sum grade','Nottingham grade'])
data_non_dmgi_drop_dum = pd.get_dummies(data_non_dmgi_drop, columns=data_non_dmgi_drop.keys())
feature_dict = {}
label_dict = {}
i = 0
# use_clinical_dummy_no = use_clinical_dummy.drop(columns=['cdr','age_','apoe']).fillna(0)
for s in sub:

    feature_dict[s] =data_non_dmgi_drop_dum.iloc[i].to_numpy()
    label_dict[s] = cdr[i]

    i+=1
#data_dmgi = data.drop(columns=['Date of Birth (Days)'])
data_dmgi = data.copy()
path_ = '/content/drive/MyDrive/MML/medical_dataset/Duke_GIRI/extracted_feature/'
features = np.loadtxt( path_ + 'train_feature.csv',delimiter=',',dtype=np.float32)
# valid_feature = np.loadtxt('./' + path_ + 'valid_feature.csv',delimiter=',',dtype=np.float32)
ids = pd.read_csv( path_ +'train_id.csv', header=None)
# valid_id = pd.read_csv('./' + path_ +'valid_id.csv', header=None)
for col in data_dmgi:
    data_dmgi[col].fillna(data_dmgi[col].mode()[0],inplace=True)
idx_list = []
for col in data_dmgi.keys():
    idx = data_dmgi[data_dmgi[col] =='NP'].index
    idx_list.append(idx)
    if len(idx)>0:
        data_dmgi.loc[idx, col] = -1



available_cols = set(data_dmgi.columns)
type_list = [
    [col for col in group1 if col in available_cols],
    [col for col in group2 if col in available_cols],
    [col for col in group3 if col in available_cols],
    [col for col in group4 if col in available_cols]
]

K = 4 # Force K to 4 since we have 4 explicit groups
# ==========================================

processed_ids = [f"{int(idss):03}" for idss in ids[0].tolist()]
id_list = []
for processed_id in processed_ids:
    a = 'Breast_MRI_' + processed_id
    id_list.append(a)
    
from sklearn.preprocessing import minmax_scale
import sklearn
import sklearn.metrics.pairwise

k = 0
save_list= []
thres = 0.8
for types in type_list:
    before_adj_ = []
    print('type' + str(k))
    print("Variables in this group:", types)
    print("********")
    use_clinical_dummy_multi = pd.get_dummies(data_dmgi[['Nottingham grade']+types], columns=types)
    ll = use_clinical_dummy_multi.drop(columns=['Nottingham grade'])
    ll = ll.fillna(0)
    for id_ in id_list:
        lp = data_dmgi[data_dmgi['Patient ID'] ==id_].index.item()
        p = ll.loc[lp].tolist()
        before_adj_.append(p)
    p_ = np.array(before_adj_)
    
    # Check if group had any valid columns to prevent minmax_scale crashing
    if p_.shape[1] > 0:
        p_ = minmax_scale(p_, axis=0, copy =True)
        cos_ = sklearn.metrics.pairwise.cosine_similarity(p_,p_)
    else:
        # Fallback if a group ended up entirely empty after data cleaning
        cos_ = np.zeros((len(id_list), len(id_list)))
        
    adj_ = np.zeros(cos_.shape)
    for i in range(cos_.shape[0]):
        for j in range(cos_.shape[0]):
            if cos_[i][j] > thres:
                adj_[i][j] = 1
            else:
                adj_[i][j] = 0
    print('dense')
    print(sum(sum(adj_))/(adj_.shape[0]*adj_.shape[0]))
    print("********")
    save_list.append(adj_)
    k+=1

feat = []
lab = []

for i in range(len(id_list)):
    a_ = id_list[i]
    img_feat = features[i]
    l_ = data_dmgi.drop(columns=['Patient ID', 'Nottingham grade'])
    l_ = l_.fillna(0)

    l_dum = pd.get_dummies(l_, columns=l_.keys())
    l = data_dmgi[data_dmgi['Patient ID'] ==a_].index.item()
    p = l_dum.loc[l].tolist()
    p_num = np.array(p)
    pp = np.concatenate((img_feat,p_num))
    feat.append(pp)
    label = data_dmgi.loc[l,'Nottingham grade'].item()
    lab.append(label)

concate_feature = np.array(feat)
indexes = [i for i in range(len(lab))]

train_ = len(indexes[:int(len(indexes)*0.6)])
valid_ = len(indexes[int(len(indexes)*0.6):int(len(indexes)*0.7)])
test_ = len(indexes[int(len(indexes)*0.7):])

import random
random.shuffle(indexes)
train_index = indexes[:train_]
valid_index = indexes[train_:train_+valid_]
test_index = indexes[train_+valid_:]

y = np.zeros((len(lab),3))
for i in range(len(lab)):
    if lab[i] ==1:
        y[i,0]=1
    elif lab[i]==2:
        y[i,1]=1
    else:
        y[i,2] =1

train_index = np.array(train_index)
valid_index = np.array(valid_index)
test_index = np.array(test_index)

# Since we forced K=4 above, we only need the K==4 block
adj_0 = save_list[0]
adj_1 = save_list[1]
adj_2 = save_list[2]
adj_3 = save_list[3]
oasis3_multi = {
    'label':y,
    'type0':adj_0,
    'type1':adj_1,
    'type2':adj_2,
    'type3':adj_3,
    'train_idx':train_index,
    'val_idx':valid_index,
    'test_idx':test_index,
    'feature':concate_feature
}
# k_means_list = data_dmgi.drop(columns = ['Patient ID','Tumor Grade','Unnamed: 32','Unnamed: 32','Summation','Sum grade']).select_dtypes(include=[np.number]).keys()
# import random
# transpose_num = data_dmgi[k_means_list].T
# from sklearn.cluster import KMeans
# kmeans = KMeans(n_clusters=K)
# kmeans.fit(transpose_num)

# type_list = []
# for k in range(K):
#     type_list.append([])
# for i in range(len(k_means_list)):
#     type_list[kmeans.labels_[i]].append(k_means_list[i])

# processed_ids = [f"{int(idss):03}" for idss in ids[0].tolist()]
# id_list = []
# for processed_id in processed_ids:
#     a = 'Breast_MRI_' + processed_id
#     id_list.append(a)
# from sklearn.preprocessing import minmax_scale
# import sklearn
# from sklearn.cluster import KMeans
# import sklearn.metrics.pairwise

# k = 0
# save_list= []
# threses = [0.8, 0.8, 0.8, 0.8]
# thres = 0.8
# for types in type_list:
#     before_adj_ = []
#     print('type' + str(k))
#     print("********")
#     use_clinical_dummy_multi = pd.get_dummies(data_dmgi[['Tumor Grade']+types], columns=types)
#     ll = use_clinical_dummy_multi.drop(columns=['Tumor Grade'])
#     ll = ll.fillna(0)
#     for id_ in id_list:
#         lp = data_dmgi[data_dmgi['Patient ID'] ==id_].index.item()
#         p = ll.loc[lp].tolist()
#         before_adj_.append(p)
#     p_ = np.array(before_adj_)
#     p_ = minmax_scale(p_, axis=0, copy =True)

#     cos_ = sklearn.metrics.pairwise.cosine_similarity(p_,p_)
#     adj_ = np.zeros(cos_.shape)
#     thres = 0.8
#     for i in range(cos_.shape[0]):
#         for j in range(cos_.shape[0]):
#             if cos_[i][j] > thres:
#                 adj_[i][j] = 1
#             else:
#                 adj_[i][j] = 0
#     print('dense')
#     print(sum(sum(adj_))/(adj_.shape[0]*adj_.shape[0]))
#     print("********")
#     save_list.append(adj_)
#     k+=1

# feat = []
# lab = []






# for i in range(len(id_list)):
#     a_ = id_list[i]
#     img_feat = features[i]
#     l_ = data_dmgi.drop(columns=['Patient ID', 'Tumor Grade'])
#     l_ = l_.fillna(0)

#     l_dum = pd.get_dummies(l_, columns=l_.keys())
#     l = data_dmgi[data_dmgi['Patient ID'] ==a_].index.item()
#     p = l_dum.loc[l].tolist()
#     p_num = np.array(p)
#     pp = np.concatenate((img_feat,p_num))
#     feat.append(pp)
#     label = data_dmgi.loc[l,'Tumor Grade'].item()
#     lab.append(label)
#     i+=1
# concate_feature = np.array(feat)
# indexes = [i for i in range(len(lab))]
# # indexes=train_index.tolist() + valid_index.tolist() + test_index.tolist()
# train_ = len(indexes[:int(len(indexes)*0.6)])
# valid_ = len(indexes[int(len(indexes)*0.6):int(len(indexes)*0.7)])
# test_ = len(indexes[int(len(indexes)*0.7):])

# import random
# random.shuffle(indexes)
# train_index = indexes[:train_]
# valid_index = indexes[train_:train_+valid_]
# test_index = indexes[train_+valid_:]
# # train_index = [i for i in range(train_)]
# # valid_index = [i+max(train_index)+1 for i in range(valid_)]
# # test_index = [i+max(valid_index)+1 for i in range(test_)]

# y = np.zeros((len(lab),3))
# for i in range(len(lab)):
#     if lab[i] ==1:
#         y[i,0]=1
#     elif lab[i]==2:
#         y[i,1]=1
#     else:
#         y[i,2] =1

# train_index = np.array(train_index)
# valid_index = np.array(valid_index)
# test_index = np.array(test_index)
# if K ==6:
#     adj_0 = save_list[0]
#     adj_1 = save_list[1]
#     adj_2 = save_list[2]
#     adj_3 = save_list[3]
#     adj_4 = save_list[4]
#     adj_5 = save_list[5]
#     oasis3_multi = {'label':y,'type0':adj_0,'type1':adj_1,'type2':adj_2,'type3':adj_3,'type4':adj_4,'type5':adj_5,'train_idx':train_index,'val_idx':valid_index,'test_idx':test_index,'feature':concate_feature}
# elif K ==5:
#     adj_0 = save_list[0]
#     adj_1 = save_list[1]
#     adj_2 = save_list[2]
#     adj_3 = save_list[3]
#     adj_4 = save_list[4]
#     oasis3_multi = {'label':y,'type0':adj_0,'type1':adj_1,'type2':adj_2,'type3':adj_3,'type4':adj_4,'train_idx':train_index,'val_idx':valid_index,'test_idx':test_index,'feature':concate_feature}
# elif K ==4:
#     adj_0 = save_list[0]
#     adj_1 = save_list[1]
#     adj_2 = save_list[2]
#     adj_3 = save_list[3]
#     oasis3_multi = {'label':y,'type0':adj_0,'type1':adj_1,'type2':adj_2,'type3':adj_3,'train_idx':train_index,'val_idx':valid_index,'test_idx':test_index,'feature':concate_feature}
# elif K ==3:
#     adj_0 = save_list[0]
#     adj_1 = save_list[1]
#     adj_2 = save_list[2]
#     oasis3_multi = {'label':y,'type0':adj_0,'type1':adj_1,'type2':adj_2,'train_idx':train_index,'val_idx':valid_index,'test_idx':test_index,'feature':concate_feature}
# elif K ==2:
#     adj_0 = save_list[0]
#     adj_1 = save_list[1]
#     oasis3_multi = {'label':y,'type0':adj_0,'type1':adj_1,'train_idx':train_index,'val_idx':valid_index,'test_idx':test_index,'feature':concate_feature}

with open('/content/drive/MyDrive/MML/medical_dataset/Duke_GIRI/Duke_GIRI.pkl', 'wb') as f:
    pickle.dump(oasis3_multi, f, pickle.HIGHEST_PROTOCOL)
    