from scipy.io import loadmat
from scipy.sparse import issparse
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from Data import VFLDataset
from torch.utils.data import DataLoader
import VFL
import torch
import os
DIR = "Data"

file_name = 'arcene.mat'
mat = loadmat(os.path.join(DIR, file_name))
X = mat["X"]
y = mat["Y"]
if issparse(X):
    X = X.todense()
y = y.flatten()
print(file_name, X.shape, y.shape)
y[np.where(y == -1)] = 0
scaler = MinMaxScaler()
X = scaler.fit_transform(X)
dataset = VFLDataset(data_source=(X, y), 
                    num_clients=2,
                    gini_portion=None,
                    insert_noise=False,
                    test_size=0.5)
train_loader = DataLoader(dataset.train(), batch_size=128, shuffle=True)
val_loader = DataLoader(dataset.valid(), batch_size=1000, shuffle=True)
test_loader = DataLoader(dataset.test(), batch_size=1000, shuffle=True)
input_dim_list = dataset.get_input_dim_list()
output_dim = np.unique(y).size
criterion = torch.nn.CrossEntropyLoss()


gini_labels = dataset.gini_filter(0.5)
feat_idx_list = dataset.get_feature_index_list()

mus = VFL.initialize_mu(gini_labels, feat_idx_list)
models, top_model = VFL.make_binary_models(
    input_dim_list=input_dim_list,
    type="DualSTG",
    emb_dim=8,
    output_dim=output_dim,
    hidden_dims=[8, 8],
    activation="relu",
    mus=mus, top_lam=0.8, lam=0.2)
dual_stg_gini_history = VFL.train(
    models,
    top_model,
    train_loader,
    val_loader,
    test_loader,
    epochs=40,
    optimizer='Adam',
    criterion=criterion,
    verbose=True,
    save_mask_at=100000, freeze_top_till=0)


print(dual_stg_gini_history.tail())


dual_stg_gini_history.to_csv('DPLog/arcene_ldp_0.csv')






