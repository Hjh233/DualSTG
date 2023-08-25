import argparse
import os

import numpy as np
from scipy.io import loadmat
from scipy.sparse import issparse
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import DataLoader

from Data import VFLDataset
import VFL

DIR = "Data"

file_name = 'isolet.mat'
mat = loadmat(os.path.join(DIR, file_name))
X = mat["X"]
y = mat["Y"]
if issparse(X):
    X = X.todense()
y = y.flatten()
print(file_name, X.shape, y.shape)
y = y-1
dataset = VFLDataset(data_source=(X, y), 
                    num_clients=9,
                    gini_portion=None,
                    insert_noise=False,
                    test_size=0.3)
train_loader = DataLoader(dataset.train(), batch_size=512, shuffle=False)
val_loader = DataLoader(dataset.valid(), batch_size=1000, shuffle=False)
test_loader = DataLoader(dataset.test(), batch_size=1000, shuffle=False)
input_dim_list = dataset.get_input_dim_list()
output_dim = np.unique(y).size
criterion = torch.nn.CrossEntropyLoss()

isolet_zeta = np.array([0.1, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 15.0])
btm_z_overlap = []

if __name__ == "__main__":
    for i in range(5):

        feat_idx_list = dataset.get_feature_index_list()

        lasso_score = torch.load('Response/Review1/Lasso/isolet_{}.pt'.format(i))

        lasso_labels = dataset.lasso_filter(lasso_score, 0.5)

        mus = VFL.initialize_mu(lasso_labels, feat_idx_list)
        models, top_model = VFL.make_binary_models(
            input_dim_list=input_dim_list,
            type="DualSTG",
            emb_dim=16,
            output_dim=output_dim,
            hidden_dims=[64, 32],
            activation="relu",
            mus=mus, top_lam=0.1, lam=0.1,
            zeta=0)

        import time

        begin = time.time()

        dual_stg_gini_history, _ = VFL.train(
            models,
            top_model,
            train_loader,
            val_loader,
            test_loader,
            epochs=300,
            optimizer='Adam',
            criterion=criterion,
            verbose=True,
            models_save_dir='Checkpoints/isolet_dualstg_models.pt',
            top_model_save_dir='Checkpoints/isolet_dualstg_top_model.pt',        
            save_mask_at=100000, 
            freeze_top_till=0)
        
        end = time.time()
        print(end - begin)

        # # print(dual_stg_gini_history)
        # print(dual_stg_gini_history.tail())

        dual_stg_gini_history.to_csv('Response/Review1/Lasso/isolet_{}.csv'.format(i))
