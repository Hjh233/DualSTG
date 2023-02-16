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
file_name = 'RELATHE.mat'

mat = loadmat(os.path.join(DIR, file_name))
X = mat["X"]
y = mat["Y"]
if issparse(X):
    X = X.todense()
y = y.flatten()
print(file_name, X.shape, y.shape)
y[np.where(y == 1)] = 0
y[np.where(y == 2)] = 1
scaler = MinMaxScaler()
X = scaler.fit_transform(X)
dataset = VFLDataset(data_source=(X, y), 
                    num_clients=10,
                    gini_portion=None,
                    insert_noise=False,
                    test_size=0.2)
train_loader = DataLoader(dataset.train(), batch_size=256, shuffle=False)
val_loader = DataLoader(dataset.valid(), batch_size=1000, shuffle=False)
test_loader = DataLoader(dataset.test(), batch_size=1000, shuffle=False)
input_dim_list = dataset.get_input_dim_list()
output_dim = np.unique(y).size
criterion = torch.nn.CrossEntropyLoss()

relathe_zeta = np.array([0.1, 0.3, 0.5, 1.0, 2.0, 2.5, 3.0, 4.0, 5.0])
btm_z_overlap = []

if __name__ == "__main__":
    # STG Training Phase
    # mus = None

    # models, top_model = VFL.make_binary_models(
    #     input_dim_list=input_dim_list,
    #     type="STG",
    #     emb_dim=4,
    #     output_dim=output_dim,
    #     hidden_dims=[8, 8],
    #     activation="relu",
    #     mus=mus, lam=0.1,
    #     zeta=0)

    # original_gate_history, _ = VFL.train(
    #     models,
    #     top_model,
    #     train_loader,
    #     val_loader,
    #     test_loader,
    #     epochs=60,
    #     optimizer='Adam',
    #     criterion=criterion,
    #     verbose=True,
    #     models_save_dir='Checkpoints/relathe_stg_models.pt',
    #     top_model_save_dir='Checkpoints/relathe_stg_top_model.pt',
    #     save_mask_at=100000, 
    #     freeze_top_till=0)

    # # print(original_gate_history)
    # print(original_gate_history.tail())

    # # original_gate_history.to_csv('LDPLog/relathe_original_gate.csv')


    # STG Inference Phase
    # mus = None

    # models, top_model = VFL.make_binary_models(
    #     input_dim_list=input_dim_list,
    #     type="STG",
    #     emb_dim=4,
    #     output_dim=output_dim,
    #     hidden_dims=[8, 8],
    #     activation="relu",
    #     mus=mus, lam=0.1,
    #     zeta=0)
    
    # models_path = 'Checkpoints/relathe_stg_models.pt'
    # top_model_path = 'Checkpoints/relathe_stg_top_model.pt'

    # models_checkpoint = torch.load(models_path)

    # models[0].load_state_dict(models_checkpoint['model_0_state_dict'])
    # models[1].load_state_dict(models_checkpoint['model_1_state_dict'])
    # models[2].load_state_dict(models_checkpoint['model_2_state_dict'])

    # # models.load_state_dict(torch.load(models_path))
    # top_model.load_state_dict(torch.load(top_model_path))

    # VFL.inference(models, top_model, test_loader)


    # DualSTG Training Phase

    # gini_labels = dataset.gini_filter(0.5)
    # feat_idx_list = dataset.get_feature_index_list()

    # mus = VFL.initialize_mu(gini_labels, feat_idx_list)
    # models, top_model = VFL.make_binary_models(
    #     input_dim_list=input_dim_list,
    #     type="DualSTG",
    #     emb_dim=4,
    #     output_dim=output_dim,
    #     hidden_dims=[8, 8],
    #     activation="relu",
    #     mus=mus, top_lam=0.1, lam=0.1,
    #     zeta=0)

    # dual_stg_gini_history, _ = VFL.train(
    #     models,
    #     top_model,
    #     train_loader,
    #     val_loader,
    #     test_loader,
    #     epochs=60,
    #     optimizer='Adam',
    #     criterion=criterion,
    #     verbose=True,
    #     models_save_dir='Checkpoints/relathe_dualstg_models.pt',
    #     top_model_save_dir='Checkpoints/relathe_dualstg_top_model.pt',        
    #     save_mask_at=100000, 
    #     freeze_top_till=0)

    # # print(dual_stg_gini_history)
    # print(dual_stg_gini_history.tail())

    # dual_stg_gini_history.to_csv('LDPLog/relathe_ldp_{}.csv'.format(zeta))

    # DualSTG Inference Phase
    gini_labels = dataset.gini_filter(0.5)
    feat_idx_list = dataset.get_feature_index_list()

    mus = VFL.initialize_mu(gini_labels, feat_idx_list)
    models, top_model = VFL.make_binary_models(
        input_dim_list=input_dim_list,
        type="DualSTG",
        emb_dim=4,
        output_dim=output_dim,
        hidden_dims=[8, 8],
        activation="relu",
        mus=mus, top_lam=0.1, lam=0.1,
        zeta=0)
    
    models_path = 'Checkpoints/relathe_dualstg_models.pt'
    top_model_path = 'Checkpoints/relathe_dualstg_top_model.pt'


    models_checkpoint = torch.load(models_path)

    models[0].load_state_dict(models_checkpoint['model_0_state_dict'])
    models[1].load_state_dict(models_checkpoint['model_1_state_dict'])
    models[2].load_state_dict(models_checkpoint['model_2_state_dict'])

    # models.load_state_dict(torch.load(models_path))
    top_model.load_state_dict(torch.load(top_model_path))

    VFL.inference(models, top_model, test_loader)




    