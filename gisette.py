import argparse
import os

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.sparse import issparse
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import DataLoader

from Data import VFLDataset
import VFL

DIR = "Data"

file_name = 'gisette.mat'
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
                    test_size=0.3)
train_loader = DataLoader(dataset.train(), batch_size=512, shuffle=True)
val_loader = DataLoader(dataset.valid(), batch_size=1000, shuffle=True)
test_loader = DataLoader(dataset.test(), batch_size=1000, shuffle=True)
input_dim_list = dataset.get_input_dim_list()
output_dim = np.unique(y).size
criterion = torch.nn.CrossEntropyLoss()

gisette_zeta = np.array([1.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0])
btm_z_overlap = []


from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve

def _plot_roc(label, y_prob, file_dir):
    fpr, tpr, thresholds_roc = roc_curve(label, y_prob)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(fpr, tpr, ls="-", linewidth=2.0)
    ax.grid(True, linestyle="-.")
    ax.set_xlabel("false positive rate", labelpad=5, loc="center")
    ax.set_ylabel("true positive rate", labelpad=5, loc="center")
    ax.set_title("ROC Curve")

    # plt.show()
    plt.savefig(f"{file_dir}/ROC_Curve_gisette.png")
    plt.close()


if __name__ == "__main__":
    for i in range(1):
        feat_idx_list = dataset.get_feature_index_list()

        gini_labels = dataset.gini_filter(0.5, not_init_features=[])

        mus = VFL.initialize_mu(gini_labels, feat_idx_list)
        models, top_model = VFL.make_binary_models(
            input_dim_list=input_dim_list,
            type="DualSTG",
            emb_dim=8,
            output_dim=output_dim,
            hidden_dims=[32, 16],
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
            epochs=80,
            optimizer='Adam',
            criterion=criterion,
            verbose=True,
            models_save_dir='Response/Review3/Checkpoints/gisette_dualstg_models.pt',
            top_model_save_dir='Response/Review3/Checkpoints/gisette_dualstg_top_model.pt',        
            save_mask_at=100000, 
            freeze_top_till=0)
        
        end = time.time()
        print(end - begin)

        # print(dual_stg_gini_history)
        print(dual_stg_gini_history.tail())

        # dual_stg_gini_history.to_csv('Response/Review3/gisette_{}.csv'.format(i))

    # inference to get predict labels
    gini_labels = dataset.gini_filter(0.5)
    feat_idx_list = dataset.get_feature_index_list()

    mus = VFL.initialize_mu(gini_labels, feat_idx_list)
    models, top_model = VFL.make_binary_models(
            input_dim_list=input_dim_list,
            type="DualSTG",
            emb_dim=8,
            output_dim=output_dim,
            hidden_dims=[32, 16],
            activation="relu",
            mus=mus, top_lam=0.1, lam=0.1,
            zeta=0)

    models_path = 'Response/Review3/Checkpoints/gisette_dualstg_models.pt'
    top_model_path = 'Response/Review3/Checkpoints/gisette_dualstg_top_model.pt'

    models_checkpoint = torch.load(models_path)

    models[0].load_state_dict(models_checkpoint['model_0_state_dict'])
    models[1].load_state_dict(models_checkpoint['model_1_state_dict'])
    models[2].load_state_dict(models_checkpoint['model_2_state_dict'])

    top_model.load_state_dict(torch.load(top_model_path))

    labels, preds = VFL.inference(models, top_model, test_loader)

    results = pd.DataFrame(
            {
            "labels":labels.tolist(),
            "preds": preds.tolist(), 
            }
            )
    results.head()

    results.to_csv('Response/Review3/gisette.csv')

    _plot_roc(labels, preds, 'Response/Review3/')

