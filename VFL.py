from Data import prepare_data
from Model import FNNModel, STGEmbModel, DualSTGModel
from torch import nn
import torch
import pandas as pd
import itertools
from sklearn.metrics import accuracy_score
import seaborn as sns
import math
import numpy as np



def make_binary_models(
    input_dim_list, type='FNN', emb_dim=128, output_dim=1, hidden_dims=[512, 256],
    batch_norm=None, dropout=None, activation='relu',
    flatten=True, sigma=1.0, lam=0.1, top_sigma=1.0, top_lam=0.1):
    models = []
    num_clients = len(input_dim_list)
    if type == 'FNN':
        for input_dim in input_dim_list:
            model = FNNModel(
                input_dim=input_dim, 
                output_dim=emb_dim,
                hidden_dims=hidden_dims,
                batch_norm=batch_norm,
                dropout=dropout,
                activation=activation,
                flatten=flatten
            )
            models.append(model)
        top_model = nn.Sequential(
            nn.Linear(num_clients*emb_dim, 32), nn.ReLU(True), nn.Linear(32, output_dim), nn.Sigmoid()
        )
    elif type == 'STG':
        for input_dim in input_dim_list:
            model = STGEmbModel(
                input_dim=input_dim, 
                output_dim=emb_dim,
                hidden_dims=hidden_dims,
                batch_norm=batch_norm,
                dropout=dropout,
                activation=activation,
                flatten=flatten,
                sigma=sigma,
                lam=lam)
            models.append(model)
        top_model = nn.Sequential(
            nn.Linear(num_clients*emb_dim, 32), nn.ReLU(True), nn.Linear(32, output_dim), nn.Sigmoid()
        )
    elif type == 'DualSTG':
        for input_dim in input_dim_list:
            model = DualSTGModel(
                input_dim=input_dim, 
                output_dim=emb_dim,
                hidden_dims=hidden_dims,
                batch_norm=batch_norm,
                dropout=dropout,
                activation=activation,
                flatten=flatten,
                btm_sigma=sigma,
                btm_lam=lam, 
                top_sigma=top_sigma,
                top_lam=top_lam)
            models.append(model)
        top_model = nn.Sequential(
            nn.Linear(num_clients*emb_dim, 32), nn.ReLU(True), nn.Linear(32, output_dim), nn.Sigmoid()
        )
    return models, top_model

def binary_acc(out, y):
    acc = accuracy_score(y, out>0.5)
    return acc

def visualize_gate(z_list):
    full_z = np.concatenate(z_list)
    int_root = math.isqrt(full_z.size)
    for i in reversed(range(1, int_root)):
        if full_z.size % i == 0:
            w = int(full_z.size / i)
            break
    full_z = full_z.reshape(w, i)
    heatmap = sns.heatmap(full_z, vmin=0, cbar=False)
    return heatmap


def train(
    models, top_model,
    train_loader, test_loader, 
    criterion=nn.BCELoss(),
    optimizer = 'Adam', lr = 1e-3,
    epochs=100, freeze_btm_at=5, freeze_top_at=15,
    verbose=True, save_dir='Checkpoints/model.pt',
    log_dir='Logs/log.csv',
    save_mask_at=20, mask_dir='Mask/'):
    history = []
    column_names = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    best_acc = 0
    for e in range(1, epochs+1):
        if isinstance(models[0], DualSTGModel) or isinstance(models[0], STGEmbModel):
            if e <= freeze_btm_at:
                for model in models:
                    model.freeze_fs()
            else:
                for model in models:
                    model.unfreeze_fs()
            if isinstance(models[0], DualSTGModel):
                if e <= freeze_top_at:
                    for model in models:
                        model.freeze_top()
                else:
                    for model in models:
                        model.unfreeze_top()

        train_acc = 0
        train_loss = 0
        map(lambda m: m.train(), models)
        models = [model.to(device) for model in models]
        top_model.to(device)
        top_model.train()
        parameters = [model.parameters() for model in models]
        parameters.append(top_model.parameters())
        parameters = list(itertools.chain(*parameters))
        if optimizer == "Adam":
            optimizer = torch.optim.Adam(
               parameters, lr=lr)
        elif optimizer == "SGD":
            optimizer = torch.optim.SGD(
                parameters, lr=lr)
        for items, data_y in train_loader:
            optimizer.zero_grad()
            assert len(items) == len(models)
            embs = []
            data_y = data_y.float().to(device).view(-1, 1)
            for i, item in enumerate(items):
                data_x = item 
                data_x = data_x.float().to(device)
                emb = models[i](data_x)
                embs.append(emb)
            embs = torch.cat(embs, dim=1)
            pred = top_model(embs)
            loss = criterion(pred, data_y)
            if isinstance(models[0], DualSTGModel) or isinstance(models[0], STGEmbModel):
                reg_loss_list = [model.get_reg_loss() for model in models]
                reg_loss = torch.mean(torch.stack(reg_loss_list))
                loss += reg_loss
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_acc += binary_acc(pred.detach().cpu().numpy(), data_y.cpu().numpy())
        
        ##################################
        ########## Test ##################
        ##################################
        map(lambda m: m.eval(), models)
        top_model.eval()
        test_acc=0
        with torch.no_grad():
            for items, data_y in test_loader:
                assert len(items) == len(models)
                embs = []
                data_y = data_y.float().to(device).view(-1, 1)
                for i, item in enumerate(items):
                    data_x = item 
                    data_x = data_x.float().to(device)
                    emb = models[i](data_x)
                    embs.append(emb)
                embs = torch.cat(embs, dim=1)
                pred = top_model(embs)
                test_acc += binary_acc(pred.detach().cpu().numpy(), data_y.cpu().numpy())

        ##################################
        ###### Save model ################
        ##################################
        test_acc = test_acc / len(test_loader)
        train_acc = train_acc / len(train_loader)
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(models, save_dir)
        ##################################
        ###### Logging ###################
        ##################################
        if verbose:
            if isinstance(models[0], FNNModel):
                print("Epoch: {}, Train Loss: {:.4f}, Train Acc: {:.4f}, Test Acc: {:.4f}, Best Acc: {:.4f}".format(
                    e, train_loss, train_acc, test_acc, best_acc))
                history.append([train_loss, train_acc, test_acc])
                column_names = ['train_loss', 'train_acc', 'test_acc']
            
            elif isinstance(models[0], STGEmbModel):
                z_list = []
                num_feats = []
                for model in models:
                    z, num_feat = model.get_gates()
                    z_list.append(z)
                    num_feats.append(num_feat)
                num_feats = sum(num_feats)
                print("Epoch: {}, Train Loss: {:.4f}, Train Acc: {:.4f}, Test Acc: {:.4f}, Best Acc: {}, Num Feats: {:.4f}".format(
                    e, train_loss, train_acc, test_acc, best_acc, num_feats))
                history.append([train_loss, train_acc, test_acc, num_feats])
                column_names = ['train_loss', 'train_acc', 'test_acc', 'num_feats']
                if e%save_mask_at == 0:
                    z_heat_map = visualize_gate(z_list)
                    z_heat_map.figure.suptitle('Feature Gates')
                    z_heat_map.figure.savefig(mask_dir + 
                        'STG_gates_{}.png'.format(e))

            elif isinstance(models[0], DualSTGModel):
                top_z_list = []
                btm_z_list = []
                num_feats = []
                num_embs = []
                for model in models:
                    top_z, btm_z, num_emb, num_feat = model.get_gates()
                    top_z_list.append(top_z)
                    btm_z_list.append(btm_z)
                    num_feats.append(num_feat)
                    num_embs.append(num_emb)
                num_feats = sum(num_feats)
                num_emb = sum(num_embs)
                print("Epoch: {}, Train Loss: {:.4f}, Train Acc: {:.4f}, Test Acc: {:.4f}, Best Acc: {}, Num Feats: {:.4f}, Num Emb: {:.4f}".format(
                    e, train_loss, train_acc, test_acc, best_acc, num_feats, num_emb))
                history.append([train_loss, train_acc, test_acc, num_feats, num_emb])
                column_names = ['train_loss', 'train_acc', 'test_acc', 'num_feats', 'num_emb']
                if e%save_mask_at == 0:
                    top_z_heatmap = visualize_gate(top_z_list)
                    btm_z_heatmap = visualize_gate(btm_z_list)
                    top_z_heatmap.figure.suptitle("Embedding Gates")
                    btm_z_heatmap.figure.suptitle("Feature Gates")
                    top_z_heatmap.figure.savefig(
                        mask_dir+"Embedding_heatmap_{}.png".format(e))
                    btm_z_heatmap.figure.savefig(
                        mask_dir+"Feature_heatmap_{}.png".format(e))
                
    history = pd.DataFrame(
        history, columns=column_names)
    history.to_csv(log_dir)
    return history

            
            


if __name__ == '__main__':
    train_loader, test_loader, input_dim_list = prepare_data(
        'BASEHOCK', num_clients=1
    )
    models, top_model = make_binary_models(
        input_dim_list, type='DualSTG')
    print(models)
    train(
        models, top_model, train_loader, test_loader
    )

