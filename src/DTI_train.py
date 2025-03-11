from model import MDA
import pandas as pd
import numpy as np
from torch import optim, nn
import torch as t
import logging
from sklearn.metrics import roc_auc_score, accuracy_score, average_precision_score, f1_score, recall_score, precision_score
from sklearn.model_selection import KFold
import argparse
import os
import warnings

warnings.filterwarnings("ignore")
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  
os.environ['CUDA_VISIBLE_DEVICES'] = "1"  
# 设置日志
logging.basicConfig(filename='/root/autodl-tmp/HGTMDA-main/training.log',
                    level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')
device = t.device('cuda:0' if t.cuda.is_available() else "cpu")
t.backends.cudnn.enabled = True

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=200, metavar='N', help='number of epochs to train')
parser.add_argument('--wd', type=float, default=1e-3, help='weight_decay')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument("--hid_feats", type=int, default=1500, help='Hidden layer dimensionalities.')
parser.add_argument("--out_feats", type=int, default=732, help='Output layer dimensionalities.')
parser.add_argument("--method", default='sum', help='Merge feature method')
parser.add_argument("--num_layers", type=int, default=2, help='Number of GNN layers.')
parser.add_argument("--input_dropout", type=float, default=0, help='Dropout applied at input layer.')
parser.add_argument("--layer_dropout", type=float, default=0, help='Dropout applied at hidden layers.')
parser.add_argument('--random_seed', type=int, default=123, help='random seed')
parser.add_argument('--k', type=int, default=4, help='k order')
parser.add_argument('--early_stopping', type=int, default=200, help='stop')
parser.add_argument('--dropout', type=float, default=0.0, help='dropout')
parser.add_argument('--mlp', type=list, default=[64, 2], help='mlp layers')
parser.add_argument('--neighbor', type=int, default=20, help='neighbor')
parser.add_argument('--save_score', default='True', help='save_score')

args = parser.parse_args()  
args.dd2 = True
args.data_dir = '/root/autodl-tmp/HGTMDA-main/dataset/'
args.result_dir = 'result/'  
args.save_score = True if str(args.save_score) == 'True' else False



def loading():
    data = dict()
   
    data['s_all_sample_drug_protein'] = pd.read_csv(args.data_dir + 's_all_sample_drug_protein.csv', header=None).iloc[:, :].values
 
    data['drug'] = pd.read_csv(args.data_dir + 'node/drugs_node.csv', header=None).iloc[:, :1].values
    data['protein'] = pd.read_csv(args.data_dir + 'node/protein_node.csv', header=None).iloc[:, :1].values
    data['drug_protein'] = np.concatenate((data['drug'], data['protein']), axis=0)
   
    data['attr_drug_protein_feature'] = pd.read_csv(args.data_dir + 'drug_target/attr_drug_protein_matrix.csv', header=None).iloc[:,
                                    :].values
    
    drug_embedding = np.loadtxt(args.data_dir + 'drug_embedding.txt',dtype=float,delimiter=None,unpack=False)
    data['drug_embedding'] = drug_embedding[:732]

   
    protein_embedding = np.loadtxt(args.data_dir + 'protein_embedding.txt',dtype=float,delimiter=None,unpack=False)
    data['protein_embedding'] = protein_embedding[:1915]
    data['drug_protein_embedding'] = np.concatenate((data['drug_embedding'], data['protein_embedding']), axis=0)
    data['inter_drug_protein_feature'] = np.concatenate((data['attr_drug_protein_feature'][:2647,], data['drug_protein_embedding']), axis=1)
    return data

def make_index(data, sample):
    sample_index = []
    for i in range(sample.shape[0]):
        idx = np.where(sample[i][0] == data['drug_protein'])
        idy = np.where(sample[i][1] == data['drug_protein'])
        sample_index.append([idx[0].item(), idy[0].item()])
    sample_index = np.array(sample_index)
    return sample_index

if __name__ == '__main__':
    
    dataset = loading()
    dataset['inter_drug_protein_feature'] = t.FloatTensor(dataset['inter_drug_protein_feature']).to(device)
    model = MDA(args).to(device)
    optimizer = optim.AdamW(model.parameters(), weight_decay=args.wd, lr=args.lr)
    scheduler = t.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    cross_entropy = nn.BCELoss(reduction='mean')
    file_num = 1


    auc = 0
    auprc = 0
    acc = 0
    f1 = 0
    recall = 0
    pre = 0
   
    max_test_acc = 0
    
    k = 1
    kfold = KFold(n_splits=5, shuffle=True, random_state=123)
    for train_index, test_index in kfold.split(dataset['s_all_sample_drug_protein'][:, :2]):
        tran_sample = dataset['s_all_sample_drug_protein'][train_index][:, :2]
        tran_sample_index = make_index(dataset, tran_sample)
        tran_label = dataset['s_all_sample_drug_protein'][train_index][:, 2]
        test_sample = dataset['s_all_sample_drug_protein'][test_index][:, :2]
        test_sample_index = make_index(dataset, test_sample)
        test_label = dataset['s_all_sample_drug_protein'][test_index][:, 2]        
        tran_sample_index = t.FloatTensor(tran_sample_index).to(device)
        tran_label = t.FloatTensor(tran_label.astype(int)).to(device)
        test_sample_index = t.FloatTensor(test_sample_index).to(device)
        test_label = t.FloatTensor(test_label.astype(int)).to(device)
        for i in range(args.epochs):
            model.train()
            optimizer.zero_grad()
            train_score, test_score = model(dataset, tran_sample_index, test_sample_index, device)
            train_score = train_score.squeeze(1)

            train_label = tran_label.to(device)

            train_loss = cross_entropy(train_score, train_label)
            train_loss.backward()
            train_auc = roc_auc_score(train_label.detach().cpu().numpy(), train_score.detach().cpu().numpy())
            train_acc = accuracy_score(train_label.detach().cpu().numpy().astype(np.int64),np.rint(train_score.detach().cpu().numpy()).astype(np.int64))
            optimizer.step()
            model.eval()
            test_score = test_score.squeeze(1)
            test_label = test_label.to(device)
            # test_loss = cross_entropy(test_score, test_label)
            test_auc = roc_auc_score(test_label.detach().cpu().numpy(),
                                     test_score.detach().cpu().numpy())
            test_acc = accuracy_score(test_label.detach().cpu().numpy().astype(np.int64),
                                      np.rint(test_score.detach().cpu().numpy()).astype(np.int64))
            test_aupr = average_precision_score(test_label.detach().cpu().numpy(), test_score.detach().cpu().numpy())
            test_f1 = f1_score(test_label.detach().cpu().numpy(),
                               np.rint(test_score.detach().cpu().numpy()).astype(np.int64), average='macro')
            test_recall = recall_score(test_label.detach().cpu().numpy(),
                                       np.rint(test_score.detach().cpu().numpy()).astype(np.int64), average='macro')
            test_pre = precision_score(test_label.detach().cpu().numpy(),
                                       np.rint(test_score.detach().cpu().numpy()).astype(np.int64), average='macro')
                # 记录日志
            logging.info(f'Epoch: {i + 1}/{args.epochs}, Train Loss: {train_loss.item():.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f},Train Auc.: {train_auc:.4f},Test Auc.: {test_auc:.4f}')
            
            if test_acc > max_test_acc:
                t.save(model.state_dict(), f"/root/autodl-tmp/HGTMDA-main/log/train_model_{max_test_acc}_max_test_acc.pth")
                max_test_acc = test_acc
                auc = test_auc
                auprc = test_aupr
                acc = test_acc
                f1 = test_f1
                recall = test_recall
                pre = test_pre
                print(f'Epoch: {i + 1:03d}/{args.epochs:03d}' f'   | Learning Rate {scheduler.get_last_lr()[0]:.6f}')
                # print(f'Epoch: {i + 1:03d}/{args.epochs:03d}')
                print(f'Train Auc.: {train_auc:.4f}' f' | Test Auc.: {test_auc:.4f}')
                print(f'Train Loss.: {train_loss.item():.4f}')
                print(f'Train Acc.: {train_acc:.4f}' f' | Test Acc.: {test_acc:.4f}')
            scheduler.step()
    print(f' | Test Auc.: {auc:.4f}')
    print(f' | Test Auprc.: {auprc:.4f}')
    print(f' | Test Acc.: {acc:.4f}')
    print(f' | Test F1.: {f1:.4f}')
    print(f' | Test Recall.: {recall:.4f}')
    print(f' | Test Precision.: {pre:.4f}')
