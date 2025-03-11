import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import Linear, HGTConv
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data


drug_node = pd.read_csv('/root/autodl-tmp/HGTMDA-main/dataset/node/drugs_node.csv', header=None)
protein_node = pd.read_csv('/root/autodl-tmp/HGTMDA-main/dataset/node/protein_node.csv', header=None)
disease_node = pd.read_csv('/root/autodl-tmp/HGTMDA-main/dataset/node/disease_node.csv', header=None)

edge_list_drug_disease = pd.read_csv('/root/autodl-tmp/HGTMDA-main/dataset/drug_association/drug_disease_association.csv', header=None)
edge_list_drug_drug = pd.read_csv('/root/autodl-tmp/HGTMDA-main/dataset/drug_association/drug_drug_association.csv', header=None)
edge_list_drug_protein = pd.read_csv('/root/autodl-tmp/HGTMDA-main/dataset/drug_association/drug_protein_association.csv', header=None)
edge_list_protein_disease= pd.read_csv('/root/autodl-tmp/HGTMDA-main/dataset/drug_association/protein_disease_association.csv', header=None)
edge_list_protein_protein= pd.read_csv('/root/autodl-tmp/HGTMDA-main/dataset/drug_association/protein_protein_association.csv', header=None)


id_mapping = {}
for node_type in [drug_node, protein_node, disease_node]:
    for node_id in range(len(node_type)):
        if node_id not in id_mapping:
            id_mapping[node_type[0][node_id]] = node_type[1][node_id]

for edge_list in [edge_list_drug_disease, edge_list_drug_drug, edge_list_drug_protein, edge_list_protein_disease, edge_list_protein_protein]:
    edge_list[0] = edge_list[0].map(id_mapping)
    edge_list[1] = edge_list[1].map(id_mapping)
   


x_dict = {"drug": torch.randn(len(drug_node), 732),
          "disease": torch.randn(len(disease_node), 732), 
          "protein": torch.randn(len(protein_node), 732), 
          }


edge_index_dict = {
    ('drug', 'to', 'disease'): torch.tensor([edge_list_drug_disease[0].apply(lambda x: int(x[4:])).tolist(), edge_list_drug_disease[1].apply(lambda x: int(x[7:])).tolist()]),
    ('drug', 'to', 'protein'): torch.tensor([edge_list_drug_protein[0].apply(lambda x: int(x[4:])).tolist(), edge_list_drug_protein[1].apply(lambda x: int(x[7:])).tolist()]),
    ('drug', 'to', 'drug'): torch.tensor([edge_list_drug_drug[0].apply(lambda x: int(x[4:])).tolist(), edge_list_drug_drug[1].apply(lambda x: int(x[4:])).tolist()]),
    ('protein', 'to', 'disease'): torch.tensor([edge_list_protein_disease[1].apply(lambda x: int(x[7:])).tolist(), edge_list_protein_disease[0].apply(lambda x: int(x[7:])).tolist()]),
    ('protein', 'to', 'protein'): torch.tensor([edge_list_protein_protein[0].apply(lambda x: int(x[7:])).tolist(), edge_list_protein_protein[1].apply(lambda x: int(x[7:])).tolist()]),
}
node_types = {"drug","disease", "protein"}
metadata = (["drug","disease", "protein"],
            [('drug', 'to', 'disease'),
             ('drug', 'to', 'protein'),
            ('drug', 'to', 'drug'),
            ('protein', 'to', 'disease'),
            ('protein', 'to', 'protein')
            ])



drug_labels = drug_node[2]
disease_labels = disease_node[2]
protein_labels = protein_node[2]


node_type_labels = {"drug":torch.tensor(drug_labels, dtype=torch.long),
                    "disease":torch.tensor(disease_labels, dtype=torch.long),
                    "protein":torch.tensor(protein_labels, dtype=torch.long)}

# data = HeteroData()
data = Data(x_dict=x_dict, edge_index_dict=edge_index_dict, node_types = node_types, metadata = metadata, labels = node_type_labels)
pass

class HGT(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_heads, num_layers):
        super().__init__()

        self.lin_dict = torch.nn.ModuleDict()
        for node_type in data.node_types:
            self.lin_dict[node_type] = Linear(-1, hidden_channels) # -1表示根据输入的数据的维度自动调整

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HGTConv(hidden_channels, hidden_channels, data.metadata,
                           num_heads)
            self.convs.append(conv)

        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        x_dict = {
            node_type: self.lin_dict[node_type](x).relu_()
            for node_type, x in x_dict.items()
        }

        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
        # for ntype in data.node_types:
        #     x_dict[ntype] = self.lin(x_dict[ntype])
        return torch.sigmoid(self.lin(x_dict['drug'])), torch.sigmoid(self.lin(x_dict['protein']))


model = HGT(hidden_channels=1024, out_channels=732, num_heads=2, num_layers=2)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data, model = data.to(device), model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
torch.autograd.set_detect_anomaly(True)

for epoch in range(50):
    model.train()  
    optimizer.zero_grad()
    logits_drug_dict, logits_protein_dict = model(data.x_dict, data.edge_index_dict)
    drug_labels = data.labels['drug']
    drug_loss = criterion(logits_drug_dict, drug_labels)
    drug_loss.backward(retain_graph = True)
    protein_labels = data.labels['protein']
    protein_loss = criterion(logits_protein_dict, protein_labels)
    protein_loss.backward()
    optimizer.step()
    print(f"Epoch {epoch},  drug Loss: {drug_loss.item()}, protein Loss: {protein_loss.item()}")

model.eval() 
with torch.no_grad():
    drug_embedding, protein_embedding = model(data.x_dict, data.edge_index_dict)
print(f"drug embedding: {drug_embedding}, protein embedding: {protein_embedding}")
np.savetxt('/root/autodl-tmp/HGTMDA-main/dataset/drug_embedding.txt', drug_embedding.tolist(), fmt="%f", comments='')
np.savetxt('/root/autodl-tmp/HGTMDA-main/dataset/protein_embedding.txt', protein_embedding.tolist(), fmt="%f", comments='')


