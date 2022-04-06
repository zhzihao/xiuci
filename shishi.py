import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
from transformers import BertTokenizer
from transformers import BertModel
from dataset1 import Datasetcouplet
from utils import Metric
model_name='hfl/chinese-roberta-wwm-ext'
model_path='./model/robert.pt'
added_token=["[unused1]"]#加入特殊标记用来标志每个分句末尾
token=BertTokenizer.from_pretrained(model_name,additional_special_tokens=added_token)
pretrained=BertModel.from_pretrained(model_name)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
pretrained=pretrained.to(device)
datasets=Datasetcouplet("/home/jiebin/xiucibio/data/")
train_size=int(len(datasets)*0.7)
valid_size=int(len(datasets)*0.1)
test_size=len(datasets)-train_size-valid_size
print('train_size',train_size)
train_datasets,valid_datasets,test_datasets=torch.utils.data.random_split(datasets,[train_size,valid_size,test_size])
epochs=10000
lb2id={"B":0,"I":1,"O":2}
id2lb=["B","I","O"]
def collate_fn(data):#传入的是Datasets[j]
  sents=[i for i in data[0]] #data[0]就是datasets[j][0]即seq[j]，为一个列表，对应第j个context。i为context[j]中的每一个分句
  labels=[lb2id[i] for i in data[1]]#data[1]是Datasets[j][1]即biao[j]与sents对应
  sent_all=[str(i) for i in sents]
  sents="[unused1]".join(sent_all)+"[unused1]"#将每个分句间加入特殊符号连成整体
  data=token.encode_plus(text=sents,truncation=True,padding='max_length',max_length=512,return_tensors='pt',return_length=True)
  input_ids=data['input_ids']
  attention_mask=data['attention_mask']
  token_type_ids=data['token_type_ids']
  input_num=input_ids[0].tolist()
  query=token.get_vocab()["[unused1]"]
  biao_pl=[]#记录每个unused1的位置，为模型输出层的预测做准备
  for i in range(len(input_num)):
    if input_num[i]==query and i<512:
      biao_pl.append(i)
  labels=labels[0:len(biao_pl)]#因为会截断到512，所以labels需要与biao_pl对齐
  biao_pl=torch.tensor(biao_pl)
  labels=torch.LongTensor(labels)
  return input_ids,attention_mask,token_type_ids,labels,biao_pl

class Model(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.fc=torch.nn.Linear(768,3)
  def forward(self, input_ids,attention_mask,token_type_ids,biao_pl):
    #with torch.no_grad():
    out=pretrained(input_ids=input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids)
    #out.last_hidden_state [1,512,768]
    out=torch.index_select(out.last_hidden_state,1,biao_pl)
    #out [1,分句数,768]
    out=self.fc(out)
    #out [1,分句数，3]
    out=out.softmax(dim=2)
    return out
model=Model()
model.to(device)

from transformers import AdamW
#训练
criterion = torch.nn.CrossEntropyLoss()
def train():
  f1=0
  optimizer = AdamW(model.parameters(), lr=3e-5)
  model.zero_grad()
  num_step=0
  j=0
  for epoch in range(epochs):
    model.zero_grad()
    for i in train_datasets:
        data=collate_fn(i)
        input_ids=data[0]
        attention_mask=data[1]
        token_type_ids=data[2]
        labels=data[3]
        biao_pl=data[4]
        input_ids=input_ids.to(device)
        attention_mask= attention_mask.to(device)
        token_type_ids=token_type_ids.to(device)
        labels=labels.to(device)
        biao_pl=biao_pl.to(device)
        model.train()
        out = model(input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,biao_pl=biao_pl)
        out=torch.squeeze(out,0)# [分句数，3]
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        j+=1
        if j % 100 == 0:
            out = out.argmax(dim=1)
            accuracy = (out == labels).sum().item() / len(labels)
            print(j, loss.item(), accuracy)
            """print(out,"\n")
            out = out.argmax(dim=1)
            print(out,"\n",labels)
            accuracy = (out == labels).sum().item() / len(labels)
            print(j, loss.item(), accuracy)"""
    f1_now=valid()
    if(f1_now>f1):
      torch.save(model.state_dict(), model_path)    
def test():
  model.eval()
  correct = 0
  total = 0
  metric=Metric(id2lb)
  for i in test_datasets:
        data=collate_fn(i)      
        input_ids=data[0]
        attention_mask=data[1]
        token_type_ids=data[2]
        labels=data[3]
        biao_pl=data[4]
        input_ids=input_ids.to(device)
        attention_mask= attention_mask.to(device)
        token_type_ids=token_type_ids.to(device)
        labels=labels.to(device)
        biao_pl=biao_pl.to(device)
        with torch.no_grad():
            out = model(input_ids=input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids,biao_pl=biao_pl)
        out=torch.squeeze(out,0)
        out = out.argmax(dim=1)
        correct += (out == labels).sum().item()
        metric.add(out.cpu().numpy().tolist(),labels)
        total += len(labels)
  for j in range(3):
    p,r,f1=metric.get_tag(j)
    print(id2lb[j],'P: {:5.2f} | R: {:5.2f} | F1: {:5.2f}'.format(p * 100, r * 100, f1 * 100))
  print(correct / total)
def valid():
  model.eval()
  correct = 0
  total = 0
  metric=Metric(id2lb)
  for i in valid_datasets:
        data=collate_fn(i)      
        input_ids=data[0]
        attention_mask=data[1]
        token_type_ids=data[2]
        labels=data[3]
        biao_pl=data[4]
        input_ids=input_ids.to(device)
        attention_mask= attention_mask.to(device)
        token_type_ids=token_type_ids.to(device)
        labels=labels.to(device)
        biao_pl=biao_pl.to(device)
        with torch.no_grad():
            out = model(input_ids=input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids,biao_pl=biao_pl)
        out=torch.squeeze(out,0)
        out = out.argmax(dim=1)
        correct += (out == labels).sum().item()
        metric.add(out.cpu().numpy().tolist(),labels)
        total += len(labels)
  for j in range(3):
    p,r,f1=metric.get_tag(j)
    print(id2lb[j],'P: {:5.2f} | R: {:5.2f} | F1: {:5.2f}'.format(p * 100, r * 100, f1 * 100))
  print(correct / total)
  return f1
def main():
  #model.load_state_dict(torch.load(model_path))
  train()
  test()   
main() 