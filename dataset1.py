#coding=gbk
import torch
import os
from torch.utils.data import Dataset
import torch.nn as nn
import codecs
import numpy as np
import re
def cut_sent(para):
    para = re.sub('([!:、。;)\\\?])([^”’])', r"\1\n\2", para)  # 单字符断句符
    para = re.sub('([,])([^“])', r"\1\n\2", para)  # 单字符断句符
    para = re.sub('(\.{6})([^”’])', r"\1\n\2", para)  # 英文省略号
    para = re.sub('(\…{2})([^”’])', r"\1\n\2", para)  # 中文省略号
    para = re.sub('(\—{2})([^”’])', r"\1\n\2", para)
    para=re.sub('(\_{2})([^”’])', r"\1\n\2", para)
    para = re.sub('([!。(——)(……);\?][”’])([^,!。、:;\?])', r'\1\n\2', para)
    para = re.sub('([,][“])([^,!。、:;\?])', r'\1\n\2', para)
    para= re.sub('([“])([^,!。、:;\?])', r'\1\n\2', para)#单个“分开
    # 如果双引号前有终止符，那么双引号才是句子的终点，把分句符\n放到双引号后，注意前面的几句都小心保留了双引号
    para = para.rstrip()  # 段尾如果有多余的\n就去掉它
    return para.split("\n")
def cut_len(para,begin):
    para=para[0:begin]
    para = re.sub('([!:、。;)\\\?])([^”’])', r"\1\n\2", para)  # 单字符断句符
    para = re.sub('([,])([^“])', r"\1\n\2", para)  # 单字符断句符
    para = re.sub('(\.{6})([^”’])', r"\1\n\2", para)  # 英文省略号
    para = re.sub('(\…{2})([^”’])', r"\1\n\2", para)  # 中文省略号
    para = re.sub('(\—{2})([^”’])', r"\1\n\2", para)
    para=re.sub('(\_{2})([^”’])', r"\1\n\2", para)
    para = re.sub('([!。(——)(……);\?][”’])([^,!。、:;\?])', r'\1\n\2', para)
    para = re.sub('([,][“])([^,!。、:;\?])', r'\1\n\2', para)
    para= re.sub('([“])([^,!。、:;\?])', r'\1\n\2', para)#单个“分开
    # 如果双引号前有终止符，那么双引号才是句子的终点，把分句符\n放到双引号后，注意前面的几句都小心保留了双引号
    para = para.rstrip()  # 段尾如果有多余的\n就去掉它
    return len(para.split("\n"))
class Datasetcouplet(torch.utils.data.Dataset):
  def __init__(self,data_path):
    self.raw1=None  
    with open(os.path.join(data_path, 'final_data.txt'), "r", encoding='utf-8') as fr:
      self.raw1=fr.readlines()
    self.context=[] #完整的一段话用于查找修辞句子位置
    self.sen=[]  #包含修辞的句级单位
    self.seq=[]  #列表的列表 内部的列表是一个content拆分后对应的句子 seq[i]表示第i个content
    self.biao=[] #查询seq对应的bio
    self.preprocess()
    
  def preprocess(self):  #添加含有修辞的类别，同时将段落中的话拆分，每段先都对应到O
    seq_cnt=-1 #标明处于第几个context中
    seqq=[]#每个context对应的分句
    biaoo=[]#每个context分句对应的标注
    for item in self.raw1:
      item = item.strip().split(':',1)#防止将句子中的：也分割了
      if(item[0]=="\"content\""):
        li=item[1].strip().strip(",").strip("\"")
        self.context.append(li)
        cut_li=cut_sent(li)#将context按照标点符号切割成一段段分句
        for i in cut_li:
          seqq.append(i)
          biaoo.append("O")#先将所有分句标为“O”
        self.seq.append(seqq)
        self.biao.append(biaoo)
        seqq=[]
        biaoo=[]
        seq_cnt+=1
      if(item[0]=="\"sentence\""):
        self.sen.append(item[1].strip().strip(",").lstrip('\"')[:-1])
      if(item[0]=="\"begin\""):
        begin=int(item[1].strip().strip(","))#begin的位置是修辞单位首个字符的位置
        len1=cut_len(self.sen[-1],len(self.sen[-1])) #含修辞的句子被切成了几段
        len_seq=cut_len(self.context[seq_cnt],begin)#从0到begin，前面的句子被分割成了几段
        len_all=cut_len(self.context[seq_cnt],len(self.context[seq_cnt]))#整个context被分成了多长，防止越界
        for i in range (len_seq+1,len_seq+len1):
          if(i<len_all):
            self.biao[seq_cnt][i]="I"
        if(len_seq<len_all):   
          self.biao[seq_cnt][len_seq]="B"
  def __len__(self):
    assert len(self.seq)==len(self.biao)
    return len(self.seq)
  def __getitem__(self, index):
    return self.seq[index],self.biao[index]