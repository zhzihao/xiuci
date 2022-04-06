#coding=gbk
import torch
import os
from torch.utils.data import Dataset
import torch.nn as nn
import codecs
import numpy as np
import re
def cut_sent(para):
    para = re.sub('([!:����;)\\\?])([^����])', r"\1\n\2", para)  # ���ַ��Ͼ��
    para = re.sub('([,])([^��])', r"\1\n\2", para)  # ���ַ��Ͼ��
    para = re.sub('(\.{6})([^����])', r"\1\n\2", para)  # Ӣ��ʡ�Ժ�
    para = re.sub('(\��{2})([^����])', r"\1\n\2", para)  # ����ʡ�Ժ�
    para = re.sub('(\��{2})([^����])', r"\1\n\2", para)
    para=re.sub('(\_{2})([^����])', r"\1\n\2", para)
    para = re.sub('([!��(����)(����);\?][����])([^,!����:;\?])', r'\1\n\2', para)
    para = re.sub('([,][��])([^,!����:;\?])', r'\1\n\2', para)
    para= re.sub('([��])([^,!����:;\?])', r'\1\n\2', para)#�������ֿ�
    # ���˫����ǰ����ֹ������ô˫���Ų��Ǿ��ӵ��յ㣬�ѷ־��\n�ŵ�˫���ź�ע��ǰ��ļ��䶼С�ı�����˫����
    para = para.rstrip()  # ��β����ж����\n��ȥ����
    return para.split("\n")
def cut_len(para,begin):
    para=para[0:begin]
    para = re.sub('([!:����;)\\\?])([^����])', r"\1\n\2", para)  # ���ַ��Ͼ��
    para = re.sub('([,])([^��])', r"\1\n\2", para)  # ���ַ��Ͼ��
    para = re.sub('(\.{6})([^����])', r"\1\n\2", para)  # Ӣ��ʡ�Ժ�
    para = re.sub('(\��{2})([^����])', r"\1\n\2", para)  # ����ʡ�Ժ�
    para = re.sub('(\��{2})([^����])', r"\1\n\2", para)
    para=re.sub('(\_{2})([^����])', r"\1\n\2", para)
    para = re.sub('([!��(����)(����);\?][����])([^,!����:;\?])', r'\1\n\2', para)
    para = re.sub('([,][��])([^,!����:;\?])', r'\1\n\2', para)
    para= re.sub('([��])([^,!����:;\?])', r'\1\n\2', para)#�������ֿ�
    # ���˫����ǰ����ֹ������ô˫���Ų��Ǿ��ӵ��յ㣬�ѷ־��\n�ŵ�˫���ź�ע��ǰ��ļ��䶼С�ı�����˫����
    para = para.rstrip()  # ��β����ж����\n��ȥ����
    return len(para.split("\n"))
class Datasetcouplet(torch.utils.data.Dataset):
  def __init__(self,data_path):
    self.raw1=None  
    with open(os.path.join(data_path, 'final_data.txt'), "r", encoding='utf-8') as fr:
      self.raw1=fr.readlines()
    self.context=[] #������һ�λ����ڲ����޴Ǿ���λ��
    self.sen=[]  #�����޴ǵľ伶��λ
    self.seq=[]  #�б���б� �ڲ����б���һ��content��ֺ��Ӧ�ľ��� seq[i]��ʾ��i��content
    self.biao=[] #��ѯseq��Ӧ��bio
    self.preprocess()
    
  def preprocess(self):  #��Ӻ����޴ǵ����ͬʱ�������еĻ���֣�ÿ���ȶ���Ӧ��O
    seq_cnt=-1 #�������ڵڼ���context��
    seqq=[]#ÿ��context��Ӧ�ķ־�
    biaoo=[]#ÿ��context�־��Ӧ�ı�ע
    for item in self.raw1:
      item = item.strip().split(':',1)#��ֹ�������еģ�Ҳ�ָ���
      if(item[0]=="\"content\""):
        li=item[1].strip().strip(",").strip("\"")
        self.context.append(li)
        cut_li=cut_sent(li)#��context���ձ������и��һ�ζη־�
        for i in cut_li:
          seqq.append(i)
          biaoo.append("O")#�Ƚ����з־��Ϊ��O��
        self.seq.append(seqq)
        self.biao.append(biaoo)
        seqq=[]
        biaoo=[]
        seq_cnt+=1
      if(item[0]=="\"sentence\""):
        self.sen.append(item[1].strip().strip(",").lstrip('\"')[:-1])
      if(item[0]=="\"begin\""):
        begin=int(item[1].strip().strip(","))#begin��λ�����޴ǵ�λ�׸��ַ���λ��
        len1=cut_len(self.sen[-1],len(self.sen[-1])) #���޴ǵľ��ӱ��г��˼���
        len_seq=cut_len(self.context[seq_cnt],begin)#��0��begin��ǰ��ľ��ӱ��ָ���˼���
        len_all=cut_len(self.context[seq_cnt],len(self.context[seq_cnt]))#����context���ֳ��˶೤����ֹԽ��
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