import numpy as np
class Metric:
    def __init__(self, id2lb):
        self.gold_num = np.zeros(3)
        self.pred_num = np.zeros(3)
        self.correct = np.zeros(3)
        self.id2lb = id2lb

    def add(self, out, labels):
        for tag in range(3):
          pred_tag = self.get_pred_tag(out, tag)
          gold_tag = self.get_gold_tag(labels,tag)
          self.gold_num[tag] +=len(gold_tag)
          self.pred_num [tag]+=len(pred_tag)
          self.correct[tag] += len(self.got_right(out,labels,tag))
    def get_tag(self,tag):
        print(self.id2lb[tag],"have examples",self.gold_num[tag])
        if self.pred_num[tag] == 0 or self.gold_num[tag] == 0:
            return 0, 0, 0
        p = self.correct[tag] / self.pred_num[tag]
        r = self.correct[tag] / self.gold_num[tag]
        if p + r == 0:
            return 0, 0, 0
        f1 = 2*p*r / (p + r)
        return p, r, f1
    def get_pred_tag(self,out,tag):
        pred_tag=[]
        for i in range(len(out)):
          if out[i]==tag:
            pred_tag.append(i)
        return pred_tag
    def get_gold_tag(self,labels,tag):
       gold_tag=[]
       for i in range(len(labels)):
         if labels[i]==tag:
           gold_tag.append(i)
       return gold_tag
    def got_right(self,out,labels,tag):
        correct=[]
        for i in range(len(labels)):
          if labels[i]==tag and out[i]==tag:
            correct.append(i)
        return correct
    def get_wrong_example(self,input_ids,out,labels,token):
        for i in range(len(out)):
          if(out[i]!=labels[i]):
            print(token.convert_ids_to_tokens(input_ids[i],skip_special_tokens=True))
            print('predict:',self.id2lb[out[i]])
            print('fact:',self.id2lb[labels[i]])            
        
        