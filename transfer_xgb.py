import scipy as sp
import numpy as np
from sklearn.cross_validation import train_test_split

from sklearn.metrics import precision_recall_curve, roc_curve, auc,roc_auc_score
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib
import xgboost as xgb
import time
def train(data_dir,data_name_dir,model_dir):
    start_time = time.time()
    data = []
    labels = []
    fr1 = open(data_dir,'r')
    fr2 = open(data_name_dir,'r')
    for line in fr1.readlines():
        term = line.strip().split(' ')
        data_x = []
        for item in term:
            data_x.append(float(item))
        data.append(data_x)
    
    for line in fr2.readlines():
        label = 0 if line == 'good\n' else 1 
        labels.append(label)

    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size = 0.1)
    xgb_train = xgb.DMatrix(x_train,y_train)
    xgb_test  = xgb.DMatrix(x_test,y_test)
    #调参步骤：max_depth(3-10) ->min_child_weight(2-12 ,step=2)->  
    #        -->gamma(0-0.5 step=0.1)  -> sample(0.6-1,step=0.1)
    #        -->alpha(1e-5-100,bi-cro)
    
    #resnet152-1536 <4,5,0.4,0.85,0.65,10,0.25>
    #mobilev2       <3,4,0.4,0.85,0.65,12,0.25>
    #dense201-960   <6,6,0.2,0.9, 0.7, 10,0.25>
    params = {
        'booster':'gbtree', 'silent':1 , 'eta': 0.05,#(0.01-0.2)
        
        'max_depth':3, 'min_child_weight':4, 
        'gamma':0.4,'subsample':0.85, 'colsample_bytree':0.65, 
        'lambda':12, 'alpha':0.1, 
        
        'seed':1000, 'eval_metric': 'auc'
        }
    plst = list(params.items())
    num_rounds = 700 # 迭代次数
    watchlist = [(xgb_train, 'train'),(xgb_test, 'val')]
    
    #训练模型并保存
    # early_stopping_rounds 当设置的迭代次数较大时，early_stopping_rounds 可在一定的迭代次数内准确率没有提升就停止训练
    model = xgb.train(plst, xgb_train, num_rounds, watchlist)
    
    print ("best best_ntree_limit",model.best_ntree_limit) 
    y_pred = model.predict(xgb_test,ntree_limit=model.best_ntree_limit)
    print(y_pred.mean(),y_pred.var())
    print ('error=%f' % (  sum(1 for i in range(len(y_pred)) if int(y_pred[i]>0.5)!=y_test[i]) /float(len(y_pred))))  
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred)      
    report = y_pred > 0.5      
    print(classification_report(y_test, report, target_names = ['neg', 'pos']))  
    #输出运行时长
    cost_time = time.time()-start_time
    print ("xgboost success!",'\n',"cost time:",cost_time,"(s)......")

    model.save_model(model_dir)

def test(model_dir,test_dir,test_name_dir,res_dir):
    model = xgb.Booster()
    model.load_model(model_dir)
    x_test = []
    names = []
    fr1 = open(test_dir,'r')
    fr2 = open(test_name_dir,'r')
    for line in fr1.readlines():
        term = line.strip().split(' ')
        data_x = []
        for item in term:
            data_x.append(float(item))
        x_test.append(data_x)
    for line in fr2.readlines():
        term = line.strip().split('\\')
        names.append(term[0])
    dtest = xgb.DMatrix(x_test)

    answer = model.predict(dtest)
    print(answer.mean(),answer.var())
    fw1 = open(res_dir,'w')
    fw1.write('filename,probability\n')
    for item,n in zip(answer,names):
        if item >= 1.0:
            item = 0.999999
        item = "%.6f"%item
        if item == '0.000000':
            item = '0.000001' 
        if item == '1.000000':
            item = '0.999999'        
        fw1.write(n+','+item+'\n')
    fw1.close()

def model_hy(res1,res2,res3,out):
    fr1 = open(res1,'r')
    fr2 = open(res2,'r')
    fr3 = open(res3,'r')
    fw1 = open(out,'w')
    index = 0
    for line1,line2,line3 in zip(fr1.readlines(), fr2.readlines(), fr3.readlines()):
        if(index == 0):
            fw1.write('filename,probability\n')
        else:
            term1 = line1.strip().split(',')
            term2 = line2.strip().split(',')
            term3 = line3.strip().split(',')
            pro1 = float(term1[1])
            pro2 = float(term2[1])
            pro3 = float(term3[1])
            pro = (1.5*pro1+0.5*pro2+8*pro3)/10
            fw1.write(term1[0]+','+"%.6f"%pro+'\n')
        index += 1
    fw1.close()



dir = 'data/xgb/dense/'
model_dir      = dir + 'dense_t_xgb880.model'
data_dir       = dir + 'net_data.csv'
data_label_dir = dir + 'net_label.csv'
test_dir       = dir + 'net_test.csv'
test_name_dir  = dir + 'net_test_name.csv'
res_dir        = dir + 'res.csv'
#train(data_dir,data_label_dir,model_dir)
#test(model_dir,test_dir,test_name_dir,res_dir)

res1 = dir + 'res880.csv'
res2 = dir + 'res896.csv'
res3 = dir + 'res899.csv'
out  = dir + 'reshy.csv'
model_hy(res1,res2,res3,out)


