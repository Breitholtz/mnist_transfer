from experiments.training import *
from util.kl import *

def joint_error(newlist,newlist2,true_label):
    ## expected joint error
        shapes=newlist.shape
        e_s=0
        # e_S= E_h,h' E_x,y L(h(x),y)L(h'(x),y)     
        for i in range(shapes[0]):
            for j in range(shapes[1]):
                e_s+=(newlist[i][j]-true_label[i][j])*(newlist2[i][j]-true_label[i][j])
        e_s/=(2*shapes[0])
        return e_s

def classifier_disagreement(newlist,newlist2):
    ### classifier disagreement, i.e. R(h,h')= 1/n sum(L( h(x),h'(x) ))
    shapes=newlist.shape
    d=0
    arr=np.abs(newlist-newlist2)
    for i in arr:
        if np.sum(i)==2:
            d+=1
    d/=(shapes[0])
    return d
def calculate_germain_bound(train_error,e_s,e_t,d_tx,d_sx, KL,delta,a,omega,m,L):
    bound=[]
    aprime=2*a/(1-np.exp(-2*a))
    omegaprime=omega/(1-np.exp(-omega))
    a1=np.zeros(L)
    a2=np.zeros(L)
    a3=np.zeros(L)
    a4=np.zeros(L)
    a5=np.zeros(L)
    for i in range(L):
        dis_rho=np.abs(e_t[i]-e_s[i])
        lambda_rho=np.abs(d_tx[i]-d_sx[i])
        a1[i]=omegaprime*train_error[i]
        a2[i]=aprime/2*(dis_rho)
        a3[i]=(omegaprime/omega+aprime/a)*(KL[i]+np.log(3/delta))/m
        a4[i]=lambda_rho
        a5[i]=(aprime-1)/2
        bound.append(a1[i]+a2[i]+a3[i]+a4[i]+a5[i])
    return bound,a1,a2,a3,a4,a5

def make_01(predictions):
    ## takes in non integer predictions and returns 1 for the most likely prediction and 0 for the others
    newlist=np.zeros(predictions.shape)
    for i, row in enumerate(predictions):
        idx = np.where(row == np.amax(row))
        row=np.zeros(row.shape)
        row[idx]=1
        newlist[i]=row
    return newlist
