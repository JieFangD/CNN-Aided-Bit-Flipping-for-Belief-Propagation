import numpy as np
import tensorflow as tf
from itertools import permutations
import keras.backend.tensorflow_backend as tfb
import scipy.io
from itertools import combinations
from multiprocessing import Pool
import time

in_N = 0
in_K = 0
N = 16
K = 8 + in_N - in_K
Rc = K/N
# Rc = (K-in_N+in_K)/N
CRC = in_N>0 # 1 = encode CRC, 0 = no CRC
Rm = 1 # BPSK
ebn0 = np.arange(0,3,1,dtype=np.float32)
ebn0_num = 10**(ebn0/10)
SNR = ebn0 + 10*np.log10(Rc*Rm) + 10*np.log10(2)
SNR_num = 10**(SNR/10)
ISI = 0
h = np.array([0.9, 0.6, 0.3])
h = np.array([0.92700798,0.27227378,0.25792089])
h = h/(sum(h**2))**(0.5)
word_seed = 786000
noise_seed = 345000
wordRandom = np.random.RandomState(word_seed)
noiseRandom = np.random.RandomState(noise_seed)
validation_ratio = 0.2
numOfWord = 25000
batch_size = numOfWord*len(ebn0)
batches_train = int(np.round(10/len(ebn0))*len(ebn0))
batches_test = int(np.round(20/len(ebn0))*len(ebn0))
batches_val = int(batches_train*validation_ratio)
batches_train = int(batches_train*(1-validation_ratio))
patience = 10

if(CRC):
    print('Load CRC Matrix: CRC_'+str(in_N)+'_'+str(in_K)+'.mat')
    Data = scipy.io.loadmat('./metrices/CRC_'+str(in_N)+'_'+str(in_K)+'.mat')
    PCM = np.array(Data['PCM'],dtype=np.int8)
    GM = np.array(Data['GM'],dtype=np.int8)

n = int(np.log2(N))
Fi = np.ones([1])
for i in range(n):
    Fi = np.vstack((np.hstack((Fi,Fi*0)),np.hstack((Fi,Fi))))
F_kron_n = Fi
combins = [np.array(c) for c in permutations(np.arange(0,n,1),n)]

indices = np.loadtxt('FrozenBit/'+str(N)+'.txt',dtype=int)-1
FZlookup = np.zeros((N))
FZlookup[indices[:K]] = -1
bitreversedindices = np.zeros((N),dtype=int)
for i in range(N):
    b = '{:0{width}b}'.format(i, width=n)
    bitreversedindices[i] = int(b[::-1], 2)

FER = np.zeros(len(ebn0))
BER = np.zeros(len(ebn0))

def permutation():
    shift = np.zeros((np.math.factorial(n),N),dtype=int)
    r_shift = np.zeros((np.math.factorial(n),N),dtype=int)
    for j in range(np.math.factorial(n)):
        for i in range(N):
            b = '{:0{width}b}'.format(i, width=n)
            b_shift = b[combins[j][0]]
            for k in range(1,n):
                b_shift = b_shift+b[combins[j][k]]
    #         print(b_shift)
            shift[j,i] = int(b_shift, 2)
        r_shift[j,:] = np.argsort(shift[j,:])
# print(shift)
# print(r_shift)
    return shift, r_shift

def dict2arr(_net_dict, bp_iter_num):
    output = np.zeros((batch_size, n+1, N, bp_iter_num, 2))
    for i in range(n+1):
        for j in range(N):
            for k in range(bp_iter_num):
                output[:,i,j,k,0] = _net_dict["output_L_{0}{1}{2}".format(i,j,k)]
                output[:,i,j,k,1] = _net_dict["output_R_{0}{1}{2}".format(i,j,k)]
    return output

def fFunction(a,b):
    c = tf.sign(a)*tf.sign(b)*tf.minimum(tf.abs(a),tf.abs(b))
    return c

def fFunction1(a,b,beta):
    c = tf.sign(a)*tf.sign(b)*tf.maximum(tf.minimum(tf.abs(a),tf.abs(b))-beta,0)
    return c

def fFunction_np(a,b):
    c = np.sign(a)*np.sign(b)*np.minimum(np.abs(a),np.abs(b))
    return c

def BP_decoder(x,y,R_init,bp_iter_num,LW=np.ones((n,N)),RW=np.ones((n,N))):
    net_dict = {}
    L = np.zeros((x.shape[0],n+1,N))
    R = np.zeros((x.shape[0],n+1,N))
    arr = np.zeros((x.shape[0],n+1,N,bp_iter_num))

#     bp_iter_num = 5
    # ss = [0,3,2,1]
    ss = np.hstack([0,np.arange(n)[::-1]+1])
    inf_num = 1000

    for j in range(N):
        L[:,n,j] = x[:,j]
        R[:,0,j] = R_init[:,j]*inf_num

    # bp algorithm
    for k in range(bp_iter_num):
        for i in range(n,0,-1):
            for phi in range(2**ss[i]):
                psi = int(np.floor(phi/2))
                if(np.mod(phi,2)!=0):
                    for omega in range(2**(n-ss[i])):
                        R[:,n+1-i,psi+2*omega*2**(ss[i]-1)] = RW[n-i,psi+2*omega*2**(ss[i]-1)]*fFunction_np(L[:,n+1-i,psi+(2*omega+1)*2**(ss[i]-1)]+R[:,n-i,psi+(2*omega+1)*2**(ss[i]-1)], R[:,n-i,psi+2*omega*2**(ss[i]-1)])
                        R[:,n+1-i,psi+(2*omega+1)*2**(ss[i]-1)] = R[:,n-i,psi+(2*omega+1)*2**(ss[i]-1)]+RW[n-i,psi+(2*omega+1)*2**(ss[i]-1)]*fFunction_np(L[:,n+1-i,psi+2*omega*2**(ss[i]-1)],R[:,n-i,psi+2*omega*2**(ss[i]-1)])
        for i in range(1,n+1):
            for phi in range(2**ss[i]):
                psi = int(np.floor(phi/2))
                if(np.mod(phi,2)!=0):
                    for omega in range(2**(n-ss[i])):
                        L[:,n-i,psi+2*omega*2**(ss[i]-1)] = LW[n-i,psi+2*omega*2**(ss[i]-1)]*fFunction_np(L[:,n+1-i,psi+2*omega*2**(ss[i]-1)],L[:,n+1-i,psi+(2*omega+1)*2**(ss[i]-1)]+R[:,n-i,psi+(2*omega+1)*2**(ss[i]-1)])
                        L[:,n-i,psi+(2*omega+1)*2**(ss[i]-1)] = L[:,n+1-i,psi+(2*omega+1)*2**(ss[i]-1)]+LW[n-i,psi+(2*omega+1)*2**(ss[i]-1)]*fFunction_np(L[:,n+1-i,psi+2*omega*2**(ss[i]-1)],R[:,n-i,psi+2*omega*2**(ss[i]-1)])
        arr[:,:,:,k] = L+R
    y_output = (L[:,0,FZlookup==-1]+R[:,0,FZlookup==-1])*-1
    return y_output, arr

def Parallel_BP_decoder(x,y,R_init,bp_iter_num,n_workers,LW=np.ones((n,N)),RW=np.ones((n,N))):
#     n_workers = 16
    results = [None] * n_workers
    with Pool(processes=n_workers) as pool:
        for i in range(n_workers):
            batch_start = (x.shape[0] // n_workers) * i
            if i == n_workers - 1:
                batch_end = x.shape[0]
            else:
                batch_end = (x.shape[0] // n_workers) * (i + 1)
            results[i] = pool.apply_async(BP_decoder, (x[batch_start: batch_end], y[batch_start: batch_end], R_init[batch_start: batch_end], bp_iter_num, LW, RW))
        pool.close()
        pool.join()

    y_pred = np.zeros((x.shape[0],K))
    arr = np.zeros((x.shape[0],n+1,N,bp_iter_num))
    for i in range(n_workers):
        batch_start = (x.shape[0] // n_workers) * i
        if i == n_workers - 1:
            batch_end = x.shape[0]
        else:
            batch_end = (x.shape[0] // n_workers) * (i + 1)
        y_pred[batch_start: batch_end,:], arr[batch_start: batch_end,:,:,:] = results[i].get()
    return y_pred, arr

def Submatrix(N):
    n = int(np.log2(N))
    Fi = np.ones([1])
    for i in range(n):
        Fi = np.vstack((np.hstack((Fi,Fi*0)),np.hstack((Fi,Fi))))
    F_kron_n = Fi
    combins = [np.array(c) for c in permutations(np.arange(0,n,1),n)]

    indices = np.loadtxt('FrozenBit/'+str(N)+'.txt',dtype=int)-1
    FZlookup = np.zeros((N))
    FZlookup[indices[:K]] = -1
    bitreversedindices = np.zeros((N),dtype=int)
    for i in range(N):
        b = '{:0{width}b}'.format(i, width=n)
        bitreversedindices[i] = int(b[::-1], 2)
    return F_kron_n

def SSFG(m, arr_hard, arr_soft, cr_th, llr_th):
    H = Submatrix(2**m)
    batch_size = arr_hard.shape[0]
    
    _cr_th = np.mean(-1*FZlookup.reshape(-1,2**m),1) # w < w_th
    _cr_th_mask = np.tile(_cr_th<=cr_th,batch_size)   # Repeat for batch_size
    inf_matrix = np.tile(FZlookup*-1,(batch_size,1)).reshape(-1,2**m) # indicate the position of information bit
    check = np.abs(np.mod(np.dot(arr_hard[:,0,:].reshape(-1,2**m),H),2)-arr_hard[:,m,:].reshape(-1,2**m))
    fail_idx = np.where(np.sum(check,1) != 0)[0]
    inf_matrix[fail_idx,:] = 0 # clear the position doesn't pass G_matrix check
    inf_matrix[_cr_th_mask == False,:] *= (np.abs(arr_soft[:,0,:].reshape(-1,2**m)[_cr_th_mask == False,:]) > llr_th)*1 # w > w_th => check abs_LLR
    inf_matrix = inf_matrix.reshape(-1,N)
    return inf_matrix

def get_weight(sess, LV, RV):
    LWeight = sess.run(LV)
    RWeight = sess.run(RV)
    return LWeight, RWeight

def assign_weight(LWeight, RWeight, sess, LV, RV):
    sess.run(LV.assign(LWeight))
    sess.run(RV.assign(RWeight))
    
def quantize(arr,binary_prec):   
    val = 2**(-1*binary_prec+1)
    arr = np.floor(arr/val)*val
    return arr

def fix_pt(_in,int_L,deci_L,_min,_max):
    v = tf.fake_quant_with_min_max_vars(_in, _min, _max, int_L+deci_L+1)
    return v

def quantizeToClosestBinary(arr,binary_prec):
    val = 2**(-1*binary_prec+1)
    
    arr = np.round(arr/val)*val
    arr[arr<0] = 0
    return arr

def quantizeToBins(arr,bin_bit):
    remove_1 = np.setdiff1d(arr,np.ones((1))) # remove 1
    binvals = [np.percentile(remove_1,pr) for pr in np.linspace(0,100,2**(bin_bit))]
    binvals = binvals
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            for k in range(arr.shape[2]):
                    arr[i,j,k] = min(binvals, key=lambda x:abs(x - arr[i,j,k]))
    return arr

def BINARYquantizeToBins(arr,bin_bit,binary_prec):
    BINarr = quantizeToClosestBinary(arr,binary_prec)
    unique, counts = np.unique(BINarr,return_counts=True)
    idx = np.argsort(counts)
#     print('Unique:',unique)
#     print('Counts:',counts)
#     print(idx)
    if(len(unique) > 2**bin_bit):
        binvals = unique[idx[-2**bin_bit:]]
    else:
        binvals = unique
#     print('Binvals:',binvals)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            for k in range(arr.shape[2]):
                    arr[i,j,k] = min(binvals, key=lambda x:abs(x - arr[i,j,k]))
    return arr

def gendata(const,flag,perm):
    X = np.zeros((len(ebn0)*numOfWord,N))
    Y = np.zeros((len(ebn0)*numOfWord,K))
    
    for i in range(len(ebn0)):
        if(CRC):
            u = wordRandom.randint(2,size=(numOfWord,in_K))
            u = np.mod(np.matmul(u,GM),2).reshape(numOfWord*K,)
        else:
            u = wordRandom.randint(2,size=(numOfWord*K))
#             u = np.zeros((numOfWord*K))
        x = np.tile(FZlookup.copy(),(numOfWord))
        x[x==-1] = u # -1's will get replaced by message bits below
        x = np.reshape(x,(-1,N))
        if not(perm):
            x = x[:,bitreversedindices] # bit reversal
        x = np.mod(np.matmul(x,F_kron_n),2) # encode
        tx_waveform = 2*x-1 # bpsk
        if(ISI):
            isi_waveform = tx_waveform.copy()
            for j in range(isi_waveform.shape[0]):
                isi_waveform[j,:] = np.convolve(h,tx_waveform[j,:])[:-len(h)+1]
        if(flag):
            if(ISI):
                ISI_SNR = SNR[const] - 10*np.log10(np.sum(np.abs(isi_waveform)**2,1)/N)
                rx_waveform = noiseRandom.normal(0.0,1.0,tx_waveform.shape)*np.reshape(np.sqrt(1/(10**(ISI_SNR/10))),(numOfWord,1)) + isi_waveform
                # EQ
                for j in range(rx_waveform.shape[0]):
                    rx_waveform[j,:] = np.convolve(trained_h[:,const],rx_waveform[j,:])[:-len(trained_h[:,const])+1]
            else:
                rx_waveform = noiseRandom.normal(0.0,1.0,tx_waveform.shape)*np.sqrt(1/SNR_num[const]) + tx_waveform
            initia_llr = -2*rx_waveform*SNR_num[const] #away 0
        else:
            if(ISI):
                ISI_SNR = SNR[i] - 10*np.log10(np.sum(np.abs(isi_waveform)**2,1)/N)
                rx_waveform = noiseRandom.normal(0.0,1.0,tx_waveform.shape)*np.reshape(np.sqrt(1/(10**(ISI_SNR/10))),(numOfWord,1)) + isi_waveform
                # EQ
                for j in range(rx_waveform.shape[0]):
                    rx_waveform[j,:] = np.convolve(trained_h[:,i],rx_waveform[j,:])[:-len(trained_h[:,i])+1]
            else:
                rx_waveform = noiseRandom.normal(0.0,1.0,tx_waveform.shape)*np.sqrt(1/SNR_num[i]) + tx_waveform   
            initia_llr = -2*rx_waveform*SNR_num[i] #away 0

        X[i*numOfWord:(i+1)*numOfWord,:] = initia_llr
        Y[i*numOfWord:(i+1)*numOfWord,:] = np.reshape(u,(-1,K))
    return X, Y

def checkCRC(target):
    return (np.sum(np.mod(np.matmul(target,np.transpose(PCM)),2),1)==0)

def FocalLoss(target, output):
    gamma = 5
    alpha = 0.3
    output = tfb.clip(output, tfb.epsilon(), 1-tfb.epsilon())
    value = -alpha*target*tf.log(output+tfb.epsilon())*tf.pow(1-output,gamma)-(1-alpha)*(1-target)*tf.log(1-output+tfb.epsilon())*tf.pow(output,gamma)
    return tf.reduce_mean(value)

def weighted_binary_crossentropy(target, output):
    """
    Weighted binary crossentropy between an output tensor 
    and a target tensor. POS_WEIGHT is used as a multiplier 
    for the positive targets.

    Combination of the following functions:
    * keras.losses.binary_crossentropy
    * keras.backend.tensorflow_backend.binary_crossentropy
    * tf.nn.weighted_cross_entropy_with_logits
    """
    POS_WEIGHT = -50  # multiplier for positive targets, needs to be tuned
    # transform back to logits
    _epsilon = tfb._to_tensor(tfb.epsilon(), output.dtype.base_dtype)
    output = tf.clip_by_value(output, _epsilon, 1 - _epsilon)
    output = tf.log(output / (1 - output))
    # compute weighted loss
    loss = tf.nn.weighted_cross_entropy_with_logits(targets=target,
                                                    logits=output,
                                                    pos_weight=POS_WEIGHT)
    return tf.reduce_mean(loss, axis=-1)

def weighted_BCE(y_true, y_pred):
    # scale predictions so that the class probas of each sample sum to 1
#     weights = tfb.variable(1/np.array([0.07050923, 0.24034695, 0.19802742, 0.09862899, 0.16046447, 0.08317012, 0.10002798, 0.04882485]))
    weights = tfb.variable(np.array([1,1,1,1,1,1,1,1]))
    y_pred /= tfb.sum(y_pred, axis=-1, keepdims=True)
    # clip to prevent NaN's and Inf's
    y_pred = tfb.clip(y_pred, tfb.epsilon(), 1 - tfb.epsilon())
    # calc
    loss = y_true * tfb.log(y_pred) * weights
    loss = -tfb.sum(loss, -1)
    return loss

def find_CS(FZlookup_new,indices):
    rank=np.zeros(N,dtype=int)
    rank[FZlookup_new==-1]=1

    for i in range (n):
        for j in range (2**(n-i-1)):
            if rank[j*2**(i+1)]==1 and rank[j*2**(i+1)+2**i]==1:
                rank[j*2**(i+1)+2**i]=0
    CS = np.where(rank==1)[0]
    sort = CS.copy()
    for i in range(len(CS)):
        sort[i] = np.where(indices == CS[i])[0]
    idx = np.argsort(sort)[::-1]
    return CS[idx]

def construct_CS_mat(t): #  t is the number of bits to flip
    CS = find_CS(FZlookup,indices)
    CS_mat=np.zeros((len(CS),t),dtype=int)
    CS_mat[:,0] = CS
    for i in range(len(CS)):
        j = 1
        FZlookup_new = FZlookup.copy()
        FZlookup_new[CS[i]] = 0
        CS_new = find_CS(FZlookup_new,indices)
        while(j < t):
            CS_mat[i,j] = CS_new[0]
            FZlookup_new[CS_new[0]] = 0
            CS_new = find_CS(FZlookup_new,indices)
            j = j+1    
    return CS_mat

Y2K = np.cumsum(-1*FZlookup)-1
CS = Y2K[construct_CS_mat(1)]