from Orchestrator import *

#可視化用
from tqdm import tqdm

def selfMH(target, probed, sampled, verbose=False ):
    '''
    self Metropolis-Hastings:自己メトロポリスヘイスティング法
    globalサンプリングなMHサンプリング
    sampledは提案分布のサンプル値の配列
    targetは目標分布で関数当てる、probedはsampledの取りうる確率の配列を格納する。
    '''
    x = sampled[0].reshape(-1, 1)
    ind = 0
    sampled = sampled[1:]
    x_list = []
    for i, s in tqdm(enumerate(sampled), total=len(sampled), disable=not verbose):
        next_x = s.reshape(-1, 1)
        next_ind = i+1
        r = target(*next_x)*probed[ind]/target(*x)/probed[next_ind]
        if r>np.random.rand():
            x = next_x
            ind = next_ind
            x_list.append(x)
    return x_list

def calc_loss(target, prob, sample, epsilon=1e-7):
    loss = []
    for s,q in zip(sample, prob):
        loss.append(-target(*s.reshape(-1, 1))/(q+epsilon)*np.log(q+epsilon))
    return np.mean(loss)

def sample(orch, iter, sample_num, lr=1, filter_func=selfMH, train=True, filter=True, alpha=0.01, verbose=True, loss_func='CE'):
    sample_list = []
    loss_list = []
    sloss = 0
    snum = 0
    with tqdm(range(iter), disable=not verbose) as pbar:
        for i in pbar:
            q, sample = orch.step(lr=lr, sample_num=sample_num, train=True, loss_func=loss_func)
            fsample = selfMH(target=orch.target, probed=q, sampled=sample, ) if filter else sample
            sample_list.append(fsample)
            loss = calc_loss(orch.target, q, sample, )
            sloss = alpha*loss + (1-alpha)*loss if i != 0 else loss
            snum = alpha*len(fsample)+ (1-alpha)*snum if i != 0 else len(fsample)
            pbar.set_postfix(accept_num=snum, loss=sloss)
            loss_list.append(loss)
    return sample_list, loss_list
