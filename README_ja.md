# QFTSampler (Quantum Fourier Transform Sampler)
====

## 概要
IntenF/QFTSampler - 量子フーリエ変換を利用したサンプラー．
量子フーリエ変換(QFT)を古典コンピュータで高速にシミュレートしサンプラーとして応用します．

## 説明
QFTSamplerは任意のターゲット分布を機械学習によって学習しQFTを使って高速にサンプリングします．
サンプリングされた乱数は必ずしもターゲット分布と一致しませんが，メトロポリス・ヘイスティング(MH法)によって適切に棄却/採択することでターゲット分布と一致する分布からサンプリングできます．

## デモ

```python
from QFTSampler import Orchestrator
from QFTSampler.transformers import AffineNonLinearBasis, Constant
from QFTSampler.ExpTargetDists import Target_gauss2d_dependent

import matplotlib.pyplot as plt

N = 10 #QFT回路の量子ビット数.乱数の１次元あたりに使うビット数を表す
M = 4 #QFT回路の非ゼロ量子ビット数. 作成する分布の周波数を2^Mまでに制限する

# ターゲット分布を生成
target = Target_gauss2d_dependent(N=N, M=M, )

# ターゲット分布を学習する機械学習器を生成
# ex) Constant: 定数φを学習する
# ex) AffineNonLinearBasis: φを非線形基底を用いて出力する線形関数を学習する
transformer_list = [Constant(M), AffineNonLinearBasis(N,M)]

# 学習しつつサンプリングするオーケストレーターを作成
orch = Orchestrator(N=N, M=M, trasnsformer_list=transformer_list, target=target)

# 学習しつつサンプリング（この例ではサンプルはすべて捨てる）
for _ in range(1000):
  # CE=Cross Entropy
  probs, samples = orch.step(sample_num=32, lr=1e-2, train=True, loss_func='CE')
  #samples: サンプルされた点
  #probs: samplesをサンプルする確率

# 学習した分布を可視化
pmap = orch.pmap() #ターゲット分布
qmap = orch.qmap() #qftsamplerによる提案分布
plt.title('target dist')
plt.imshow(pmap)
plt.show()
plt.title('qft dist')
plt.imshow(qmap)
plt.show()
```
**outputs**

![target dist](image/exp_target_dist.png)

![qft dist](image/exp_qft_dist.png)

## 依存環境
- Python 3.7
- numpy

### option
- scipy(for expample target distribution (ANPAN))
- ot(to calculate Wasserstein Distance)
- matplotlib(for visualization)
- tqdm(for process visualization)

## 使い方
1. あなたの作りたい分布をOrchestrator用にコードを書きます．この際，QFTSampler.ExpTargetDistsにある例を参考にしてください．
1. 機械学習に使う関数のコードを書きます．基本的な機械学習器（線形学習機:Affine, 非線形基底線形学習器:AffineNonLinearBasis, 全結合層NN:Densenet）はQFTSampler.transformers内に定義されています．自分で新たに作る際はAffineなどを参考にしてください．機械学習器はロス関数のそれぞれの入力に対する１次微分値から学習できる必要があります．
1. transformer_list[i]は1~(i-1)次元までのサンプル値を入力に取り，i次元目の乱数を生成する量子状態φ_iを生成します．
1. Orchestrator.stepを使ってサンプリングと学習を行います．


## インストール方法

```bash
git clone git@github.com:IntenF/QFTSampler.git
```

## Licence
This software is released under the MIT License, see LICENSE.txt.

## Author
- Katsuhiro Endo([umu1729](https://github.com/umu1729))
- Taichi Nakamura([IntenF](https://github.com/IntenF))

This READEME_ja.md was wirtten by reference to this [page](https://deeeet.com/writing/2014/07/31/readme/)
