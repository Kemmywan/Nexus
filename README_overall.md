# Net Detector

## Implement


## Prerequisites

```
conda create --name provenance python=3.12
pip install pandas
pip install tqdm
pip install git+https://github.com/casics/nostril.git
pip install gensim
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install matplotlib
pip install seaborn
pip install streamz
pip install schedule
pip install nearpy
pip install pydot
pip install graphviz
...
```

## Implement
Detect net data anomaly. 

## Experiments

---

# 实验一：NODLINK数据集检测异常节点

## Part1: data preprocess

* 原始日志：NODLINK SimulatedUbuntu json格式，json文件中每行为dict数据，存储一条日志

  ```
  {"evt.args":"res=0 ","evt.num":63,"evt.time":1648196412041127565,"evt.type":"fstat","fd.name":"/proc/296059/cmdline","proc.cmdline":"node /root/.vscode-server/bin/ccbaa2d27e38e5afa3e5c21c1c7bef4657064247/out/bootstrap-fork --type=ptyHost","proc.name":"node","proc.pcmdline":"node /root/.vscode-server/bin/ccbaa2d27e38e5afa3e5c21c1c7bef4657064247/out/vs/server/main.js --start-server --host=127.0.0.1 --enable-remote-auto-shutdown --port=0 --connection-secret /root/.vscode-server/.ccbaa2d27e38e5afa3e5c21c1c7bef4657064247.token","proc.pname":"node"}
  ```

* 日志读取，转换为二维表格数据

* 将日志划分为file、process、network三类

  ```
  Benign:
  Reading logs: 15342095line [01:20, 189533.21line/s]
  Finish log parsing.
  total logs count: 15342095
  file logs count: 12201224
  process logs count: 18937
  net logs count: 40169
  
  anomaly
  Reading logs: 3832342line [00:34, 110158.89line/s]
  Finish log parsing.
  total logs count: 3832342
  file logs count: 1766913
  process logs count: 8823
  net logs count: 5947
  ```

## Part2：embedding

* 语料库提取，词向量嵌入模型训练

## Part3：train

* VAE训练

* 阈值选取

* VAE测试

  ```
  90:  0.6253968477249161
  80:  0.23421537280082705
  70:  0.1419504702091217
  60:  0.10555592775344848
  anomaly threshold:  1.5781963348388655
  90:  1.717025756835938
  80:  0.47042333483695986
  70:  0.1690540730953216
  60:  0.1126775860786438
  ground-truth id:  [1062, 1063, 1064]
  1062 50.460716247558594
  1063 14.18455696105957
  1064 29.408004760742188
  detected anomaly: 634/5947
  recall:  1.0
  precision:  0.00473186119873817
  ground truth: 3
  detected process: 634
  ```

  # 实验二：ATLAS系统日志攻击检测

* NODLINK 的方法是VAE学习良性日志的潜在空间分布，要求数据分为benign和anomaly
* ATLAS数据集只提供了整个时间段的所有日志，视作anomaly；需要提取benign数据集

## 整体思路

用ATLAS方法构建来源图，用NODLINK方法生成嵌入向量和后续模型训练

## 溯源图构建

* 用ATLAS方法三类日志统一格式合并，直接使用ATLAS预处理后的日志数据
* 从三个训练集中提取benign日志
* 构建溯源图（问题：网络日志是否能与系统日志有直接因果关联；回答：可以，在溯源图构建过程中，一条日志被转换为两个相邻结点和一条边，结点为进程、IP等类型）
  * 实际只用到系统日志
  * idea：在溯源图构建时，以实体为节点。具体来讲，即使同一实体在数据集中前后出现多次，在溯源图中只会将不同时间出现的同一实体视作同一节点。换句话讲，在溯源图中损失了实体的时间信息，是否可以保留时间信息？相比目前的溯源图有什么表现变化？

* 从溯源图中提取进程结点相关的事件
  * 溯源图中没有时间概念，我认为可以将三个来自不同场景的良性数据集处理得到的事件合并
  * atlas中主进程进程名为'-'，pid为0，这样的日志在构建成事件时是单个'-'，没法计算权重，手动去除


## 特征提取

* 用NODLINK思路，提取系统日志特征
  * FastText 词嵌入模型训练
  * idea：词嵌入语料库训练是否可以使用anomaly数据
* VAE模型训练
  * idea：思考VAE模型的阈值的选取方式，是在训练集上按比例选取还是在测试集上选取
  * 设定阈值
  * 异常事件检测

```
numbers of anomaly connected subgraphs: 1
[anomaly] graph score:  2164.369573851869 nodes number:  264
node-level recall: 8/9 =  0.8888888888888888
{'SPECIALPROCESSNAME_0'}
node-level precision: 3/23 =  0.13043478260869565
```

# 实验三：ATLAS多类型日志攻击检测

## 思路简述

1. 节点合并：DNS和http日志合并到系统日志中的IP节点

## 代码框架

1. `preprocess.py`：构建溯源图，提取语料库
   * 溯源图：
     * 从系统日志中提取 PROCESS, FILE, IP 类型节点
     * 从DNS日志中提取 DOMAIN, IP 类型节点
     * 从http日志中提取DOMAIN, WEB 类型节点
   * 特征聚合：以PROCESS类型节点为中心节点，聚合一跳邻域内非PROCESS节点，遇到IP类型节点再拓展一跳邻域
2. `embedding.py`：FastText词向量模型训练，计算权重相关信息
3. `train.py`：训练VAE模型
4. `main.py`：攻击调查

## 结果分析

> 改变日志类型：注释 preprocess.py 中construct_G 函数中相关代码

为了验证多类型日志对攻击检测分析效果的影响，设置对比实验：其它条件不变，对比不同日志类型。

更多日志类型：

* 坏处:
  * 溯源图节点更多更复杂，得到的攻击子图更大，需设计优化方式



### 仅系统日志



### 系统日志、dns日志

# 实验四：基于OpTC数据集的多类型异常探索

**Issue**：VAE，训练分布和测试分布差异大

```
90:  3.117022548003088
80:  0.8709974560235537
70:  0.2429403235386499
60:  0.13566986005842224

90:  47.250989923719644
80:  22.475632030884338
70:  17.852270935725063
60:  12.77303013330158
```

**Explanation**：训练数据和测试数据的主机不同

---

**Issue**：训练数据的重构得分分布集中，可能存在过拟合、泛化性差的问题

![image-20250224112513455](./img/image-20250224112513455.png)

**Try**：

1. 增加训练数据，效果不理想

```
train-test: 9-1
recall:  0.5
precision:  0.007952286282306162
detected ground truth: 4
detected alert:  503
ground truth:  8


```



![image-20250224190508153](./img/image-20250224190508153.png)

---

**Issue**：训练数据重构损失分布与测试数据存在较大偏差，测试数据整体偏大

```
25
90:  12.875549811489467
80:  6.653049253869719
70:  4.122568085216533
60:  2.807351592460327
90:  54.978127620964095
80:  19.084825668332677
70:  9.449615343010315
60:  4.230690141870335

100
90:  6.562101687751883
80:  3.635442502474958
70:  2.1248899411223032
60:  1.306145202203221
90:  42.56413351529371
80:  12.515374477179801
70:  4.9065186927822015
60:  1.9391330380566765

200
90:  5.621111127907055
80:  3.1176201992082335
70:  1.752222905637743
60:  1.0126503390952486
90:  38.35693985385854
80:  11.912639081461045
70:  4.476041515707172
60:  1.7490242236531583

400
90:  5.118955755007832
80:  3.018212959906038
70:  1.744670960606268
60:  0.9633366973030368
90:  38.25704533359275
80:  10.499070782067367
70:  4.216189425107294
60:  1.6064156577903543
```

**Thinking**：

* 随着训练收敛，在训练集上重构损失逐渐变小，符合直觉
* 随着训练收敛，在测试集上重构损失逐渐变小，说明模型的更接近正常数据空间分布
* 随着训练收敛程度增大，继续训练，训练集和测试集上重构损失分布几乎不变，此时train和test的分布仍不够理想