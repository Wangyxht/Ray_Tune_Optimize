# *example*的实验结果

实验参数：

利用随机搜索、贝叶斯搜索（GP/TPE)、ASHA搜索、PBT搜索对两层全连接神经网络进行优化：

神经网络结构定义如下：

```python
class MinstNet(nn.Module):
    """
    MINST全连接网络结构
    :param l1: 第一个隐藏层的节点数
    :param l2: 第二个隐藏层的节点数
    """
    def __init__(self, l1: int, l2: int):
        super(MinstNet, self).__init__()
        self.layer1 = nn.Linear(in_features=28 * 28, out_features=l1)
        self.layer2 = nn.Linear(in_features=l1, out_features=l2)
        self.layer3 = nn.Linear(in_features=l2, out_features=10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        return x
```

+ 数据集：MINST手写数字图片
+ 损失函数：交叉熵
+ 超参数搜索空间：


| 超参数 | 描述               | 取值范围              |
| ------ | ------------------ | --------------------- |
| *l1*   | 第一隐藏层节点个数 | [512, 256, 128]       |
| *l2*   | 第二隐藏层节点个数 | [64, 32, 16]          |
| *lr*   | 学习率（SGD）      | (0.001, 0.1) 对数分布 |

+ 超参数搜索空间定义：

```python
    config = {
        "l1": tune.choice([512, 256, 128]),
        "l2": tune.choice([64, 32, 16]),
        "lr": tune.loguniform(0.001, 0.1),
    }
```

## *简单随机搜索*（*example_Random_optimize.py*）：

```plaintext

Trial status: 20 TERMINATED
Current time: 2024-08-16 14:22:30. Total running time: 10min 14s
Logical resource usage: 1.0/16 CPUs, 0.2/1 GPUs (0.0/1.0 accelerator_type:G)
╭────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ Trial name                status         l1     l2           lr     iter     total time (s)     test_accuracy     train_loss     test_loss │
├────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ train_minst_7c89e_00000   TERMINATED    256     32   0.0193047        10            128.361            0.9652      0.0958442     0.112506  │
│ train_minst_7c89e_00001   TERMINATED    512     32   0.00119414       10            129.5              0.8929      0.405533      0.381277  │
│ train_minst_7c89e_00002   TERMINATED    128     32   0.0180713        10            129.835            0.9638      0.114451      0.11565   │
│ train_minst_7c89e_00003   TERMINATED    512     32   0.00180891       10            129.868            0.9002      0.3566        0.338372  │
│ train_minst_7c89e_00004   TERMINATED    256     32   0.013255         10            129.499            0.9573      0.139301      0.147423  │
│ train_minst_7c89e_00005   TERMINATED    128     32   0.0206445        10            144.358            0.9655      0.105118      0.117841  │
│ train_minst_7c89e_00006   TERMINATED    128     16   0.0180847        10            144.047            0.9641      0.112206      0.119821  │
│ train_minst_7c89e_00007   TERMINATED    256     16   0.00350126       10            144.783            0.9163      0.297392      0.287994  │
│ train_minst_7c89e_00008   TERMINATED    128     64   0.0135571        10            144.65             0.9544      0.149689      0.151458  │
│ train_minst_7c89e_00009   TERMINATED    128     64   0.0818134        10            144.066            0.9626      0.0488834     0.130452  │
│ train_minst_7c89e_00010   TERMINATED    512     32   0.00130585       10            144.851            0.8981      0.387623      0.363246  │
│ train_minst_7c89e_00011   TERMINATED    256     64   0.00538483       10            150.284            0.9257      0.250689      0.251121  │
│ train_minst_7c89e_00012   TERMINATED    256     16   0.00207117       10            146.364            0.9073      0.343772      0.325679  │
│ train_minst_7c89e_00013   TERMINATED    128     32   0.0119489        10            145.452            0.9537      0.157515      0.159192  │
│ train_minst_7c89e_00014   TERMINATED    128     64   0.0470626        10            145.557            0.9684      0.0619149     0.104423  │
│ train_minst_7c89e_00015   TERMINATED    512     64   0.0743762        10            152.72             0.9777      0.0356238     0.0740307 │
│ train_minst_7c89e_00016   TERMINATED    512     16   0.00772462       10            152.25             0.9431      0.189805      0.190497  │
│ train_minst_7c89e_00017   TERMINATED    256     64   0.0463242        10            152.359            0.9747      0.0515979     0.0817458 │
│ train_minst_7c89e_00018   TERMINATED    128     32   0.002113         10            152.233            0.9011      0.349749      0.333834  │
│ train_minst_7c89e_00019   TERMINATED    512     16   0.00640066       10            151.555            0.9407      0.225155      0.210269  │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯

best hyperparameter information:
 {'l1': 512, 'l2': 64, 'lr': 0.07437623377678602}

```

+ 最终准确率： `0.9777`
+ 测试集LOSS： `0.074`
+ 用时：`10min 14s`

## *异步连续减半搜索（example_ASHA_optimize.py）*：

```plaintext
Trial status: 20 TERMINATED
Current time: 2024-08-16 14:37:18. Total running time: 6min 16s
Logical resource usage: 1.0/16 CPUs, 0.2/1 GPUs (0.0/1.0 accelerator_type:G)
╭────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ Trial name                status         l1     l2           lr     iter     total time (s)     test_accuracy     train_loss     test_loss │
├────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ train_minst_1bd89_00000   TERMINATED    256     32   0.034776         10           142.118             0.9706      0.0644087     0.0889516 │
│ train_minst_1bd89_00001   TERMINATED    256     64   0.00273314       10           140.923             0.9156      0.310914      0.295498  │
│ train_minst_1bd89_00002   TERMINATED    512     16   0.00281498        2            26.568             0.8353      0.997431      0.697917  │
│ train_minst_1bd89_00003   TERMINATED    512     64   0.00298333        2            26.4033            0.8669      0.735083      0.529131  │
│ train_minst_1bd89_00004   TERMINATED    512     64   0.0155187        10           141.377             0.9628      0.115642      0.120193  │
│ train_minst_1bd89_00005   TERMINATED    128     64   0.0062116         4            56.3177            0.9089      0.335851      0.316043  │
│ train_minst_1bd89_00006   TERMINATED    256     64   0.00128946        2            27.3026            0.7474      1.6625        1.2976    │
│ train_minst_1bd89_00007   TERMINATED    256     16   0.0825923        10           149.691             0.9736      0.0466266     0.0831178 │
│ train_minst_1bd89_00008   TERMINATED    128     64   0.016168          8           119.532             0.9542      0.147299      0.148854  │
│ train_minst_1bd89_00009   TERMINATED    256     64   0.00251958        2            30.4617            0.8274      0.992501      0.689433  │
│ train_minst_1bd89_00010   TERMINATED    512     32   0.00337881        2            30.4697            0.8617      0.698791      0.517359  │
│ train_minst_1bd89_00011   TERMINATED    512     64   0.0388737        10           150.14              0.9786      0.0520705     0.0685885 │
│ train_minst_1bd89_00012   TERMINATED    512     32   0.039565          8           118.331             0.9569      0.067142      0.139263  │
│ train_minst_1bd89_00013   TERMINATED    256     16   0.00125166        2            29.9322            0.3982      1.92294       1.75318   │
│ train_minst_1bd89_00014   TERMINATED    128     64   0.00823507        4            59.4304            0.9196      0.306805      0.28122   │
│ train_minst_1bd89_00015   TERMINATED    512     32   0.0212398        10           135.508             0.969       0.0878323     0.102333  │
│ train_minst_1bd89_00016   TERMINATED    256     64   0.00520168        2            29.9047            0.8889      0.489077      0.391935  │
│ train_minst_1bd89_00017   TERMINATED    512     16   0.00287425        2            30.341             0.8361      0.923896      0.649984  │
│ train_minst_1bd89_00018   TERMINATED    128     16   0.0011968         2            26.0224            0.6917      1.63729       1.33192   │
│ train_minst_1bd89_00019   TERMINATED    512     16   0.00478939        2            25.0929            0.8767      0.607532      0.465485  │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯

2024-08-16 14:37:18,996	INFO tune.py:1009 -- Wrote the latest version of all result files and experiment state to 'C:/Users/Wangyx/ray_results/train_minst_2024-08-16_14-30-54' in 0.0391s.
best hyperparameter information:
 {'l1': 512, 'l2': 64, 'lr': 0.03887370457508451}
```

+ 最终准确率： `0.9786`
+ 测试集LOSS： `0.052`
+ 用时：`6min 16s`

## *贝叶斯优化，引入ASHA机制（example_ASHA_optimize.py)*

```plaintext
Trial status: 20 TERMINATED
Current time: 2024-08-16 14:47:45. Total running time: 7min 41s
Logical resource usage: 1.0/16 CPUs, 0.2/1 GPUs (0.0/1.0 accelerator_type:G)
╭─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ Trial name             status         l1     l2           lr     iter     total time (s)     test_accuracy     train_loss     test_loss │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ train_minst_c8001b85   TERMINATED    128     64   0.00224569       10           133.096             0.9082      0.330252      0.313212  │
│ train_minst_5f81d8d8   TERMINATED    512     32   0.00260221       10           135.361             0.9135      0.314786      0.300743  │
│ train_minst_e6bf24e5   TERMINATED    512     64   0.0138081        10           138.18              0.963       0.12385       0.124442  │
│ train_minst_dec80b33   TERMINATED    256     16   0.0216335        10           138.338             0.9679      0.0954869     0.0983496 │
│ train_minst_5ee3ef53   TERMINATED    256     64   0.00419518        4            52.5428            0.9027      0.368717      0.338081  │
│ train_minst_03201c75   TERMINATED    128     16   0.00379042        2            27.9333            0.8511      0.847677      0.578     │
│ train_minst_958a44fa   TERMINATED    512     32   0.0161449        10           147.946             0.967       0.107109      0.114158  │
│ train_minst_4f261d95   TERMINATED    512     64   0.056343          8           118.446             0.9407      0.0555161     0.180531  │
│ train_minst_fa5823a5   TERMINATED    128     32   0.00208515        2            30.4618            0.7582      1.45169       1.04316   │
│ train_minst_21b212e8   TERMINATED    512     64   0.0227423        10           146.827             0.9728      0.079799      0.0885438 │
│ train_minst_251c65ae   TERMINATED    256     64   0.00291255        2            28.8115            0.8543      0.78949       0.570442  │
│ train_minst_bdfe2d16   TERMINATED    128     16   0.0150029         4            58.4934            0.9301      0.26229       0.243378  │
│ train_minst_6d109ada   TERMINATED    512     16   0.00122285        2            29.3673            0.6334      1.75699       1.45936   │
│ train_minst_91b0fbab   TERMINATED    128     16   0.0172254         4            58.4488            0.9305      0.253327      0.235054  │
│ train_minst_757a910d   TERMINATED    128     32   0.050297         10           144.874             0.9693      0.0610662     0.0965149 │
│ train_minst_82ee0d04   TERMINATED    256     16   0.0704608         2            28.8837            0.8925      0.199619      0.329365  │
│ train_minst_264004d1   TERMINATED    256     16   0.0212148        10           142.425             0.9657      0.0912939     0.110728  │
│ train_minst_765d5464   TERMINATED    256     32   0.0359193        10           139.566             0.9709      0.0599807     0.0916268 │
│ train_minst_48c9789b   TERMINATED    256     64   0.038961         10           138.552             0.9764      0.0594157     0.0734329 │
│ train_minst_c25c370b   TERMINATED    256     64   0.032086          8           112.896             0.958       0.0866121     0.13739   │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯

best hyperparameter information:
 {'l1': 256, 'l2': 64, 'lr': 0.03896099972638312}
```
+ 最终准确率： `0.9764`
+ 测试集LOSS： `0.073`
+ 用时：`7min 41s`

## *PBT（example_PBT_optimize.py)*
```text
Trial status: 20 TERMINATED
Current time: 2024-08-16 15:18:10. Total running time: 15min 5s
Logical resource usage: 1.0/16 CPUs, 0.2/1 GPUs (0.0/1.0 accelerator_type:G)
╭────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ Trial name                                status         l1     l2           lr     iter     total time (s)     test_accuracy     train_loss     test_loss │
├────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ train_minst_with_checkpoint_95e5d_00000   TERMINATED    512     32   0.0546228        10            146.962            0.9467       0.180619      0.179693 │
│ train_minst_with_checkpoint_95e5d_00001   TERMINATED    512     32   0.045519         10            145.998            0.9467       0.180547      0.179693 │
│ train_minst_with_checkpoint_95e5d_00002   TERMINATED    512     32   0.0364152        10            146.188            0.9467       0.180527      0.179693 │
│ train_minst_with_checkpoint_95e5d_00003   TERMINATED    512     64   0.0609883        10            148.007            0.9606       0.128087      0.134229 │
│ train_minst_with_checkpoint_95e5d_00004   TERMINATED    512     32   0.0132889        10            144.911            0.9467       0.180498      0.179693 │
│ train_minst_with_checkpoint_95e5d_00005   TERMINATED    512     32   0.0166111        10            145.898            0.9467       0.180611      0.179693 │
│ train_minst_with_checkpoint_95e5d_00006   TERMINATED    256     64   0.0141052        10            146.56             0.9097       0.313267      0.303871 │
│ train_minst_with_checkpoint_95e5d_00007   TERMINATED    512     32   0.0197015        10            146.141            0.9467       0.180501      0.179693 │
│ train_minst_with_checkpoint_95e5d_00008   TERMINATED    256     64   0.0232205        10            145.763            0.9218       0.268929      0.260765 │
│ train_minst_with_checkpoint_95e5d_00009   TERMINATED    512     32   0.0159467        10            144.75             0.9467       0.180513      0.179693 │
│ train_minst_with_checkpoint_95e5d_00010   TERMINATED    512     32   0.0132889        10            145.742            0.9467       0.180531      0.179693 │
│ train_minst_with_checkpoint_95e5d_00011   TERMINATED    256     64   0.0185764        10            147.573            0.9218       0.269043      0.260765 │
│ train_minst_with_checkpoint_95e5d_00012   TERMINATED    512     64   0.0762353        10            148.352            0.9606       0.128081      0.134229 │
│ train_minst_with_checkpoint_95e5d_00013   TERMINATED    512     32   0.0546228        10            145.525            0.9467       0.180536      0.179693 │
│ train_minst_with_checkpoint_95e5d_00014   TERMINATED    256     64   0.0222917        10            146.116            0.9218       0.268994      0.260765 │
│ train_minst_with_checkpoint_95e5d_00015   TERMINATED    128     64   0.0283795        10            149.771            0.9326       0.234647      0.233139 │
│ train_minst_with_checkpoint_95e5d_00016   TERMINATED    512     64   0.00864878       10            147.407            0.9606       0.128112      0.134229 │
│ train_minst_with_checkpoint_95e5d_00017   TERMINATED    512     64   0.0731859        10            147.396            0.9606       0.128054      0.134229 │
│ train_minst_with_checkpoint_95e5d_00018   TERMINATED    128     64   0.0272443        10            149.765            0.9326       0.234688      0.233139 │
│ train_minst_with_checkpoint_95e5d_00019   TERMINATED    128     64   0.0340554        10            147.044            0.9326       0.234582      0.233139 │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
2024-08-16 15:18:10,999	INFO tune.py:1009 -- Wrote the latest version of all result files and experiment state to 'C:/Users/Wangyx/ray_results/train' in 0.0470s.

best hyperparameter information:
 {'l1': 512, 'l2': 64, 'lr': 0.06098827356230923}
```
+ 最终准确率： `0.9606`
+ 测试集LOSS： `0.13`
+ 用时：`15min 5s`
