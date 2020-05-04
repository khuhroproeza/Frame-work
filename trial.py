""""
import torch

a = torch.FloatTensor(3, 2)

v1 = torch.tensor([1.0,1.0], requires_grad = True)
v2 = torch.tensor([2.0,1.0])

v_sum = v1+ v2
v_res = (v_sum*2).sum()

print(v_res.backward())
print(v1.grad)

import pandas as pd
filer = pd.read_csv('/home/khuhroproeza/FrameWork/Datasets/HPC2Performance.csv'
                    '')
from arch.bootstrap import IndependentSamplesBootstrap


def mean_diff(x, y):
    return x.mean() - y.mean()
import numpy as np

rs = np.random.RandomState(0)
treatment = 0.2 + rs.standard_normal(200)
control = rs.standard_normal(800)

bs = IndependentSamplesBootstrap(filer, random_state=rs)
print(type(bs))

"""


i = randint(1,100000)