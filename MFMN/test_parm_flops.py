from PIL import Image
import os
import numpy as np
import torch

from NEW_ARCH import Encoder
from SegDataset import TestData
from fvcore.nn import FlopCountAnalysis, parameter_count_table


model = Encoder()

tensor = (torch.rand(1,3,256,256),)

flops = FlopCountAnalysis(model,tensor)

print("FLOPS:",flops.total())
print("param:",parameter_count_table(model))


