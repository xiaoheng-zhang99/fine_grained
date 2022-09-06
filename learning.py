'''
import torch
batch_size=64
input_dim=300
hidden_size=256
model = torch.nn.LSTM(input_size = 300, hidden_size = 256, num_layers = 1)
input = torch.randn(1000, batch_size, input_dim)
h_0 = torch.randn(1, batch_size, hidden_size)
c_0  = torch.randn(1, batch_size, hidden_size)
output, (h, c) = model(input,(h_0, c_0))
print(output.shape)
'''
import time
from tqdm import tqdm, trange

for i in trange(100):
    time.sleep(0.5)

for i in tqdm(range(100), desc='Processing'):
    time.sleep(0.05)

dic = ['a', 'b', 'c', 'd', 'e']
pbar = tqdm(dic)
for i in pbar:
    pbar.set_description('Processing '+i)
    time.sleep(0.2)
