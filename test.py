import torch
from transformers import BertModel,BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
sentences=["i guess we don't need","excuse me"]
inputs = tokenizer(sentences, padding=True, return_tensors='pt')
#print(inputs)
model = BertModel.from_pretrained('bert-base-uncased')
outputs = model(**inputs)
features = outputs[0][:, 0, :].detach().numpy()
print(outputs[0].shape)
print(outputs[0][:, 0, :].shape)
print("features",features.shape)

input_ids = tokenizer(["i guess we don't need"], return_tensors='pt')  # Batch size 1
input_ids2 = torch.tensor(tokenizer.encode("i guess we don't need", add_special_tokens=True))
input_ids3=tokenizer.encode("i guess we don't need", add_special_tokens=True)
print(input_ids,input_ids["token_type_ids"],input_ids2,input_ids3)

model = BertModel.from_pretrained('bert-base-uncased')

outputs = model(**input_ids)
#encoded_layers, _ = model(input_ids2)
#print("try",encoded_layers)

last_hidden_states = outputs.last_hidden_state
print('last_hidden_states:' ,last_hidden_states.shape)
pooler_output = outputs.pooler_output
print('---pooler_output: ', pooler_output.shape)
'''
from my_model2 import tsc


model=tsc()
input=torch.rand([300,156])
out=model.semantic_cnn1(input)
out=model.semantic_cnn2(out)
out=model.semantic_cnn3(out)
print(out.shape)

audio_input=torch.randn(500,16,40)
h_0 = torch.randn(1,16,768)
c_0  = torch.randn(1, 16,768)
audio_out=model.acoustic_lstm(audio_input)
print(audio_out[0].shape)

'''