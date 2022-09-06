import re
import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import time
import yaml
import pickle
import argparse
from transformers import BertModel,BertTokenizer
import numpy as np
from tqdm import tqdm
from functools import reduce
import torch
import torchaudio
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from models import NeoMeanMaxExcite_v2
from my_model2 import my_model2
import warnings
warnings.filterwarnings("ignore")

def evaluate_metrics(pred_label, true_label):
    pred_label = np.array(pred_label)
    true_label = np.array(true_label)
    wa = np.mean(pred_label.astype(int) == true_label.astype(int))
    pred_onehot = np.eye(4)[pred_label.astype(int)]
    true_onehot = np.eye(4)[true_label.astype(int)]
    ua = np.mean(np.sum((pred_onehot == true_onehot) * true_onehot, axis=0) / np.sum(true_onehot, axis=0))
    key_metric, report_metric = 0.9 * wa + 0.1 * ua, {'wa': wa, 'ua': ua}
    return key_metric, report_metric


class IEMOCAPDataset(object):
    def __init__(self, config, data_list):
        self.data_list = data_list
        self.vocabulary_dict = pickle.load(open("E:\data\iemocap\glove300d_w2i.pkl", 'rb'))
        self.audio_length = 3000
        self.feature_name ='fbank'
        self.feature_dim = 40

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        audio_path, asr_text, align_path, label = self.data_list[index]
        audio_name = os.path.basename(audio_path)
        # ------------- extract the audio features -------------#
        waveform, sample_rate = torchaudio.load(audio_path)

        audio_input = torchaudio.compliance.kaldi.fbank(
                waveform, sample_frequency=sample_rate, num_mel_bins=self.feature_dim,
                frame_length=25, frame_shift=10, use_log_fbank=True)
        audio_input = audio_input[:self.audio_length, :]
        #print("audio_input.size",audio_input.size)
        audio_length = audio_input.size(0)

        # ------------- extract the text contexts -------------#
        # text_words = [x.lower() for x in re.split(' +', re.sub('[\.,\?\!]', ' ', asr_text))]
        # text_input = torch.LongTensor([int(self.vocabulary_dict.get(x, '-1')) for x in text_words if len(x) > 0])
        from transformers import BertModel, BertTokenizer
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        text_input_model = tokenizer(asr_text, return_tensors='pt')
        text_input = torch.tensor(tokenizer.encode(asr_text, add_special_tokens=True))
        # Here we use the 0 to represent the padding tokens
        # text_input = text_input + 1
        # text_length = text_input.size(0)
        text_length = len(tokenizer.encode(asr_text, add_special_tokens=True))
        #print(text_input_model, text_length)

        # ------------- generate the force alignment matrix -------------#
        align_info = open(align_path, 'r').readlines()[1:-1]  # get rid off the head and tail info
        align_info = [re.split('\ +', x) for x in align_info]
        align_info = [(x[-1].strip('\n').split('(')[0].lower(), int(x[1]), int(x[2])) for x in align_info]
        # For the silence probably we can make some use
        align_info = [x for x in align_info if x[0] not in ['<s>', '<sil>', '</s>']]
        align_input = []
        for _, begin_time, end_time in align_info:
            begin_idx = int(begin_time * 0.01 / 0.01)
            end_idx = int(end_time * 0.01 / 0.01) + 1
            align_slice = torch.zeros(audio_input.size(0))
            align_slice[begin_idx:end_idx] = 1.0
            align_input.append(align_slice[None, :])
            #print(align_input)
        align_input = torch.cat(align_input, dim=0)
        # ------------- wrap up all the output info the dict format -------------#
        return {'audio_input': audio_input, 'text':asr_text,'text_input': text_input, 'text_input_model': text_input_model,'audio_length': audio_length,
                'text_length': text_length, 'align_input': align_input, 'label': label, 'audio_name': audio_name}


def collate(sample_list):

    batch_audio = [x['audio_input'] for x in sample_list]
    batch_asrtext = [x['text'] for x in sample_list]
    batch_text = [x['text_input'] for x in sample_list]
    batch_text2 = [x['text_input_model'] for x in sample_list]
    batch_audio = pad_sequence(batch_audio, batch_first=True)
    batch_text = pad_sequence(batch_text, batch_first=True)
    print(batch_asrtext)
    audio_length = torch.LongTensor([x['audio_length'] for x in sample_list])
    text_length = torch.LongTensor([x['text_length'] for x in sample_list])

    batch_label = torch.tensor([x['label'] for x in sample_list], dtype=torch.long)
    batch_name = [x['audio_name'] for x in sample_list]
    # ------------- pad for the alignment result -------------#
    batch_align = [F.pad( x['align_input'],(0, int((torch.max(audio_length) - x['align_input'].size(1)).numpy()),
         0, int((torch.max(text_length) - x['align_input'].size(0)).numpy())),
        "constant", 0 )[None, :, :] for x in sample_list]
    batch_align = torch.cat(batch_align, dim=0)
    #print("text_input_model", batch_text2)
    return ((batch_audio, audio_length), (batch_asrtext,batch_text, batch_text2, text_length), batch_align), batch_label, batch_name
def tmp_func(x):
    return collate(x)

def run(config, train_data, valid_data):
    num_workers = 0
    batch_size = 7
    epochs = 50
    learning_rate = 5e-4
    ############################## PREPARE DATASET ##########################
    train_dataset = IEMOCAPDataset(config, train_data)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=batch_size, collate_fn=tmp_func,
        shuffle=True, num_workers=num_workers
    )

    valid_dataset = IEMOCAPDataset(config, valid_data)
    valid_loader = torch.utils.data.DataLoader(
        dataset=valid_dataset, batch_size=batch_size, collate_fn=tmp_func,
        shuffle=False, num_workers=num_workers
    )

    ########################### CREATE MODEL #################################
    model = my_model2(config)
    model.to("cuda")
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = nn.CrossEntropyLoss()
    ########################### TRAINING #####################################
    count, best_metric, save_metric, best_epoch = 0, -np.inf, None, 0

    for epoch in range(epochs):
        print("epoch----training----",epoch)
        epoch_train_loss = []
        model.train()
        start_time = time.time()

        for batch_input, label_input, _ in train_loader:
            acoustic_input, acoustic_length = batch_input[0]
            acoustic_input = acoustic_input.cuda()
            acoustic_length = acoustic_length.cuda()
            asr_text,semantic_input, semantic_input_model, semantic_length = batch_input[1]
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            asr_text = tokenizer(asr_text, padding=True, return_tensors='pt')

            asr_text["input_ids"] = asr_text["input_ids"].cuda()
            asr_text["token_type_ids"] = asr_text["token_type_ids"].cuda()
            asr_text["attention_mask"] = asr_text["attention_mask"].cuda()
            semantic_input = semantic_input.cuda()
            for i in range(len(semantic_input_model)):
                semantic_input_model[i]["input_ids"] = semantic_input_model[i]["input_ids"].cuda()
                semantic_input_model[i]["token_type_ids"] = semantic_input_model[i]["token_type_ids"].cuda()
                semantic_input_model[i]["attention_mask"]=semantic_input_model[i]["attention_mask"].cuda()
            align_input = batch_input[2].cuda() #0
            label_input = label_input.cuda()
            model.zero_grad()
            logits = model(acoustic_input, acoustic_length,asr_text,
                           semantic_input, semantic_input_model,semantic_length,
                           align_input, )

            loss = loss_function(logits, label_input.long())

            epoch_train_loss.append(loss)
            loss.backward()
            optimizer.step()
            count += 1
            acc_train_loss = torch.mean(torch.tensor(epoch_train_loss)).cpu().detach().numpy()
     #testing
        print("--------valid---------")
        model.eval()
        pred_y, true_y = [], []
        with torch.no_grad():
            for batch_input, label_input, _ in valid_loader:
                acoustic_input, acoustic_length = batch_input[0]
                acoustic_input = acoustic_input.cuda()
                acoustic_length = acoustic_length.cuda()
                semantic_input, semantic_length = batch_input[1]
                semantic_input = semantic_input.cuda()
                semantic_length = semantic_length.cuda()
                align_input = batch_input[2].cuda()

                true_y.extend(list(label_input.numpy()))
                logits = model(
                    acoustic_input, acoustic_length,
                    semantic_input, semantic_length,
                    align_input,
                )
                prediction = torch.argmax(logits, axis=1)
                label_outputs = prediction.cpu().detach().numpy().astype(int)
                pred_y.extend(list(label_outputs))

        key_metric, report_metric = evaluate_metrics(pred_y, true_y)
        epoch_train_loss = torch.mean(torch.tensor(epoch_train_loss)).cpu().detach().numpy()

        print('Valid Metric: {} - Train Loss: {:.3f}'.format(
            ' - '.join(['{}: {:.3f}'.format(key, value) for key, value in report_metric.items()]),epoch_train_loss))

        if key_metric > best_metric:
            best_metric, best_epoch = key_metric, epoch
            print('Better Metric found on dev, calculate performance on Test')
            pred_y, true_y = [], []
            with torch.no_grad():
                time.sleep(2)  # avoid the deadlock during the switch between the different dataloaders
                for batch_input, label_input, _ in valid_loader:
                    acoustic_input, acoustic_length = batch_input[0]
                    acoustic_input = acoustic_input.cuda()
                    acoustic_length = acoustic_length.cuda()
                    semantic_input, semantic_length = batch_input[1]
                    semantic_input = semantic_input.cuda()
                    semantic_length = semantic_length.cuda()
                    align_input = batch_input[2].cuda()
                    #align_input = batch_input[2]

                    true_y.extend(list(label_input.numpy()))

                    logits = model(
                        acoustic_input, acoustic_length,
                        semantic_input, semantic_length,
                        align_input,
                    )

                    prediction = torch.argmax(logits, axis=1)
                    label_outputs = prediction.cpu().detach().numpy().astype(int)

                    pred_y.extend(list(label_outputs))
            _, save_metric = evaluate_metrics(pred_y, true_y)
            print("Test Metric: {}".format(
                ' - '.join(['{}: {:.3f}'.format(key, value) for key, value in save_metric.items()])
            ))

    print("End. Best epoch {:03d}: {}".format(best_epoch, ' - '.join(
        ['{}: {:.3f}'.format(key, value) for key, value in save_metric.items()])))
    c = confusion_matrix(true_y, pred_y)
    disp = ConfusionMatrixDisplay(confusion_matrix=c, display_labels=[0, 1, 2, 3])
    disp.plot()
    plt.show()
    return save_metric


if __name__ == '__main__':
    config = yaml.load(open("D:/appstorage/PyCharm Community Edition 2022.2.1/fine_grained/config.yaml", 'r'), Loader=yaml.FullLoader)
    report_result = []
    data_root = 'E:/data/iemocap/'
    #data_source = ['Session1-wav.pkl', 'Session2-wav.pkl', 'Session3-wav.pkl', 'Session4-wav.pkl', 'Session5-wav.pkl']
    data_source = ['Session1-wav.pkl', 'Session2-wav.pkl']

    for i in range(2):
        valid_path = os.path.join(data_root, data_source[i])
        valid_data = pickle.load(open(valid_path, 'rb'))
        valid_data = [(os.path.join(data_root, x[0]), x[1], os.path.join(data_root, x[2]), x[3]) for x in valid_data]
        #print(valid_data)

        train_path = [os.path.join(data_root, x) for x in data_source[:i] + data_source[i + 1:]]
        train_data = list(reduce(lambda a, b: a + b, [pickle.load(open(x, 'rb')) for x in train_path]))
        train_data = [(os.path.join(data_root, x[0]), x[1], os.path.join(data_root, x[2]), x[3]) for x in train_data]

    report_metric = run(config, train_data, valid_data)
    report_result.append(report_metric)
    os.makedirs('E:/data/iemocap/save', exist_ok=True)
    pickle.dump(report_result, open(os.path.join('E:/data/iemocap/save/', 'metric_report.pkl'), 'wb'))
