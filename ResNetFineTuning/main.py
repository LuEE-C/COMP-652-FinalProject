"""Training script.
usage: simple_cnn_approach.py [options]

options:
    --hidden_size=hs            Hidden size of nn [default: 64]
    --learning_rate=lr          Learning rate of inner loop [default: 1e-3]
    --batch_size=bs             Size of task to train with [default: 16]
    --focal_loss=fl             Focal loss ratio    [default: 1]
    --model=m                   Name of model to load [default: PretrainedResnet50]
    --oversample=om             Wether to oversample the rare classe [default: 50]
    --data_root=dr              Root containing the raw data [default: ../prots_data/]
    -h, --help                  Show this help message and exit
"""
from docopt import docopt
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
matplotlib.use('Agg')
from tensorboardX import SummaryWriter
import pickle as pkl
import pandas as pd
import numpy as np
import os
from data_util import TrainProtsDataset, ValProtsDataset, TestProtsDataset
from model import ResNet18, PretrainedInception, PretrainedZoo, PretrainedResnet152, PretrainedResnet50
from tqdm import tqdm
from sklearn.metrics import f1_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super().__init__()
        self.gamma = gamma

    def forward(self, input, target):
        if not (target.size() == input.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})"
                             .format(target.size(), input.size()))

        max_val = (-input).clamp(min=0)
        loss = input - input * target + max_val + \
               ((-max_val).exp() + (-input - max_val).exp()).log()

        invprobs = F.logsigmoid(-input * (target * 2.0 - 1.0))
        loss = (invprobs * self.gamma).exp() * loss

        return loss.sum(dim=1).mean()


class PretrainedApproach:
    def __init__(self, lr, hidden_size, batch_size, data_root, model, oversample,focal_loss_ratio):

        train_transform = transforms.Compose([
            transforms.Resize(336),
            transforms.RandomRotation(30),
            transforms.RandomVerticalFlip(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(10),
            transforms.ColorJitter(brightness=0.5, contrast=.5, saturation=.5),
            transforms.RandomResizedCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.08069, 0.05258, 0.05487, 0.08282], [0.13704, 0.10145, 0.15313, 0.13814])])

        test_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize([0.08069, 0.05258, 0.05487, 0.08282], [0.13704, 0.10145, 0.15313, 0.13814])])

        self.train_dataset = TrainProtsDataset(data_root, train_transform, oversample)
        self.valid_dataset = ValProtsDataset(data_root, test_transform)
        self.test_dataset = TestProtsDataset(data_root, test_transform)
        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
        self.valid_loader = DataLoader(self.valid_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=False)
        self.test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=False)

        self.test_df = pd.read_csv(os.path.join(data_root, 'sample_submission.csv'))

        self.id_string = 'Runs/fl_cos_{}_lr{}_hs{}_bsize{}_oversample{}_flr{}_80_20'.format(model, str(lr), str(hidden_size), str(batch_size), oversample, str(focal_loss_ratio))
        self.model = eval(model)().to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=5, eta_min=lr/10)
        self.lr = lr
        self.criterion = FocalLoss(focal_loss_ratio)
        self.class_threshold = np.ones(shape=(28,)) - 0.5

        self.writer = SummaryWriter(self.id_string)
        self.steps = 0
        self.epochs = 0
        self.load_model()

    def load_model(self):
        if os.path.isfile(self.id_string + '/model') and os.path.isfile(self.id_string + '/ep_step.pkl'):
            print('Loading model')
            self.model = torch.load(self.id_string + '/model')
            self.optimizer = torch.load(self.id_string + '/optimizer')
            ep_step = pkl.load(open(self.id_string + '/ep_step.pkl', 'rb'))
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=30)
            self.epochs = ep_step[0]
            self.scheduler.step(self.epochs)
            self.steps = ep_step[1]

    def checkpoint_model(self):
        torch.save(self.model, self.id_string + '/model')
        torch.save(self.optimizer, self.id_string + '/optimizer')
        pkl.dump([self.epochs, self.steps], open(self.id_string + '/ep_step.pkl', 'wb'))

    def train_till_convergence(self):
        for _ in tqdm(range(200)):
            self.epoch_train()

    def finetune(self):
        for param in self.model.pretrained.parameters():
            param.requires_grad = False
        self.epoch_train()

        for param in self.model.parameters():
            param.requires_grad = True
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr/10)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=10, eta_min=self.lr/100)
        for _ in tqdm(range(500)):
            self.epoch_train()

    def epoch_train(self):
        for batch_x, batch_y in self.train_loader:
            self.train_on_batch(batch_x.to(device), batch_y.to(device))
        self.validation_run()
        self.make_test_predictions()
        self.checkpoint_model()
        self.epochs += 1

    def output_csv(self):
        train_features, train_targets = [], []
        test_features = []
        with torch.no_grad():
            self.model.eval()
            for batch_x, batch_y in tqdm(self.train_loader):
                preds = self.model.pretrained(batch_x.to(device))
                train_features.append(preds.cpu().numpy())
                train_targets.append(batch_y.cpu().numpy())

            for batch_x in self.test_loader:
                preds = self.model.pretrained(batch_x.to(device))
                test_features.append(preds.cpu().numpy())

        train_features = np.vstack(train_features)
        train_targets = np.vstack(train_targets)
        test_features = np.vstack(test_features)
        np.save('Train_x.npy', train_features)
        np.save('Train_y.npy', train_targets)
        np.save('Test_x.npy', test_features)

    # custom function
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def validation_run(self):
        validation_loss = []
        preds, real = [], []
        with torch.no_grad():
            for batch_x, batch_y in self.valid_loader:
                val_loss, pred = self.validate_on_batch(batch_x.to(device), batch_y.to(device))
                validation_loss.append(val_loss)
                preds.append(pred.detach().cpu().numpy())
                real.append(batch_y.detach().cpu().numpy())
        preds = np.vstack(preds)
        # Sigmoid the preds
        preds = 1 / (1 + np.exp(-preds))
        real = np.vstack(real)
        validation_f1_threshold, validation_f1_50, validation_f1_ratios = self.get_f1_score(real, preds)
        # self.update_class_weights(real, preds)
        self.writer.add_scalar('Validation loss', np.mean(validation_loss), self.epochs)
        self.writer.add_scalar('Validation F1 threshold', np.mean(validation_f1_threshold), self.epochs)
        self.writer.add_scalar('Validation F1 50', np.mean(validation_f1_50), self.epochs)
        self.writer.add_scalar('Validation F1 ratios', np.mean(validation_f1_ratios), self.epochs)
        self.scheduler.step()

    def train_on_batch(self, batch_x, batch_y):
        self.model.train()
        preds = self.model(batch_x)
        loss = self.criterion(preds, batch_y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.writer.add_scalar('Training loss', loss.item(), self.steps)
        self.steps += 1
        return preds

    def update_class_weights(self, true_y, true_pred):
        for i in range(28):
            best_val, best_f1 = 0.5, 0
            for j in range(500):
                f1 = f1_score(true_y[:, i], (true_pred[:, i] > j / 500).astype(int))
                if f1 > best_f1:
                    best_f1 = f1
                    best_val = j/500
            self.class_threshold[i] = best_val

    def get_f1_score(self, true_y, true_pred):
        preds = (true_pred > self.class_threshold).astype(int)
        preds_50 = (true_pred > 0.5).astype(int)
        preds_train_ratio = true_pred.copy()
        for i in range(28):
            preds_train_ratio[:, i] = (preds_train_ratio[:, i] > np.quantile(preds_train_ratio[:, i], 1 - self.train_dataset.ratios[i])).astype(int)

        individual_f1 = {'class': [], 'f1_score_50': [], 'f1_score_train_ratio': [], 'f1_score_threshold':[]}
        for i in range(28):
            individual_f1['class'].append(i)
            individual_f1['f1_score_threshold'].append(f1_score(true_y[:, i], preds[:, i]))
            individual_f1['f1_score_50'].append(f1_score(true_y[:, i], preds_50[:, i]))
            individual_f1['f1_score_train_ratio'].append(f1_score(true_y[:, i], preds_train_ratio[:, i]))
        f1_df = pd.DataFrame.from_dict(individual_f1)
        f1_df.to_csv(self.id_string + '/valid_f1_class_' + str(self.epochs) + '.csv', index=False)
        self.update_class_weights(true_y, true_pred)
        return f1_score(true_y, preds, average='macro'), f1_score(true_y, preds_50, average='macro'), f1_score(true_y, preds_train_ratio, average='macro')

    def validate_on_batch(self, batch_x, batch_y):
        self.model.eval()
        preds = self.model(batch_x)
        loss = self.criterion(preds, batch_y)
        return loss.item(), preds

    def make_test_predictions(self):
        self.model.eval()
        predictions = []
        with torch.no_grad():
            for batch_x in self.test_loader:
                preds = self.model(batch_x.to(device)).detach().cpu().numpy()
                # Sigmoid the preds
                preds = 1 / (1 + np.exp(-preds))
                predictions.append(preds)

        predictions = np.vstack(predictions)
        preds_threshold = (predictions > self.class_threshold).astype(int)
        preds_50 = (predictions > 0.5).astype(int)
        preds_train_ratio = predictions.copy()
        for i in range(28):
            preds_train_ratio[:, i] = (preds_train_ratio[:, i] > np.quantile(preds_train_ratio[:, i], 1 - self.train_dataset.ratios[i])).astype(int)

        preds_threshold_answer = []
        preds_50_answer = []
        preds_train_ratio_answer = []
        for i in range(preds_threshold.shape[0]):
            pred_threshold_answer = []
            pred_50_answer = []
            pred_train_ratio_answer = []
            for j in range(28):
                if preds_threshold[i, j] == 1:
                    pred_threshold_answer.append(j)
                if preds_50[i, j] == 1:
                    pred_50_answer.append(j)
                if preds_train_ratio[i, j] == 1:
                    pred_train_ratio_answer.append(j)
            preds_threshold_answer.append(pred_threshold_answer)
            preds_50_answer.append(pred_50_answer)
            preds_train_ratio_answer.append(pred_train_ratio_answer)

        str_predictions = [' '.join(map(str, p)) for p in preds_threshold_answer]
        self.test_df['Predicted'] = str_predictions
        self.test_df.to_csv(self.id_string + '/test_preds_threshold_' + str(self.epochs) + '.csv', index=False)

        str_predictions = [' '.join(map(str, p)) for p in preds_50_answer]
        self.test_df['Predicted'] = str_predictions
        self.test_df.to_csv(self.id_string + '/test_preds_50_' + str(self.epochs) + '.csv', index=False)

        str_predictions = [' '.join(map(str, p)) for p in preds_train_ratio_answer]
        self.test_df['Predicted'] = str_predictions
        self.test_df.to_csv(self.id_string + '/test_preds_train_ratio_' + str(self.epochs) + '.csv', index=False)


if __name__ == '__main__':

    # train_x = np.load('Train_x.npy')
    # print(train_x.shape)
    # train_y = np.load('Train_y.npy')
    # print(train_y.shape)
    # test_x = np.load('Test_x.npy')
    # print(test_x.shape)

    args = docopt(__doc__)
    hidden_size = int(args['--hidden_size'])
    learning_rate = float(args['--learning_rate'])
    batch_size = int(args['--batch_size'])
    data_root = args['--data_root']
    model = args['--model']
    oversample = int(args['--oversample'])
    focal_loss_ratio = float(args['--focal_loss'])

    trainer = PretrainedApproach(learning_rate, hidden_size, batch_size, data_root, model, oversample, focal_loss_ratio)
    trainer.finetune()
