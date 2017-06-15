#coding:utf8

import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as pl

log_file = open(sys.argv[1])
prefix = sys.argv[2]
batch_size = 1
epoch_size = -1

os.chdir('logs/pic')

train_epochs = []
train_acc = []
train_logloss = []
train_smoothl1loss = []
for l in log_file:
    if 'Speed' in l:
        sp = l.strip().split()
        epoch = int(sp[1][sp[1].find('[') + 1 : sp[1].find(']')])
        batch = int(sp[3][sp[3].find('[') + 1 : sp[3].find(']')])
        epoch_size = max(epoch_size, batch)
        train_epochs.append(epoch * epoch_size + batch)
        train_acc.append(float(sp[-3].split('=')[-1][:-1]))
        train_logloss.append(float(sp[-2].split('=')[-1][:-1]))
        train_smoothl1loss.append(float(sp[-1].split('=')[-1][:-1]))

pl.plot(train_epochs, train_acc)
pl.ylim(0.90, 1.)
pl.savefig('{}_acc.jpg'.format(prefix))
pl.clf()

pl.plot(train_epochs, train_logloss)
pl.ylim(0., 0.5)
pl.savefig('{}_logloss.jpg'.format(prefix))
pl.clf()

pl.plot(train_epochs, train_smoothl1loss)
pl.ylim(0.02, 0.2)
pl.savefig('{}_smoothl1loss.jpg'.format(prefix))
