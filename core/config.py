import numpy as np
import os
from easydict import EasyDict as edict

import json
cfg = edict()

cfg.GPU_ID = '0'
cfg.LR = '[0.0001]*20000'
cfg.NUM_ITERS = len(eval(cfg.LR))
cfg.NUM_CLASSES = 20
cfg.MODAL = 'all'
cfg.FEATS_DIM = 2048
cfg.BATCH_SIZE = 32
cfg.DATA_PATH = './data/THUMOS14'
cfg.NUM_WORKERS = 8
cfg.LAMBDA = 0.01
cfg.R_P = 5
cfg.R_N = 20
cfg.m = 3
cfg.M = 6
cfg.TEST_FREQ = 1
cfg.PRINT_FREQ = 20
cfg.CLASS_THRESH = 0.2
cfg.NMS_THRESH = 0.6
cfg.CAS_THRESH = np.arange(0.0, 0.25, 0.025)
cfg.ANESS_THRESH = np.arange(0.1, 0.925, 0.025)
cfg.TIOU_THRESH = np.linspace(0.1, 0.7, 7)
cfg.UP_SCALE = 24
cfg.GT_PATH = os.path.join(cfg.DATA_PATH, 'gt.json')
cfg.SEED = 0
cfg.FEATS_FPS = 25
cfg.NUM_SEGMENTS = 750
cfg.CLASS_DICT = {'BaseballPitch': 0, 'BasketballDunk': 1, 'Billiards': 2, 
                  'CleanAndJerk': 3, 'CliffDiving': 4, 'CricketBowling': 5, 
                  'CricketShot': 6, 'Diving': 7, 'FrisbeeCatch': 8, 
                  'GolfSwing': 9, 'HammerThrow': 10, 'HighJump': 11, 
                  'JavelinThrow': 12, 'LongJump': 13, 'PoleVault': 14, 
                  'Shotput': 15, 'SoccerPenalty': 16, 'TennisSwing': 17, 
                  'ThrowDiscus': 18, 'VolleyballSpiking': 19}

# 聚类
cfg.UPDATE_LABEL_ITERS = 500

cfg.MAGNITUDES_THRESH =  np.arange(0.4, 0.875, 0.025)

cfg.N_SIMILAR = 16
