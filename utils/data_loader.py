# -*- coding: utf-8 -*-

import numpy as np
from utils import format_filename
from config import PROCESSED_DATA_DIR, TRAIN_DATA_TEMPLATE, DEV_DATA_TEMPLATE, TEST_DATA_TEMPLATE

#这个没用到了
def load_data(dataset: str, data_type: str):
    if data_type == 'train':
        return np.load(format_filename(PROCESSED_DATA_DIR, TRAIN_DATA_TEMPLATE, dataset=dataset))
    elif data_type == 'dev':
        return np.load(format_filename(PROCESSED_DATA_DIR, DEV_DATA_TEMPLATE, dataset=dataset))
    elif data_type == 'test':
        return np.load(format_filename(PROCESSED_DATA_DIR, TEST_DATA_TEMPLATE, dataset=dataset))
    else:
        raise ValueError('`data_type` not understood: {data_type}')
