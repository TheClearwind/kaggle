import torch
import numpy as np


class Config:
    save_path = 'models/new_model_74.pth'
    device = "cuda" if torch.cuda.is_available() else "cpu"
    camera_matrix = np.array([[2304.5479, 0, 1686.2379],
                              [0, 2305.8757, 1354.9849],
                              [0, 0, 1]], dtype=np.float32)
    IMG_WIDTH = 2048
    IMG_HEIGHT = IMG_WIDTH // 4
    MODEL_SCALE = 8

    PATH = "./"

    SWITCH_LOSS_EPOCH = 4

    DISTANCE_THRESH_CLEAR = 2
    IMG_SHAPE = (2710, 3384, 3)
