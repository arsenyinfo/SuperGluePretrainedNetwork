import torch
import cv2


def frame2tensor(frame, device='cpu'):
    if len(frame.shape) == 3 and frame.shape[-1] == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return torch.from_numpy(frame / 255.).float()[None, None].to(device)


def read_image(path):
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return frame2tensor(image)
