import cv2
import numpy as np
from collections import namedtuple, deque
import random
import imageio
import os

def process_image(image, crop_size=(34,194,0,160),target_size=(84,84),normalize=True):
    '''
    Grayscale, crop and resize image

    Input
    - image: shape(h,w,c),(210,160,3)
    - crop_size: shape(min_h,max_h,min_w,max_w)
    - target_size: (h,w)
    - normalize: [0,255] -> [0,1]
    
    Output
    - shape(84,84)
    '''
    frame = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) # To grayscale
    frame = frame[crop_size[0]:crop_size[1],crop_size[2]:crop_size[3]]
    frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)  # Resize
    if normalize:
        return frame.astype(np.float32)/255 #normalize
    else:
        return frame


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        '''
        return List[Transition]
        '''
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    

class VideoRecorder:
    def __init__(self, dir_name, fps=30):
        self.dir_name = dir_name
        self.fps = fps
        self.frames = []

    def reset(self):
        self.frames = []

    def record(self, frame):
        self.frames.append(frame)

    def save(self, file_name):
        path = os.path.join(self.dir_name, file_name)
        imageio.mimsave(path, self.frames, fps=self.fps, macro_block_size = None)
