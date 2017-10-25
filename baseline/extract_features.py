from baseline.extractor import Extractor
import numpy as np
from baseline.utils import rescale_list
import imageio
import os
import pickle as pkl


extractor = Extractor()
NUMBER_OF_FRAMES = 40
PATH_TO_DIR = os.path.join(os.getcwd().replace('/baseline', ''), 'input/data')


def get_features(frames):
    features = list()
    for frame in frames:
        activations = extractor.extract(frame)
        features.append(activations)
    return np.array(features, np.float32)


def store_features():
    videos = os.listdir(PATH_TO_DIR)
    path_to_store = os.path.join(os.getcwd().replace('/baseline', ''), 'input', 'Inception_Activations')
    if not os.path.isdir(path_to_store):
        os.mkdir(path_to_store)
    for video_file in videos:
        path_to_video = os.path.join(PATH_TO_DIR, video_file)
        video = imageio.get_reader(path_to_video, size=(480, 360))
        frames = [frame for frame in video]
        if len(frames) < NUMBER_OF_FRAMES:
            continue
        frames = rescale_list(frames, NUMBER_OF_FRAMES)
        features = get_features(frames)
        name = video_file.split('.')[0] + '.pkl'
        if len(features) != 0:
            path = os.path.join(path_to_store, name)
            if os.path.isfile(path):
                print('{} already stored.'.format(name))
                continue
            with open(path, 'wb') as file:
                print('storing {}...'.format(name))
                pkl.dump(features, file, protocol=pkl.HIGHEST_PROTOCOL)


store_features()
