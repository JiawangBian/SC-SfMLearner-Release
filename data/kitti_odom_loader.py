from __future__ import division
import numpy as np
from path import Path
import scipy.misc
from collections import Counter
import os

class KittiOdomLoader(object):
    def __init__(self,
                 dataset_dir,
                 img_height=256,
                 img_width=832):

        self.dataset_dir = Path(dataset_dir)
        self.img_height = img_height
        self.img_width = img_width
        self.cam_ids = ['2', '3']
        self.train_sets = ['00', '01', '02', '03', '04', '05', '06', '07', '08']
        self.test_sets = ['09', '10']
        self.collect_train_folders()

    def collect_train_folders(self):
        self.scenes = []
        sequence_list = (self.dataset_dir/'sequences').dirs()
        for sequence in sequence_list:
            if sequence.name in self.train_sets:
                self.scenes.append(sequence)

    def collect_scenes(self, drive):
        train_scenes = []
        for c in self.cam_ids:
            scene_data = {'cid': c, 'dir': drive, 'frame_id': [], 'rel_path': drive.name + '_' + c}
            
            img_dir = Path(scene_data['dir']/'image_{}/'.format(scene_data['cid']))
            scene_data['frame_id'] = [x.split('.')[0] for x in os.listdir(img_dir)]

            sample, zoom_x, zoom_y = self.load_image(scene_data, 0)
            if sample is None:
                return []

            scene_data['intrinsics'] = self.read_calib_file(c, drive/'calib.txt', zoom_x, zoom_y)
            train_scenes.append(scene_data)

        return train_scenes

    def get_scene_imgs(self, scene_data):
        for (i,frame_id) in enumerate(scene_data['frame_id']):
            yield {"img":self.load_image(scene_data, i)[0], "id":frame_id}

    def load_image(self, scene_data, tgt_idx):
        img_file = scene_data['dir']/'image_{}'.format(scene_data['cid'])/scene_data['frame_id'][tgt_idx]+'.png'
        if not img_file.isfile():
            return None
        img = scipy.misc.imread(img_file)
        zoom_y = self.img_height/img.shape[0]
        zoom_x = self.img_width/img.shape[1]
        img = scipy.misc.imresize(img, (self.img_height, self.img_width))
        return img, zoom_x, zoom_y

    def read_calib_file(self, cid, filepath, zoom_x, zoom_y):
        with open(filepath, 'r') as f:
            C = f.readlines()
        def parseLine(L, shape):
            data = L.split()
            data = np.array(data[1:]).reshape(shape).astype(np.float32)
            return data
        proj_c2p = parseLine(C[int(cid)], shape=(3,4))
        calib = proj_c2p[0:3, 0:3]
        calib[0,:] *=  zoom_x
        calib[1,:] *=  zoom_y

        return calib


 