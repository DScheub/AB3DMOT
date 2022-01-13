import numpy as np

from AB3DMOT_libs.model import AB3DMOT
from utils.read_datastructure import generate_dense_datastructure, get_img_shape
from SeeingThroughFog.tools.DatasetViewer.lib.read import get_kitti_object_list
from utils.dense_transforms import Calibration, get_calib_from_json

from pcdet.utils.box_utils import boxes3d_kitti_camera_to_imageboxes


class Tracker:

    def __init__(self, dense_struct, stf_path):

        self.tracked_objects = []
        self._run_tracking(dense_struct, stf_path)

    def _run_tracking(self, dense_struct, stf_path):

        calib = Calibration(get_calib_from_json(stf_path, sensor_type='hdl64'))
        mot_tracker = AB3DMOT()

        for predictions in dense_struct:

            img_shape = get_img_shape(predictions['img'])
            pred_label_path = predictions['pred_label']
            pred_labels = get_kitti_object_list(pred_label_path)

            assert len(pred_labels) > 0

            detections = []
            infos = []
            for idx, pred_label in enumerate(pred_labels):

                if pred_label['identity'] != 'Car':
                    continue

                detection = [pred_label['height'], pred_label['width'], pred_label['length']]
                detection += [pred_label['posx'], pred_label['posy'], pred_label['posz'], pred_label['orient3d']]
                detections.append(detection)

                info = [pred_label['score']]
                infos.append(info)

            if not detections:
                self.tracked_objects.append([])
                continue

            # dets [[h,w,l,x,y,z,theta],...]
            detections = {'dets': np.asarray(detections),
                          'info': np.asarray(infos)}

            # trackers [[h,w,l,x,y,z,theta,id],...]
            trackers = mot_tracker.update(detections)

            tracked_objects_in_frame = []
            for tracked_obj in trackers:

                box_camera = np.asarray([tracked_obj[3], tracked_obj[4], tracked_obj[5],
                                         tracked_obj[2], tracked_obj[0], tracked_obj[1], tracked_obj[6]])
                box_camera = box_camera.reshape((1, -1))
                box_camera_img = boxes3d_kitti_camera_to_imageboxes(box_camera, calib, img_shape)  # returns [x1, y1, x2, y2]

                box_camera = box_camera.reshape(-1)
                box_camera_img = box_camera_img.reshape(-1)

                tracked_obj_label = {'identity': 'Car',
                                     'xleft': int(box_camera_img[0]),
                                     'ytop': int(box_camera_img[1]),
                                     'xright': int(box_camera_img[2]),
                                     'ybottom': int(box_camera_img[3]),
                                     'posx': box_camera[0],
                                     'posy': box_camera[1],
                                     'posz': box_camera[2],
                                     'length': box_camera[3],
                                     'heigth': box_camera[4],
                                     'width': box_camera[5],
                                     'orient3d': box_camera[6],
                                     'id': int(tracked_obj[7]),
                                     'confidence': tracked_obj[-1]}

                tracked_objects_in_frame.append(tracked_obj_label)

            self.tracked_objects.append(tracked_objects_in_frame)

    def get_num_of_frames(self):
        return len(self.tracked_objects)

    def get_tracked_objects_in_frame(self, frame_idx):
        assert frame_idx < self.get_num_of_frames(), f'Index {frame_idx} is out of range. Elements in dataset: {self.get_num_of_frames()}'
        return self.tracked_objects[frame_idx]


if __name__ == "__main__":

    base_file = '2018-02-03_21-04-07_00000'
    dense_path = '/lhome/dscheub/ObjectDetection/data/external/SeeingThroughFog'
    stf_path = '/lhome/dscheub/ObjectDetection/AB3DMOT/SeeingThroughFog'

    dense_struct = generate_dense_datastructure(dense_path, base_file, past_idx=-6, future_idx=0)
    tracker = Tracker(dense_struct, stf_path)

    for frame_idx in range(tracker.get_num_of_frames()):
        for obj in tracker.get_tracked_objects_in_frame(frame_idx):
            print(obj)
        print('============')
