import numpy as np
import os

from AB3DMOT_libs.model_dense import AB3DMOT
from utils.read_datastructure import generate_dense_datastructure, get_img_shape, generate_indexed_datastructure
from SeeingThroughFog.tools.DatasetViewer.lib.read import get_kitti_object_list, load_radar_points
from utils.dense_transforms import Calibration, get_calib_from_json

from pcdet.utils.box_utils import boxes3d_kitti_camera_to_imageboxes


class Tracker:

    def __init__(self, dense_struct, stf_path):

        self.tracked_objects = []
        self.tracked_objects_assigned = []
        self.mot_tracker = AB3DMOT()

        self.calib = Calibration(get_calib_from_json(stf_path, sensor_type='hdl64'))
        self.img_shape = get_img_shape(dense_struct[0]['img'])

        # Loop over all frames in dense_struct
        for predictions in dense_struct:
            if predictions['gt_label'] is not None:
                label_path = predictions['gt_label']
            else:
                label_path = predictions['pred_label']

            assert os.path.exists(label_path), f'{label_path} does not exist'
            labels = get_kitti_object_list(label_path)

            radar_path = None
            if 'radar' in predictions.keys():
                radar_path = predictions['radar'] 
            self._run_tracking_for_single_frame(labels, radar_path)

        obj_with_radar = []
        for tracked_obj_in_frame in reversed(self.tracked_objects):
            assigned_obj = []
            for obj in tracked_obj_in_frame:
                if obj['has_radar'] or (obj['id'] in obj_with_radar):
                    obj_with_radar.append(obj['id'])
                    assigned_obj.append(obj)
            self.tracked_objects_assigned.append(assigned_obj)
        self.tracked_objects_assigned.reverse()


    def _run_tracking_for_single_frame(self, labels_in_frame, radar_path=None):

        # assert len(labels_in_frame) > 0  # Remove later, empty label should be fine

        detections = []
        scores = []

        # Loop over all objects in current frame as read from label
        for idx, obj in enumerate(labels_in_frame):

            if obj['identity'] not in ['Car', 'PassengerCar', 'RidableVehicle']:
                continue

            detection = [obj['height'], obj['width'], obj['length']]
            detection += [obj['posx'], obj['posy'], obj['posz'], obj['orient3d']]
            detections.append(detection)

            scores.append(obj['score'])

        if not detections:
            self.tracked_objects.append([])
            return

        # dets [[h,w,l,x,y,z,theta],...]
        detections = np.asarray(detections)
        scores = np.asarray(scores).reshape(-1)

        radar_det = None
        if radar_path is not None:
            # Radar detections in camera coordinates
            assert os.path.exists(radar_path), f'{radar_path} does not exist'
            pts_radar = load_radar_points(radar_path)
            vel_mask = pts_radar[:, 3] > 0
            pts_radar = pts_radar[vel_mask, :]
            if pts_radar.size > 0:
                pts_radar_cam = self.calib.radar_to_rect(pts_radar[:, :3])
                radar_det = pts_radar_cam.reshape((-1, 3))

        # trackers [[h,w,l,x,y,z,theta,id],...]
        trackers = self.mot_tracker.update(detections, scores, radar_det)

        tracked_objects_in_frame = []
        # Loop over all tracked objects in current frame
        for tracked_obj in trackers:

            # box_camera [x, y, z, l, h, w, r] in rect camera coords
            box_camera = np.asarray([tracked_obj[3], tracked_obj[4], tracked_obj[5],
                                     tracked_obj[2], tracked_obj[0], tracked_obj[1], tracked_obj[6]])
            box_camera = box_camera.reshape((1, -1))

            # box_camera_img [x1, y1, x2, y2]
            box_camera_img = boxes3d_kitti_camera_to_imageboxes(box_camera, self.calib, self.img_shape)

            box_camera = box_camera.reshape(-1)
            box_camera_img = box_camera_img.reshape(-1)

            xyz_camera = box_camera[:3].reshape(1, -1)
            xyz_lidar = self.calib.rect_to_lidar(xyz_camera).reshape(-1)

            tracked_obj_label = {'identity': 'Car',
                                 'xleft': int(box_camera_img[0]),
                                 'ytop': int(box_camera_img[1]),
                                 'xright': int(box_camera_img[2]),
                                 'ybottom': int(box_camera_img[3]),
                                 'posx': box_camera[0],
                                 'posy': box_camera[1],
                                 'posz': box_camera[2],
                                 'length': box_camera[3],
                                 'height': box_camera[4],
                                 'width': box_camera[5],
                                 'orient3d': box_camera[6],
                                 'posx_lidar': xyz_lidar[0],
                                 'posy_lidar': xyz_lidar[1],
                                 'posz_lidar': xyz_lidar[2],
                                 'rotx': 0,
                                 'roty': 0,
                                 'rotz': box_camera[6] + np.pi/2,
                                 'id': int(tracked_obj[7]),
                                 'has_radar': tracked_obj[8]}

            tracked_objects_in_frame.append(tracked_obj_label)

        self.tracked_objects.append(tracked_objects_in_frame)

    def get_num_of_frames(self):
        return len(self.tracked_objects)

    def get_tracked_objects_in_frame(self, frame_idx):
        assert frame_idx < self.get_num_of_frames(), f'Index {frame_idx} is out of range. Elements in dataset: {self.get_num_of_frames()}'
        return self.tracked_objects_assigned[frame_idx]


if __name__ == "__main__":

    # base_file = '2018-02-03_21-04-07_00000'
    # dense_path = '/lhome/dscheub/ObjectDetection/data/external/SeeingThroughFog'
    # stf_path = '/lhome/dscheub/ObjectDetection/AB3DMOT/SeeingThroughFog'

    # dense_struct = generate_dense_datastructure(dense_path, base_file, past_idx=-6, future_idx=0)
    # frame = [dense_struct[-1]]
    # tracker = Tracker(frame, stf_path)

    # print(frame[0]['gt_label'])
    # raw_label = get_kitti_object_list(frame[0]['gt_label'])

    # for obj in tracker.get_tracked_objects_in_frame(0):
    #    print(obj)
    # print('============')
    # for obj in raw_label:
    #    print(obj)

    stf_path = '/lhome/dscheub/ObjectDetection/AB3DMOT/SeeingThroughFog'
    seq_path = '/lhome/dscheub/ObjectDetection/data/external/SprayAugmentation/2019-09-11_21-15-42'
    dense_struct = generate_indexed_datastructure(seq_path)
    tracker = Tracker(dense_struct, stf_path)
