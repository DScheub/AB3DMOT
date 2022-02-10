import numpy as np
import os
from pyquaternion import Quaternion

from AB3DMOT_libs.model_dense import AB3DMOT
from utils.read_datastructure import generate_dense_datastructure, get_img_shape, generate_indexed_datastructure
from SeeingThroughFog.tools.DatasetViewer.lib.read import get_kitti_object_list, load_radar_points
from utils.dense_transforms import Calibration, get_calib_from_json
from utils.planes import ObjectAnchor

from pcdet.utils.box_utils import boxes3d_kitti_camera_to_imageboxes


class Tracker:

    def __init__(self, dense_struct, stf_path):

        self.tracked_objects = []
        self.mot_tracker = AB3DMOT()

        self.calib = Calibration(get_calib_from_json(stf_path, sensor_type='hdl64'))
        self.img_shape = get_img_shape(dense_struct[0]['img'])

        self.anchor = ObjectAnchor(dense_struct, 'single')

        # Loop over all frames in dense_struct and prepare data
        dets = []
        conf = []
        radar_dets = []
        for predictions in dense_struct:

            # Load label
            if predictions['gt_label'] is not None:
                label_path = predictions['gt_label']
            elif predictions['pred_label'] is not None:
                label_path = predictions['pred_label']
            else:
                label_path = None

            assert label_path, 'No label for pc: ' + str(predictions['pc'])
            if label_path:
                assert os.path.exists(label_path), f'{label_path} does not exist'
                labels = get_kitti_object_list(label_path)

            # Read detections from label
            dets_per_frame = []
            conf_per_frame = []
            if len(labels) > 0:
                dets_per_frame = np.zeros((len(labels), 7))
                conf_per_frame = np.zeros(len(labels))
                labels_anchored = self.anchor.anchor_object_to_ground(predictions['pc'], labels, self.calib, self.img_shape)
                # for idx, obj in enumerate(labels):
                for idx, obj in enumerate(labels_anchored):
                    # obj = anchor_object_to_ground(predictions['pc'], obj)
                    if (obj['width'] < 1) or (obj['height'] < 1):
                        continue
                    dets_per_frame[idx, :] = np.asarray([obj['posx'], obj['posy'], obj['posz'],
                                                         obj['orient3d'], obj['length'], obj['width'], obj['height']])
                    conf_per_frame[idx] = obj['score']
            dets.append(dets_per_frame)
            conf.append(conf_per_frame)

            # Read radar detections
            radar_dets_per_frame = [] 
            if 'radar' in predictions.keys():
                pts_radar = load_radar_points(predictions['radar'])
                if pts_radar.size > 0:
                    pts_radar_cam = self.calib.radar_to_rect(pts_radar[:, :3]).reshape((-1, 3))
                    radar_dets_per_frame = np.concatenate((pts_radar_cam, pts_radar[:, 3:]), axis=1)
            radar_dets.append(radar_dets_per_frame)

        # Run tracker
        self.mot_tracker.run_for_all_frames(dets, conf, radar_dets)

        # Generate Labels
        assert self.mot_tracker.get_num_of_frames() == len(dense_struct)
        for frame_idx in range(self.mot_tracker.get_num_of_frames()):
            trackers_in_frame = self.mot_tracker.get_tracked_obj_in_frame(frame_idx)
            labels_in_frame = []
            for obj in trackers_in_frame:
                state = obj.get_state()  # [x y z r l w h]
                # box_camera [x, y, z, l, h, w, r] in rect camera coords
                box_camera = np.concatenate((state[0:3], state[4:5], state[6:7], state[5:6], state[3:4]))
                box_camera = box_camera.reshape(1, -1)
                xyz_camera = box_camera[0, :3].reshape(1, -1)
                xyz_lidar = self.calib.rect_to_lidar(xyz_camera).reshape(-1)
                # box_camera_img [x1, y1, x2, y2]
                box_camera_img = boxes3d_kitti_camera_to_imageboxes(box_camera, self.calib, self.img_shape)

                box_camera = box_camera.reshape(-1)
                box_camera_img = box_camera_img.reshape(-1)
                qt = Quaternion(axis=[0, 0, 1], angle=box_camera[6] + np.pi/2)
                tracked_obj_label = {'identity': 'Car',
                                     'trunctuated': -1,
                                     'occlusion': -1,
                                     'angle': -1,
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
                                     'rotx': 0,
                                     'roty': 0,
                                     'rotz': box_camera[6] + np.pi/2,
                                     'score': 1.0,
                                     'qx': qt[0],
                                     'qy': qt[1],
                                     'qz': qt[2],
                                     'qw': qt[3],
                                     'visibleRGB': True,
                                     'visibleGated': False,
                                     'visibleLidar': True,
                                     'visibleRadar': False,
                                     'posx_lidar': xyz_lidar[0],
                                     'posy_lidar': xyz_lidar[1],
                                     'posz_lidar': xyz_lidar[2],
                                     'id': int(obj.id)
                                     }
                labels_in_frame.append(tracked_obj_label)
            self.tracked_objects.append(labels_in_frame)

    def get_num_of_frames(self):
        return len(self.tracked_objects)

    def get_tracked_objects_in_frame(self, frame_idx):
        assert frame_idx < self.get_num_of_frames(), f'Index {frame_idx} is out of range. Elements in dataset: {self.get_num_of_frames()}'
        return self.tracked_objects[frame_idx]


if __name__ == "__main__":

    from utils.read_datastructure import write_kitti_label

    stf_path = '/lhome/dscheub/ObjectDetection/AB3DMOT/SeeingThroughFog'
    # seq_path = '/lhome/dscheub/ObjectDetection/data/external/SprayAugmentation/2019-09-11_21-15-42'
    # seq_path = '/media/dscheub/Data_Spray/2021-07-26_19-17-21'
    # drive_path = '/media/dscheub/Data_Spray'
    for rec_dir in os.listdir(drive_path):
        if '2021' in rec_dir:
            seq_path = os.path.join(drive_path, rec_dir)
            label_path = f'{seq_path}/labels/tracked_labels'
            if os.path.exists(label_path):
                print(f'{label_path} already exists -> Skipping')
                continue
            else:
                os.mkdir(label_path)
            dense_struct = generate_indexed_datastructure(seq_path)
            tracker = Tracker(dense_struct, stf_path)
            for idx in range(tracker.get_num_of_frames()):
                kitti_object_list = tracker.get_tracked_objects_in_frame(idx)
                frame_id = dense_struct[idx]['frame_id']['base']
                label_file = os.path.join(label_path, f'{frame_id}.txt')
                write_kitti_label(label_file, kitti_object_list)
