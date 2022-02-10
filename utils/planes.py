from sklearn.linear_model import RANSACRegressor
import numpy as np
from pcdet.utils.box_utils import boxes3d_kitti_camera_to_imageboxes

def calculate_plane(pointcloud, x_of_interest=0, standart_height=-1.65):
    """
    caluclates plane from loaded pointcloud

    :param pointcloud: binary with x,y,z, coordinates
    :param tolerance:
    :return:
    """
                # (pointcloud[:, 0] > 10) & \
                # (pointcloud[:, 0] < 70) & \
    valid_loc = (pointcloud[:, 2] < -1.55-0.01*pointcloud[:, 0]) & \
                (pointcloud[:, 2] > -1.86-0.01*pointcloud[:, 0]) & \
                (pointcloud[:, 0] < x_of_interest + 15 + 0.5*x_of_interest) & \
                (pointcloud[:, 0] > x_of_interest - 10 - 0.5*x_of_interest) & \
                (pointcloud[:, 1] > -7) & \
                (pointcloud[:, 1] < 10)
    pc_rect = pointcloud[valid_loc]
    #print('pc_rect.shape', pc_rect.shape)
    if pc_rect.shape[0] <= pc_rect.shape[1]:
        w = [0, 0, 1]
        # Standard height from vehicle mounting position in dense
        h = standart_height
    else:
        try:
            reg = RANSACRegressor(loss='squared_loss',max_trials=1000).fit(pc_rect[:, [0, 1]], pc_rect[:, 2])
            w = np.zeros(3)
            w[0] = reg.estimator_.coef_[0]
            w[1] = reg.estimator_.coef_[1]
            w[2] = 1.0
            h = reg.estimator_.intercept_
            w = w / np.linalg.norm(w)

            # print(reg.estimator_.coef_)
            # print(reg.get_params())
            # print(w,h)
        except:
            print('Was not able to estimate a ground plane. Using default flat earth assumption')
            w = [0, 0, 1]
            # Standard height from vehicle mounting position in dense
            h = standart_height

    return w, h, pc_rect


class ObjectAnchor:

    def __init__(self, dataset, method):

        assert method in ['mean', 'avg', 'buffer', 'single'], f'Unknown method {method}'

        self.w = np.asarray([0, 0, 1])
        self.h = -1.65

        self.use_buffer = False
        self.buffer_size = 200
        self.buffer_index = 0
        self.buffer_w = [None for _ in range(self.buffer_size)]
        self.buffer_h = [None for _ in range(self.buffer_size)]

        self.use_single = False

        if method == 'mean':
            w_stacked = np.zeros((3, len(dataset)))
            h_stacked = np.zeros(len(dataset))
            for idx, frame in enumerate(dataset):
                pc = np.fromfile(frame['pc'], dtype=np.float32).reshape(-1, 5)
                w_stacked[:, idx], h_stacked[idx], _ = calculate_plane(pc)
            w = np.mean(w_stacked, axis=1)
            self.w = w / np.linalg.norm(w)
            self.h = np.mean(h_stacked)
        elif method == 'avg':
            pc = None
            for frame in dataset:
                pc_current = np.fromfile(frame['pc'], dtype=np.float32).reshape(-1, 5)
                _, _, pc_interest = calculate_plane(pc_current)
                if isinstance(pc, np.ndarray):
                    np.concatenate((pc, pc_interest), axis=0)
                else:
                    pc = pc_interest
            self.w, self.h, _ = calculate_plane(pc)
        elif method == 'buffer':
            self.use_buffer = True
        elif method == 'single':
            self.use_single = True

    def anchor_object_to_ground(self, pc_path, objects, calib, img_shape):

        if self.use_buffer:
            pc = np.fromfile(pc_path, dtype=np.float32).reshape(-1, 5)
            idx = self.buffer_index % self.buffer_size
            self.buffer_w[idx], self.buffer_h[idx], _ = calculate_plane(pc)
            self.w, self.h, i = np.zeros(3), 0, 0
            for w, h in zip(self.buffer_w, self.buffer_h):
                if isinstance(w, np.ndarray) and (h is not None):
                    self.w = (self.w*i + w) / (i + 1)
                    self.h = (self.h*i + h) / (i + 1)
                    i += 1
            self.buffer_index += 1

        elif self.use_single:
            pc = np.fromfile(pc_path, dtype=np.float32).reshape(-1, 5)
            self.w, self.h, _ = calculate_plane(pc)

        ground_normal, dist_lidar_to_ground = self.w, self.h  # lidar coord
        # ground_normal = np.asarray([-ground_normal[1], -ground_normal[2], ground_normal[0]])  # camera coord
        ground_normal = np.dot(calib.V2C[:3, :3], ground_normal)  # camera coord
        # print(f'{ground_normal=}')
        # print(f'{dist_lidar_to_ground=}')
        # print('==================')

        objects_anchored = []
        for obj in objects:
            # self.w, self.h, _ = calculate_plane(pc, x_of_interest=obj['posx_lidar'])
            # ground_normal, dist_lidar_to_ground = self.w, self.h  # lidar coord
            # ground_normal = np.dot(calib.V2C[:3, :3], ground_normal)  # camera coord

            base_old = np.asarray([obj['posx'], obj['posy'], obj['posz']])  # cammera coord
            # print(f'{base_old=}')
            pc = np.fromfile(pc_path, np.float32).reshape(-1, 5)
            # ground_normal, dist_lidar_to_ground = calculate_averaged_planes(dataset)  # lidar coord
            if dist_lidar_to_ground > 0:
                print(f'Unlikely: ground normal lidar coord: {ground_normal}, dist: {dist_lidar_to_ground}')
                objects_anchored.append(obj)
                continue
            # print(f'{ground_normal=}')
            # print(f'{dist_lidar_to_ground=}')

            dist_base_to_ground = np.dot(ground_normal, base_old) - dist_lidar_to_ground
            # print(f'{dist_base_to_ground=}')
            base_new = base_old - dist_base_to_ground*ground_normal
            # print(f'{base_new=}')
            
            obj_anchored = obj.copy()
            obj_anchored['posx'], obj_anchored['posy'], obj_anchored['posz'] = base_new[0], base_new[1], base_new[2]
            obj_anchored['height'] += dist_base_to_ground
            if obj_anchored['height'] < 1.0:
                print('Height ', obj_anchored['height'], ' unlikely -> Skipping')
                objects_anchored.append(obj)
                continue
            box_cam = np.asarray([obj_anchored['posx'], obj_anchored['posy'], obj_anchored['posz'], obj_anchored['length'],
                                  obj_anchored['height'], obj_anchored['width'], obj_anchored['orient3d']])
            box_cam = box_cam.reshape((1, -1))
            box_cam_img = boxes3d_kitti_camera_to_imageboxes(box_cam, calib, img_shape)
            obj_anchored['xleft'] = int(box_cam_img[0, 0])
            obj_anchored['ytop'] = int(box_cam_img[0, 1])
            obj_anchored['xright'] = int(box_cam_img[0, 2])
            obj_anchored['ybottom'] = int(box_cam_img[0, 3])

            pos_cam = np.concatenate((box_cam[0, :3], np.ones(1)))
            pos_lidar = np.matmul(calib.C2V, pos_cam.reshape(-1, 1))
            obj_anchored['posx_lidar'] = pos_lidar[0]
            obj_anchored['posy_lidar'] = pos_lidar[1]
            obj_anchored['posz_lidar'] = pos_lidar[2]

            # print(f'Dist to ground new base: {np.dot(ground_normal, box_cam[0, :3]) + dist_lidar_to_ground}')

            objects_anchored.append(obj_anchored)

        return objects_anchored


if __name__ == "__main__":

    from SeeingThroughFog.tools.DatasetViewer.lib.read import get_kitti_object_list
    from utils.dense_transforms import Calibration, get_calib_from_json
    from utils.read_datastructure import generate_indexed_datastructure
    import open3d as o3d

    stf_path = '/lhome/dscheub/ObjectDetection/AB3DMOT/SeeingThroughFog'
    idx = 70 
    # data_path = '/lhome/dscheub/ObjectDetection/data/external/SprayAugmentation/2019-09-11_21-15-42'
    data_path = '/lhome/dscheub/ObjectDetection/data/external/SprayAugmentation/2021-07-26_19-17-21/'
    # pc_path = f'{data_path}/lidar_hdl64_s3/velodyne_pointclouds/strongest_echo/00072_1568229354102870000.bin'

    dense_dataset = generate_indexed_datastructure(data_path)
    calib = Calibration(get_calib_from_json(stf_path=stf_path))

    idx = 0
    pc = np.fromfile(dense_dataset[idx]['pc'], dtype=np.float32)
    obj_list = get_kitti_object_list(dense_dataset[idx]['tracked_label'], calib.C2V)
    pc = pc.reshape((-1, 5))
    w, h, pc_ground = calculate_plane(pc, x_of_interest=obj_list[0]['posx_lidar'])
    print("posx_lidar: ",  obj_list[0]['posx_lidar'])
    print("w: ", w)
    print("h: ", h) 

    pc_draw = o3d.geometry.PointCloud()
    pc_draw.points = o3d.utility.Vector3dVector(pc[:, 0:3])
    ground_draw = o3d.geometry.PointCloud()
    ground_draw.points = o3d.utility.Vector3dVector(pc_ground[:, 0:3])
    ground_draw.paint_uniform_color([0, 0, 0])
    base_new = o3d.geometry.PointCloud()
    # base_new_lidar = np.asarray([obj_corrected['posx_lidar'], obj_corrected['posy_lidar'], obj_corrected['posz_lidar']])
    # base_new_lidar = base_new_lidar.reshape(1, 3)
    # base_new.points = o3d.utility.Vector3dVector(base_new_lidar)
    # base_new.paint_uniform_color([0, 1.0, 0])
    o3d.visualization.draw_geometries([pc_draw, ground_draw])

    asdfasd


    anchor = ObjectAnchor(dense_dataset, 'mean')
    print(f'Mean: w = {anchor.w}, h = {anchor.h}')
    anchor = ObjectAnchor(dense_dataset, 'avg')
    print(f'Avg: w = {anchor.w}, h = {anchor.h}')

    anchor = ObjectAnchor(dense_dataset, 'buffer')
    for i, frame in enumerate(dense_dataset):
        obj_list = get_kitti_object_list(frame['tracked_label'], calib.C2V)
        # obj = obj_list[0]
        anchor.anchor_object_to_ground(frame['pc'], obj_list, calib, [1920, 1080])
        if i % 100 == 0:
            print(f'Buffer: w = {anchor.w}, h = {anchor.h}')

    pc = np.fromfile(pc_path, dtype=np.float32)
    pc = pc.reshape((-1, 5))
    w, h, pc_ground = calculate_plane(pc)
    print("w: ", w)
    print("h: ", h) 

