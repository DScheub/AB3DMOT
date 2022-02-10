import sys
import os
import cv2
import numpy as np
import matplotlib.cm as cm
import matplotlib as mpl
from pathlib import Path
from utils.planes import ObjectAnchor 

from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QMainWindow, QGridLayout, QDesktopWidget, QPushButton
from PyQt5.QtGui import QPixmap, QImage, QVector3D
from PyQt5.Qt import Qt
import pyqtgraph.opengl as gl

from pcdet.utils.box_utils import boxes3d_kitti_camera_to_imageboxes
from SeeingThroughFog.tools.DatasetViewer.lib.read import get_kitti_object_list, load_radar_points
from utils.read_datastructure import generate_indexed_datastructure, get_img_shape
from utils.dense_transforms import Calibration, get_calib_from_json
from AB3DMOT_libs.model_dense import associate_radar_to_trackers
from dense_tracker import Tracker

# DENSE = Path.home() / 'ObjectDetection/data/external/SeeingThroughFog'
# DENSE = Path.home() / 'ObjectDetection/data/external/SprayAugmentation/2019-09-11_21-15-42'

DENSE = Path.home() / 'ObjectDetection/data/external/SprayAugmentation/2021-07-26_19-17-21'
# DENSE = '/media/dscheub/Data_Spray/2021-07-26_19-17-21'
STF = Path.home() / 'ObjectDetection/AB3DMOT/SeeingThroughFog'

RUN_TRACKER = True
USE_RADAR = True


def reproject_camera_bbox(labels, calib, img_shape):

    reprojected_label = labels.copy()

    for idx, obj in enumerate(labels):
        # box_camera [x, y, z, l, h, w, r] in rect camera coords
        box_cam = np.asarray([obj['posx'], obj['posy'], obj['posz'], obj['length'],
                              obj['height'], obj['width'], obj['orient3d']])
        box_cam = box_cam.reshape((1, -1))
        box_cam_img = boxes3d_kitti_camera_to_imageboxes(box_cam, calib, img_shape)
        reprojected_label[idx]['xleft'] = int(box_cam_img[0, 0])
        reprojected_label[idx]['ytop'] = int(box_cam_img[0, 1])
        reprojected_label[idx]['xright'] = int(box_cam_img[0, 2])
        reprojected_label[idx]['ybottom'] = int(box_cam_img[0, 3])

    return reprojected_label


class DenseDrawer:

    # RGB
    COLOR = {'pred_label': (0, 0, 255),
             'gt_label': (255, 0, 0),
             'tracked': (0, 225, 0)}

    def __init__(self, dense_struct, stf_path):

        assert dense_struct

        self.dense_data = dense_struct
        # self.dense_data.reverse()
        self.index = 0
        self.num_data = len(self.dense_data)

        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 1

        self.calib = Calibration(get_calib_from_json(STF))
        self.img_shape = get_img_shape(self.dense_data[0]['img'])
        if RUN_TRACKER:
            self.tracker = Tracker(self.dense_data, stf_path)

        self.current_image = None
        self.labels = {'pred_label': [], 'gt_label': [], 'tracked': []}
        self.radar_detections = None

        self.anchor = ObjectAnchor(self.dense_data, 'single')

    def _draw_image(self):

        img_path = self.dense_data[self.index]['img']
        assert os.path.isfile(img_path)

        self.current_image = cv2.imread(img_path)

        timestamp = self.dense_data[self.index]['frame_id']['idx']
        cv2.putText(self.current_image, f'Timestep: {timestamp}', (10, 50), self.font, self.font_scale, (0, 255, 0))

        for pred_label in self.labels['pred_label']:
            self.current_image = self._draw_bbox_from_label(self.current_image, pred_label, top_text='Score: %.2f' % pred_label['score'])

        for gt_label in self.labels['gt_label']:
            self.current_image = self._draw_bbox_from_label(self.current_image, gt_label)

        for tracked_obj in self.labels['tracked']:
            self.current_image = self._draw_bbox_from_label(self.current_image, tracked_obj, bottom_text='ID: %s' % tracked_obj['id'])

        if self.radar_detections is not None:
            pts_camera = self.calib.radar_to_rect(self.radar_detections[:, :3])
            pts_img, _ = self.calib.rect_to_img(pts_camera)
            for idx, pt in enumerate(pts_img):
                if self.radar_detections[idx, 3] > 0:
                    marker_pos = tuple((int(pt[0]), int(pt[1])))
                    cv2.drawMarker(self.current_image, marker_pos, (200, 200, 200), markerType=cv2.MARKER_STAR, markerSize=10, thickness=2)
            """
            tracker_dets = np.zeros((len(self.labels['tracked']), 3))
            for idx, obj in enumerate(self.labels['tracked']):
                tracker_dets[idx, :] = np.array([obj['posx'], obj['posy'], obj['posz']])

            if tracker_dets.size:
                matches, _, _, = associate_radar_to_trackers(pts_camera, tracker_dets, distance_threshold=2)
                for match in matches:
                    pt = pts_img[match[0], :]
                    track_id = self.labels['tracked'][int(match[1])]['id']
                    marker_pos = tuple((int(pt[0]), int(pt[1])))
                    cv2.drawMarker(self.current_image, marker_pos, (0, 255, 0), markerType=cv2.MARKER_STAR, markerSize=10, thickness=2)
                    cv2.putText(self.current_image, str(track_id), (marker_pos[0], marker_pos[1] + 20), self.font, 0.5, (0, 255, 0))
            """

        
    def _draw_bbox_from_label(self, image, label, top_text=None, bottom_text=None, font_scale=0.5):

        if label['identity'] not in ['Car', 'PassengerCar', 'RidableVehicle']:
            return image

        bbox_color_rgb = label['color']
        bbox_color = (bbox_color_rgb[2], bbox_color_rgb[1], bbox_color_rgb[0])

        x = tuple((label['xleft'], label['ytop']))
        y = tuple((label['xright'], label['ybottom']))
        cv2.rectangle(image, x, y, bbox_color, 2)

        if top_text:
            cv2.putText(image, top_text, (x[0], x[1] - 10), self.font, font_scale, bbox_color)
        if bottom_text:
            cv2.putText(image, bottom_text, (x[0], y[1] + 20), self.font, font_scale, bbox_color)

        return image

    def _update(self):

        assert self.index < self.num_data, f'Index {self.index} is out of range. Elements in dataset: {self.num_data}'
        current_frame = self.dense_data[self.index]

        self.current_image = None
        self.labels = {'pred_label': [], 'gt_label': [], 'tracked': []}
        self.radar_detections = None 

        for label_type in ['pred_label', 'gt_label']:
            label_path = current_frame[label_type]
            label_list = []
            if label_path:
                label = get_kitti_object_list(label_path, self.calib.C2V)
                for obj in label:
                    obj.update({'color': self.COLOR[label_type]})
                    label_list.append(obj)
                objects_anchor = self.anchor.anchor_object_to_ground(current_frame['pc'], label, self.calib, self.img_shape)
                for obj_anchor in objects_anchor:
                    obj_anchor.update({'color': (255, 0, 0)})
                    label_list.append(obj_anchor)
                self.labels[label_type] = label_list

        if RUN_TRACKER:
            tracked_objects_in_frame = self.tracker.get_tracked_objects_in_frame(self.index)
            tracked_list = []
            for tracked_obj in tracked_objects_in_frame:
                tracked_obj.update({'color': self.COLOR['tracked']})
                tracked_list.append(tracked_obj)
            self.labels['tracked'] = tracked_list

        if USE_RADAR and ('radar' in current_frame):
            self.radar_detections = []
            if current_frame['radar']:
                pts_radar = load_radar_points(self.dense_data[self.index]['radar'])
                pts_radar = np.asarray(pts_radar)  # [[x_sc, y_sc, 0, rVelGround, rDist], ...] (N, 5)
                self.radar_detections = pts_radar[pts_radar[:, 3] > 0, :]  # Eliminate dets with 0 velocity

        self._draw_image()

    def get_current_frame(self):
        self._update()
        labels = []
        for _, label in self.labels.items():
            labels += label

        radar_dets_lidar = None
        if self.radar_detections is not None:
            radar_dets_lidar = self.calib.radar_to_lidar(self.radar_detections[:, :3])

        return self.current_image, self.dense_data[self.index]['pc'], labels, radar_dets_lidar

    def get_next_frame(self):
        self.index = (self.index + 1) % self.num_data
        return self.get_current_frame()

    def get_prev_frame(self):
        self.index = (self.index - 1) % self.num_data
        return self.get_current_frame()


class DenseViewer(QMainWindow):

    def __init__(self, root_dir: str, base_file: str, stf_path: str, past_idx=0, future_idx=0):
        super().__init__()
        # super().keyPressEvent() = self.keyPressEvent()

        # dense_data = generate_dense_datastructure(root_dir, base_file, past_idx, future_idx)
        # dense_data = generate_indexed_datastructure(root_dir, 68, 300)
        dense_data = generate_indexed_datastructure(root_dir, 0)
        self.dense_drawer = DenseDrawer(dense_struct=dense_data, stf_path=stf_path)

        # Window settings
        self.monitor = QDesktopWidget().screenGeometry(1)
        self.setGeometry(self.monitor)
        self.showMaximized()

        self.layout = QGridLayout()
        self.widget = QWidget()
        self.widget.setLayout(self.layout)
        self.setCentralWidget(self.widget)
        self.widget.keyPressEvent = self.keyPressEvent

        self.current_row = 0

        # PC Viewer
        self.viewer = gl.GLViewWidget()
        self.grid_dimensions = 20
        self.viewer.setWindowTitle('drag & drop point cloud viewer')
        self.viewer.setCameraPosition(distance=2 * self.grid_dimensions)
        self.layout.addWidget(self.viewer, self.current_row, 0, 1, 3)
        self.grid = gl.GLGridItem()
        self.grid.setSize(self.grid_dimensions, self.grid_dimensions)
        self.grid.setSpacing(1, 1)
        self.grid.translate(0, 0, -2)
        self.viewer.addItem(self.grid)
        self.point_size = 3

        # Image
        self.img_width = 1920
        self.img_height = 1080
        self.label = QLabel(self)
        self.layout.addWidget(self.label, self.current_row, 3, 1, 3)

        self.current_row += 1

        # Buttons
        self.prev_btn = QPushButton("<-")
        self.next_btn = QPushButton("->")
        self.layout.addWidget(self.prev_btn, self.current_row, 0)
        self.layout.addWidget(self.next_btn, self.current_row, 2)
        self.prev_btn.clicked.connect(self.decrement_index)
        self.next_btn.clicked.connect(self.increment_index)

        cv_img, pc_path, labels, radar_dets = self.dense_drawer.get_current_frame()
        self.update_viewer(cv_img, pc_path, labels, radar_dets)

    def decrement_index(self):
        cv_img, pc_path, labels, radar_dets = self.dense_drawer.get_prev_frame()
        self.update_viewer(cv_img, pc_path, labels, radar_dets)

    def increment_index(self):
        cv_img, pc_path, labels, radar_dets = self.dense_drawer.get_next_frame()
        self.update_viewer(cv_img, pc_path, labels, radar_dets)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_D:
            self.increment_index()
        elif event.key() == Qt.Key_A:
            self.decrement_index()

    def update_viewer(self, cv_img, pc_path=None, label=None, radar_dets=None):
        pixmap = self.convert_cv_qt(cv_img, self.img_width, self.img_height)
        self.label.setPixmap(pixmap)

        if pc_path:
            self.viewer.items = []
            pc = np.fromfile(pc_path, dtype=np.float32)
            pc = pc.reshape((-1, 5))
            colors = self.get_pc_colors(pc)
            mesh = gl.GLScatterPlotItem(pos=np.asarray(pc[:, 0:3]), size=self.point_size, color=colors)
            self.viewer.addItem(mesh)

            if label:
                boxes = self.create_boxes(label)
                for box in boxes:
                    self.viewer.addItem(box)

            if radar_dets is not None:
                mesh_radar = gl.GLScatterPlotItem(pos=radar_dets, size=10)
                self.viewer.addItem(mesh_radar)

    def convert_cv_qt(self, cv_img, width, height):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        p = convert_to_qt_format.scaled(width, height, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)

    def get_pc_colors(self, pc):
        """ Color in z direction """
        feature = pc[:, 2]
        max_value = 0.5
        min_value = -1.5

        norm = mpl.colors.Normalize(vmin=min_value, vmax=max_value)
        m = cm.ScalarMappable(norm=norm, cmap=cm.jet)
        colors = m.to_rgba(feature)
        colors[:, [2, 1, 0, 3]] = colors[:, [0, 1, 2, 3]]
        colors[:, 3] = 0.5

        return colors

    def create_boxes(self, label):
        boxes = []
        size = QVector3D(1, 1, 1)

        for annotation in label:

            if annotation['identity'] in ['Car', 'PassengerCar', 'Pedestrian', 'RidableVehicle']:

                x = annotation['posx_lidar']
                y = annotation['posy_lidar']
                z = annotation['posz_lidar']

                box = gl.GLBoxItem(size, color=annotation['color'])
                box.setSize(annotation['length'], annotation['width'], annotation['height'])
                box.translate(-annotation['length'] / 2, -annotation['width'] / 2, -annotation['height'] / 2)
                box.rotate(angle=-annotation['rotz'] * 180 / 3.14159265359, x=0, y=0, z=1)
                box.rotate(angle=-annotation['roty'] * 180 / 3.14159265359, x=0, y=1, z=0)
                box.rotate(angle=-annotation['rotx'] * 180 / 3.14159265359, x=1, y=0, z=0)
                box.translate(0, 0, annotation['height'] / 2)
                box.translate(x, y, z)

                boxes.append(box)

        return boxes


if __name__ == "__main__":

    base_file = '2018-02-03_21-04-07_00000'

    app = QApplication(sys.argv)
    ip = DenseViewer(DENSE, base_file, STF, past_idx=-6)
    ip.show()
    sys.exit(app.exec_())
