import sys
import os
import cv2
import numpy as np
import matplotlib.cm as cm
import matplotlib as mpl
from pathlib import Path

from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QMainWindow, QGridLayout, QDesktopWidget, QPushButton
from PyQt5.QtGui import QPixmap, QImage, QVector3D
from PyQt5.Qt import Qt

import pyqtgraph.opengl as gl

from pcdet.utils.box_utils import boxes3d_kitti_camera_to_imageboxes
from SeeingThroughFog.tools.DatasetViewer.lib.read import get_kitti_object_list
from utils.read_datastructure import generate_dense_datastructure, get_img_shape
from utils.dense_transforms import Calibration, get_calib_from_json
from dense_tracker import Tracker

DENSE = Path.home() / 'ObjectDetection/data/external/SeeingThroughFog'
STF = Path.home() / 'ObjectDetection/AB3DMOT/SeeingThroughFog'


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

        self.image_list = dense_struct
        self.image_list.reverse()
        self.image_idx = 0
        self.num_images = len(self.image_list)

        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 1

        self.calib = Calibration(get_calib_from_json(STF))
        self.img_shape = get_img_shape(self.image_list[0]['img'])
        self.tracker = Tracker(self.image_list, stf_path)

        self.image = None
        self.label_list = []

    def _read_image(self):
        assert self.image_idx < self.num_images, f'Index {self.image_idx} is out of range. Elements in dataset: {self.num_images}'
        img_path = self.image_list[self.image_idx]['img']
        assert os.path.isfile(img_path)

        self.image = cv2.imread(img_path)

        timestamp = self.image_list[self.image_idx]['frame_id']['idx']
        cv2.putText(self.image, f'Timestep: {timestamp}', (10, 50), self.font, self.font_scale, (0, 255, 0))

        self.label_list = []

        pred_label_path = self.image_list[self.image_idx]['pred_label']
        pred_labels = self._get_label(pred_label_path, 'pred_label')
        for pred_label in pred_labels:
            self.image = self._draw_bbox_from_label(self.image, pred_label, top_text='Score: %.2f' % pred_label['score'])
            self.label_list.append(pred_label)

        gt_label_path = self.image_list[self.image_idx]['gt_label']
        if gt_label_path:
            gt_labels = self._get_label(gt_label_path, 'gt_label')
            gt_labels = reproject_camera_bbox(gt_labels, self.calib, self.img_shape)
            for gt_label in gt_labels:
                self.image = self._draw_bbox_from_label(self.image, gt_label)
                self.label_list.append(gt_label)

        tracked_objects_in_frame = self.tracker.get_tracked_objects_in_frame(self.image_idx)
        for tracked_obj in tracked_objects_in_frame:
            tracked_obj.update({'color': self.COLOR['tracked']})
            self.image = self._draw_bbox_from_label(self.image, tracked_obj, bottom_text='ID: %s' % tracked_obj['id'])
            self.label_list.append(tracked_obj)

    def _draw_bbox_from_label(self, image, label, top_text=None, bottom_text=None, font_scale=0.5):

        if label['identity'] not in ['Car', 'PassengerCar', 'RidableVehicle']:
            return image

        bbox_color_rgb = label['color']
        bbox_color = (bbox_color_rgb[2], bbox_color_rgb[1], bbox_color_rgb[0])

        x = tuple((label['xleft'], label['ytop']))
        y = tuple((label['xright'], label['ybottom']))
        cv2.rectangle(image, x, y, bbox_color, 2)

        if top_text:
            cv2.putText(self.image, top_text, (x[0], x[1] - 10), self.font, font_scale, bbox_color)
        if bottom_text:
            cv2.putText(self.image, bottom_text, (x[0], y[1] + 20), self.font, font_scale, bbox_color)

        return image

    def _get_label(self, label_path, label_type='pred_label'):
        label = get_kitti_object_list(label_path, self.calib.C2V)
        for obj in label:
            obj.update({'color': self.COLOR[label_type]})

        return label

    def get_current_frame(self):
        self._read_image()
        return self.image, self.image_list[self.image_idx]['pc'], self.label_list

    def get_next_frame(self):
        self.image_idx = (self.image_idx + 1) % self.num_images
        self._read_image()
        return self.image, self.image_list[self.image_idx]['pc'], self.label_list

    def get_prev_frame(self):
        self.image_idx = (self.image_idx - 1) % self.num_images
        self._read_image()
        return self.image, self.image_list[self.image_idx]['pc'], self.label_list


class DenseViewer(QMainWindow):

    def __init__(self, root_dir: str, base_file: str, stf_path: str, past_idx=0, future_idx=0):
        super().__init__()

        dense_data = generate_dense_datastructure(root_dir, base_file, past_idx, future_idx)
        self.dense_drawer = DenseDrawer(dense_struct=dense_data, stf_path=stf_path)

        # Window settings
        self.monitor = QDesktopWidget().screenGeometry(0)
        self.setGeometry(self.monitor)
        self.showMaximized()

        self.layout = QGridLayout()
        self.widget = QWidget()
        self.widget.setLayout(self.layout)
        self.setCentralWidget(self.widget)

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
        cv_img, pc_path, labels  = self.dense_drawer.get_current_frame()

        self.update_viewer(cv_img, pc_path, labels)

        self.current_row += 1

        # Buttons
        self.prev_btn = QPushButton("<-")
        self.next_btn = QPushButton("->")
        self.layout.addWidget(self.prev_btn, self.current_row, 0)
        self.layout.addWidget(self.next_btn, self.current_row, 2)
        self.prev_btn.clicked.connect(self.decrement_index)
        self.next_btn.clicked.connect(self.increment_index)

    def decrement_index(self):
        cv_img, pc_path, labels = self.dense_drawer.get_prev_frame()
        self.update_viewer(cv_img, pc_path, labels)

    def increment_index(self):
        cv_img, pc_path, labels = self.dense_drawer.get_next_frame()
        self.update_viewer(cv_img, pc_path, labels)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Right:
            self.increment_index()
        elif event.key() == Qt.Key_Left:
            self.decrement_index()


    def update_viewer(self, cv_img, pc_path=None, label=None):
        pixmap = self.convert_cv_qt(cv_img, self.img_width, self.img_height)
        self.label.setPixmap(pixmap)

        if pc_path:
            self.viewer.items = []
            pc = np.fromfile(pc_path, dtype=np.float32)
            pc = pc.reshape((-1, 5))
            colors = self.get_colors(pc)
            mesh = gl.GLScatterPlotItem(pos=np.asarray(pc[:, 0:3]), size=self.point_size, color=colors)
            self.viewer.addItem(mesh)

            if label:
                boxes = self.create_boxes(label)
                for box in boxes:
                    self.viewer.addItem(box)

    def convert_cv_qt(self, cv_img, width, height):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        p = convert_to_qt_format.scaled(width, height, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)

    def get_colors(self, pc: np.ndarray) -> np.ndarray:
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

    app = QApplication(sys.argv)
    ip = DenseViewer(DENSE, base_file, STF, past_idx=-6)
    ip.show()
    sys.exit(app.exec_())
