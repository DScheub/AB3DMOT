import sys
import os
import cv2
from pathlib import Path

from PyQt5.QtWidgets import QApplication, QWidget, QLabel
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.Qt import Qt

from SeeingThroughFog.tools.DatasetViewer.lib.read import get_kitti_object_list
from utils.read_datastructure import generate_dense_datastructure
from dense_tracker import Tracker

DENSE = Path.home() / 'ObjectDetection/data/external/SeeingThroughFog'
STF = Path.home() / 'ObjectDetection/AB3DMOT/SeeingThroughFog'


class ImgDrawer:

    def __init__(self, dense_struct, stf_path):

        assert dense_struct

        self.image_list = dense_struct
        self.image_idx = 0
        self.num_images = len(dense_struct)

        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 1

        self.tracker = Tracker(dense_struct, stf_path)

    def _read_image(self):
        assert self.image_idx < self.num_images, f'Index {self.image_idx} is out of range. Elements in dataset: {self.num_images}'
        img_path = self.image_list[self.image_idx]['img']
        assert os.path.isfile(img_path)

        self.image = cv2.imread(img_path)

        timestamp = self.image_list[self.image_idx]['frame_id']['idx']
        cv2.putText(self.image, f'Timestep: {timestamp}', (10, 50), self.font, self.font_scale, (0, 255, 0))

        pred_label_path = self.image_list[self.image_idx]['pred_label']
        pred_labels = get_kitti_object_list(pred_label_path)
        for pred_label in pred_labels:
            self.image = self._draw_bbox_from_label(self.image, pred_label, (255, 0, 0), top_text='Score: %.2f' % pred_label['score'])

        gt_label_path = self.image_list[self.image_idx]['gt_label']
        if gt_label_path:
            gt_labels = get_kitti_object_list(gt_label_path)
            for gt_label in gt_labels:
                self.image = self._draw_bbox_from_label(self.image, gt_label, (0, 0, 255))

        tracked_objects_in_frame = self.tracker.get_tracked_objects_in_frame(self.image_idx)
        for tracked_obj in tracked_objects_in_frame:
            self.image = self._draw_bbox_from_label(self.image, tracked_obj, (0, 255, 0), bottom_text='ID: %s' % tracked_obj['id'])

    def _draw_bbox_from_label(self, image, label, bbox_color=(255, 0, 0), top_text=None, bottom_text=None, font_scale=0.5):

        if label['identity'] not in ['Car', 'PassengerCar', 'RidableVehicle']:
            return image

        x = tuple((label['xleft'], label['ytop']))
        y = tuple((label['xright'], label['ybottom']))
        cv2.rectangle(image, x, y, bbox_color, 2)

        if top_text:
            cv2.putText(self.image, top_text, (x[0], x[1] - 10), self.font, font_scale, bbox_color)
        if bottom_text:
            cv2.putText(self.image, bottom_text, (x[0], y[1] + 20), self.font, font_scale, bbox_color)

        return image

    def get_current_image(self):
        self._read_image()
        return self.image

    def get_next_image(self):
        self.image_idx = (self.image_idx + 1) % self.num_images
        self._read_image()
        return self.image

    def get_prev_image(self):
        self.image_idx = (self.image_idx - 1) % self.num_images
        self._read_image()
        return self.image


class ImagePlayer(QWidget):
    def __init__(self, root_dir: str, base_file: str, stf_path: str, past_idx=0, future_idx=0, display_width=1920, display_height=1080, player_name="Unknown"):
        super().__init__()

        # Window settings
        self.title = f'Image Player - {player_name}'
        self.setWindowTitle(f'Image Player - {player_name}')
        self.display_width = display_width
        self.display_height = display_height
        self.setGeometry(10, 10, self.display_width, self.display_height)
        self.label = QLabel(self)

        dense_data = generate_dense_datastructure(root_dir, base_file, past_idx, future_idx)
        self.image_drawer = ImgDrawer(dense_struct=dense_data, stf_path=stf_path)
        cv_img = self.image_drawer.get_current_image()
        self.show_picture(cv_img)

    def keyPressEvent(self, event):
        cv_img = None
        if event.key() == Qt.Key_Right:
            cv_img = self.image_drawer.get_next_image()
        elif event.key() == Qt.Key_Left:
            cv_img = self.image_drawer.get_prev_image()

        if cv_img is not None:
            self.show_picture(cv_img)

    def show_picture(self, cv_img):
        pixmap = self.convert_cv_qt(cv_img)
        self.label.setPixmap(pixmap)
        self.resize(pixmap.width(), pixmap.height())
        self.show()

    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        p = convert_to_qt_format.scaled(self.display_width, self.display_height, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)


if __name__ == "__main__":

    base_file = '2018-02-03_21-04-07_00000'

    app = QApplication(sys.argv)
    ip = ImagePlayer(DENSE, base_file, STF, past_idx=-6)
    sys.exit(app.exec_())
