# -*- coding: utf-8 -*-

"""
__author__ = xujing
__date__  = 2019-07-05
"""
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import QMainWindow

from PyQt5.QtCore import *
from PyQt5.QtWidgets import  *
from PyQt5 import *
from PyQt5.QtGui import *
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import QMainWindow
from PyQt5.QtCore import (QCoreApplication, QPropertyAnimation, QDate, QDateTime,
                            QMetaObject, QObject, QPoint, QRect, QSize, QTime, QUrl, Qt, QEvent)
#from  Ui_my_main_ui import Ui_MainWindow
from  gui import Ui_MainWindow
import sys
import cv2
from car_id_detect import *
from svm_train import *
from card_seg import *


class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)
    
    def on_pushButton_clicked(self):
        QMainWindow.showMinimized(self)
    def on_pushButton_2_clicked(self):

        sys.exit(0)
    def on_pushButton_6_clicked(self):
        try:
            self.file_dir_temp,_ = QFileDialog.getOpenFileName(self,"选择被检测的车辆","D:/")
            self.file_dir = self.file_dir_temp.replace("\\","/")
            print(self.file_dir)
            
            roi, label, color = CaridDetect(self.file_dir)
            seg_dict, _, pre = Cardseg([roi],[color],None)
            print(pre)
            
            # segment
            cv2.imwrite(os.path.join("./temp/seg_card.jpg"),roi)
            seg_img = cv2.imread("./temp/seg_card.jpg")
            seg_rows, seg_cols, seg_channels = seg_img.shape
            bytesPerLine = seg_channels * seg_cols
            cv2.cvtColor(seg_img, cv2.COLOR_BGR2RGB,seg_img)
            QImg = QImage(seg_img.data, seg_cols, seg_rows,bytesPerLine, QImage.Format_RGB888)
            self.label_2.setPixmap(QPixmap.fromImage(QImg).scaled(self.label_2.size(), 
                Qt.KeepAspectRatio, Qt.SmoothTransformation))
            # reg result
            pre.insert(2,"·")
            self.label_3.setText(" "+"".join(pre))
            # clor view
            if color == "yello":
                self.label_4.setStyleSheet("background-color: rgb(255, 255, 0);")
            elif color == "green":
                self.label_4.setStyleSheet("background-color: rgb(0, 255,0);")
            elif color == "blue":
                self.label_4.setStyleSheet("background-color: rgb(0, 0, 255);")
            else:
                self.label_4.setText("未识别出车牌颜色")
            
            frame = cv2.imread(self.file_dir)
            # cv2.rectangle(frame, (label[0],label[2]), (label[1],label[3]), (0,0,255), 2)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, 'https://github.com/DataXujing/vehicle-license-plate-recognition', (10, 10), font, 0.3, (0, 0, 255), 1)
            img_rows, img_cols, channels = frame.shape
            bytesPerLine = channels * img_cols
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB,frame)
            QImg = QImage(frame.data, img_cols, img_rows,bytesPerLine, QImage.Format_RGB888)
            self.label.setPixmap(QPixmap.fromImage(QImg).scaled(self.label.size(), 
                Qt.KeepAspectRatio, Qt.SmoothTransformation))
            QtWidgets.QApplication.processEvents()
        except Exception as e:
            QMessageBox.warning(self,"错误提示","[错误提示(请联系开发人员处理)]：\n" + str(e)+"\n或识别失败导致")
    
    
    
    def mousePressEvent(self, evt):
        self.oldPos = evt.globalPos()
    def mouseMoveEvent(self, evt):
        delta = QPoint(evt.globalPos() - self.oldPos)
        self.move(self.x() + delta.x(), self.y() + delta.y())
        self.oldPos = evt.globalPos()
    

    def on_pushButton_7_clicked(self):
        print("Load video")
        QMessageBox.information(self,"加载实时视频","The video source has not yet been tested or the service has not yet been launched!")




if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    app. processEvents()
    ui =MainWindow()
    ui.show()
    sys.exit(app.exec_())
