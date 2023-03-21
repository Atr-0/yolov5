from std_msgs.msg import String
import time
from rclpy.node import Node
import rclpy
import argparse
import os
import platform
import sys
from pathlib import Path
import numpy as np
import torch
import cv2
from utils.torch_utils import select_device, smart_inference_mode
from utils.plots import Annotator, colors, save_one_box
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from models.common import DetectMultiBackend
from models.experimental import attempt_download, attempt_load  # scoped to avoid circular import
from utils.augmentations import (Albumentations, augment_hsv, classify_albumentations, classify_transforms, copy_paste,
                                 letterbox, mixup, random_perspective)
cmd, aqujieguo, cqujieguo, dqujieguo = "", "", "", ""
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

rclpy.init()


class shibieSubscriber(Node):
    def __init__(self):
        super().__init__('minimal_subscriber')
        # 订阅者的构造函数和回调函数不需要定时器Timer，因为当收到Message时，就已经启动回调函数了
        # 注意此处不是subscriber，而是subscription
        # 数据类型，话题名，回调函数名，队列长度
        self.subscription = self.create_subscription(String, 'shibie', self.listener_callback, 1)
        self.subscription  # prevent unused variable warning

    def listener_callback(self, msg):
        self.get_logger().info('I heard: "%s"' % msg.data)  # 回调函数内容为打印msg.data里的内容
        global cmd
        cmd = msg.data


Aqu_node = Node("aqu_pub")
Aqu_node_pub = Aqu_node.create_publisher(String, "detect", 10)


def aqu_pub(zhilin):
    global Aqu_node_pub
    msg = String()
    msg.data = zhilin
    # print(zhilin)
    Aqu_node_pub.publish(msg)
    time.sleep(0.03)


def run_webcam(save_path, shibie_subscriber, img_size=640, stride=32, augment=False, visualize=False):
    if cmd == "a":
        weights = r'/home/zzb/yolov5/myModels/aqubest.pt'
    else:
        weights = r'/home/zzb/yolov5/myModels/cqu.pt'
    device = 'cpu'
    w = str(weights[0] if isinstance(weights, list) else weights)
    # 导入模型
    model = attempt_load(weights if isinstance(weights, list) else w, device=device, inplace=True, fuse=False)
    img_size = check_img_size(img_size, s=stride)
    names = model.names

    # 读取视频对象: 0 表示打开本地摄像头
    cap = cv2.VideoCapture(4)

    # 获取当前视频的帧率与宽高，设置同样的格式，以确保相同帧率与宽高的视频输出
    ret_val, img0 = cap.read()
    fps, w, h = 30, int(cap.get(3)), int(cap.get(4))

    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter.fourcc('m', 'p', '4', 'v'), fps, (w, h))
    # 按q退出循环
    while True:
        ret_val, img0 = cap.read()
        wait = cv2.waitKey(30)
        if wait == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
        if not ret_val:
            break
        rclpy.spin_once(shibie_subscriber, timeout_sec=0.1)
        rclpy.spin_once(shibie_subscriber, timeout_sec=0.1)
        global cmd, aqujieguo, cqujieguo, dqujieguo

        if cmd == "n":
            print("Amode jieshu")
            vid_writer.release()
            cap.release()
            break
        else:
            aqujieguo, cqujieguo, dqujieguo = "", "", ""
            # Padded resize
            img = letterbox(img0, img_size, stride=stride, auto=True)[0]

            # Convert
            img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
            img = np.ascontiguousarray(img)

            img = torch.from_numpy(img).to(device)
            img = img.float() / 255.0   # 0 - 255 to 0.0 - 1.0
            img = img[None]     # [h w c] -> [1 h w c]

            # inference
            pred = model(img, augment=augment, visualize=visualize)[0]
            pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45, max_det=1000)

            # plot labels
            det = pred[0]
            annotator = Annotator(img0.copy(), line_width=3, example=str(names))
            if len(det):
                det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], img0.shape).round()
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)  # integer class
                    if conf > 0.7:
                        if cmd == "a":
                            if xyxy[1] > 0.7:
                                aqujieguo = aqujieguo + str(c)
                            else:
                                aqujieguo = str(c) + aqujieguo
                            label = f'{names[c]} {conf:.2f}'
                            annotator.box_label(xyxy, label, color=colors(c, True))
                        elif cmd == "c":
                            if xyxy[1] > 0.7:
                                cqujieguo = cqujieguo + "0"
                            else:
                                cqujieguo = "1" + cqujieguo
                            label = f'{names[c]} {conf:.2f}'
                            annotator.box_label(xyxy, label, color=colors(c, True))

                            # write video
            im0 = annotator.result()
            aqu_pub(aqujieguo)
            cv2.imshow('webcam:0', im0)
            cv2.waitKey(1)
            vid_writer.write(im0)

    # 按q退出循环
    vid_writer.release()
    cap.release()
    # cv2.destroyAllWindows()


def main(args=None):
    # rclpy.init()
    shibie_subscriber = shibieSubscriber()
    global cmd
    while rclpy.ok():
        rclpy.spin_once(shibie_subscriber, timeout_sec=0.1)
        aqu_pub(aqujieguo)
        if cmd == "a" or cmd == "c" or cmd == "d":
            run_webcam("/home/zzb/yolov5/test.mp4", shibie_subscriber)
        if cmd == "f":
            break
    time.sleep(0.1)
    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    shibie_subscriber.destroy_node()
    Aqu_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
