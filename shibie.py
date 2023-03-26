from utils.torch_utils import select_device, smart_inference_mode
from utils.plots import Annotator, colors, save_one_box
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from models.common import DetectMultiBackend
from models.experimental import attempt_download, attempt_load  # scoped to avoid circular import
from utils.augmentations import (Albumentations, augment_hsv, classify_albumentations, classify_transforms, copy_paste,
                                 letterbox, mixup, random_perspective)
from image_similarity import *
from typing import List
import cv2
import simcal
import torch
import numpy as np
from pathlib import Path
from keras.applications.imagenet_utils import preprocess_input
from keras.utils import image_utils as image
import sys
import platform
import os
import argparse
import rclpy
from rclpy.node import Node
import time
from std_msgs.msg import String

cmd, jieguo, = "", "",
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
    global cmd
    if cmd == "a":
        weights = r'/home/zzb/yolov5/myModels/aqubest.pt'
    elif cmd == "c":
        weights = r'/home/zzb/yolov5/myModels/cqu.pt'
    elif cmd == "d":
        weights = r'/home/zzb/yolov5/myModels/dqu.pt'
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
        global jieguo

        if cmd == "n":
            print("Amode jieshu")
            vid_writer.release()
            cap.release()
            break
        elif cmd == "a" or cmd == "c" or cmd == "d":
            jieguo = ""
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
                    if conf > 0.72:
                        if cmd == "a":
                            print(xyxy)
                            if xyxy[3] < 300:
                                # 上
                                jieguo = str(c) + jieguo
                            else:
                                # 下
                                jieguo = jieguo + str(c)
                            label = f'{names[c]} {conf:.2f}'
                            annotator.box_label(xyxy, label, color=colors(c, True))
                        elif cmd == "c":
                            print(xyxy)
                            if xyxy[3] < 300:
                                # 上
                                jieguo = "0" + jieguo
                            else:
                                # 下
                                jieguo = jieguo + "1"
                            label = f'{names[c]} {conf:.2f}'
                            annotator.box_label(xyxy, label, color=colors(c, True))
                        elif cmd == "d":
                            print(xyxy)
                            if xyxy[3] < 300:
                                # 上
                                jieguo = str((c + 1) + 10) + jieguo
                            else:
                                # 下
                                jieguo = jieguo + str((c + 1) + 20)
                            label = f'{names[c]} {conf:.2f}'
                            annotator.box_label(xyxy, label, color=colors(c, True))

                            # write video
                time.sleep(0.5)
            im0 = annotator.result()
            aqu_pub(jieguo)
            cv2.imshow('webcam:0', im0)
            cv2.waitKey(1)
            vid_writer.write(im0)
            cmd = "n"

    # 按q退出循环
    vid_writer.release()
    cap.release()
    # cv2.destroyAllWindows()


def ssim(img1, img2):
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(img1, img2):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')


def get_score(image1_rgb, image2_rgb, naed, ssim, cosine):
    # Determine the divisor for total percent calculation based on how many metrics are selected
    percent_proportion = 100.0 / sum([naed, ssim, cosine])
    total_percent = 0.0

    if naed:
        # Get euclidean distance between both neighbor-averaged images (requires grayscale images)
        ref_img_gray = cv2.cvtColor(np.float32(image1_rgb), cv2.COLOR_BGR2GRAY)
        target_img_gray = cv2.cvtColor(np.float32(image2_rgb), cv2.COLOR_BGR2GRAY)
        l2_na_value = get_L2_norm_neighbor_avg(image1_rgb, image2_rgb)
        l2_percent_of_img_size = l2_na_value / ((IMG_SIZE[0] * IMG_SIZE[1]) / 255)  # smaller percentage is better
        # convert the small percentage to a large percentage equivalent
        normalized_l2_percent = 100 - (l2_percent_of_img_size * 100)

        naed_percent = (((((l2_percent_of_img_size) - L2_NA_THRESHOLD) * 100) /
                        (0.0 - L2_NA_THRESHOLD)) * percent_proportion) / 100
        total_percent += naed_percent

    if ssim:
        # Get SSIM value
        ssim_value = get_ssim(image1_rgb, image2_rgb)
        ssim_percent = ((((ssim_value - 0.0) * 100) / (1.0 - 0.0)) * percent_proportion) / 100
        total_percent += ssim_percent

    if cosine:
        # Flatten to 1-D for cosine similarity
        image1_flattened = image1_rgb.flatten()
        image2_flattened = image2_rgb.flatten()
        cosine_sim_value = get_cosine_similarity(image1_flattened, image2_flattened)

        cosine_sim_percent = ((((cosine_sim_value - 0.0) * 100) / (1.0 - 0.0)) * percent_proportion) / 100
        total_percent += cosine_sim_percent

    return total_percent


def run_bqun(save_path, shibie_subscriber, img_size=640, stride=32, augment=False, visualize=False):
    global cmd, jieguo
    weights = r'/home/zzb/yolov5/myModels/bqu.pt'
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
        # cv2.imshow('web', img0)
        cv2.waitKey(1)
        if not ret_val:
            print("xxxxxxxxxxxxxxxxxxxxx")
            break
        rclpy.spin_once(shibie_subscriber, timeout_sec=0.05)
        # print(f'video {frame} {save_path}')
        if cmd == "b":
            jieguo = ""
            # if cmd=="y":
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

            # plot label
            det = pred[0]
            annotator = Annotator(img0.copy(), line_width=3, example=str(names))
            gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            up = []
            down = []
            if len(det):
                det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], img0.shape).round()
                for *xyxy, conf, cls in reversed(det):
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    c = int(cls)  # integer class
                    if conf > 0.6:
                        if xyxy[3] < 300:
                            up.append(xyxy)
                        else:
                            down.append(xyxy)
                        label = f'{names[c]} {conf:.2f}' + str(xyxy[1])
                        annotator.box_label(xyxy, label, color=colors(c, True))
            im0 = annotator.result()
            cv2.imshow('webcam:0', im0)
            up.sort(key=lambda x: x[0])
            img_data_list = []
            if len(up) == 3:
                print(up)
                tmp0 = img0[int(up[0][1]):int(up[0][3]), int(up[0][0]):int(up[0][2])]
                for i in range(3):
                    tmp0 = img0[int(up[i][1]):int(up[i][3]), int(up[i][0]):int(up[i][2])]
                    cv2.imwrite("/home/zzb/yolov5/u" + str(i) + str(1) + ".jpg", tmp0)
                    tmp0 = cv2.resize(tmp0, (224, 224), interpolation=cv2.INTER_AREA)
                    x = image.img_to_array(tmp0)
                    x = np.expand_dims(x, axis=0)
                    x = preprocess_input(x)
                    img_data_list.append(x)
                simcal.SIM(img_data_list[0], img_data_list[1])
                yingshe = {2: 1, 0: 0, 1: 2}
                simlist = []
                for i in range(2, -1, -1):
                    xiangsidu = simcal.SIM(img_data_list[i], img_data_list[i - 1])
                    simlist.append(xiangsidu)
                print(simlist)
                if max(simlist) - min(simlist) > 10:
                    print("shangcengyiwu" + str(yingshe[simlist.index(max(simlist))]))
                    jieguo = "3" + str(yingshe[simlist.index(max(simlist))]) + jieguo
                else:
                    print("shangchengzhengchang")
            down.sort(key=lambda x: x[0])
            img_data_list = []
            if len(down) == 3:
                print(down)
                tmp0 = img0[int(down[0][1]):int(down[0][3]), int(down[0][0]):int(down[0][2])]

                for i in range(3):
                    tmp0 = img0[int(down[i][1]):int(down[i][3]), int(down[i][0]):int(down[i][2])]
                    cv2.imwrite("/home/zzb/yolov5/d" + str(i) + str(1) + ".jpg", tmp0)
                    tmp0 = cv2.resize(tmp0, (224, 224), interpolation=cv2.INTER_AREA)
                    x = image.img_to_array(tmp0)
                    x = np.expand_dims(x, axis=0)
                    x = preprocess_input(x)
                    img_data_list.append(x)
                simcal.SIM(img_data_list[0], img_data_list[1])
                yingshe = {2: 1, 0: 0, 1: 2}
                simlist = []
                for i in range(2, -1, -1):
                    xiangsidu = simcal.SIM(img_data_list[i], img_data_list[i - 1])
                    simlist.append(xiangsidu)
                print(simlist)
                if max(simlist) - min(simlist) > 10:
                    print("xiacengyiwu" + str(yingshe[simlist.index(max(simlist))]))
                    jieguo = jieguo + str(yingshe[simlist.index(max(simlist))]) + "4"
                else:
                    print("xiacengzhengchang")
            time.sleep(1)
            # write video

            aqu_pub(jieguo)
            # Aqu_Publisher.pub(Aqujieguo)

            cv2.waitKey(1)
            cmd = "n"
            # vid_writer.write(im0)
        if cmd == "n":
            print("Amode jieshu")
            break
    # 按q退出循环
    vid_writer.release()
    cap.release()


def main(args=None):
    # rclpy.init()
    shibie_subscriber = shibieSubscriber()
    global cmd

    while rclpy.ok():
        rclpy.spin_once(shibie_subscriber, timeout_sec=0.1)
        aqu_pub(jieguo)
        if cmd == "b":
            run_bqun("/home/zzb/yolov5/test.mp4", shibie_subscriber)
        if cmd == "a" or cmd == "c" or cmd == "d":
            run_webcam("/home/zzb/yolov5/test.mp4", shibie_subscriber)
        if cmd == "f":
            break
    time.sleep(0.1)
    # run_webcam("/home/zzb/yolov5/test.mp4", shibie_subscriber)
    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    shibie_subscriber.destroy_node()
    Aqu_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
