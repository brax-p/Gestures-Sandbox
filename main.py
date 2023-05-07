import dearpygui.dearpygui as dpg
import numpy as np
import cv2
from typing import Tuple
import tvm
import tvm.relay
import time
import torch
import torch.onnx
from PIL import Image
from dataclasses import dataclass
from helpers import *



@dataclass
class Application:
    SOFTMAX_THRES = 0
    HISTORY_LOGIT = True
    REFINE_OUTPUT = True

    catigories = [
        "Doing other things",  # 0
        "Drumming Fingers",  # 1
        "No gesture",  # 2
        "Pulling Hand In",  # 3
        "Pulling Two Fingers In",  # 4
        "Pushing Hand Away",  # 5
        "Pushing Two Fingers Away",  # 6
        "Rolling Hand Backward",  # 7
        "Rolling Hand Forward",  # 8
        "Shaking Hand",  # 9
        "Sliding Two Fingers Down",  # 10
        "Sliding Two Fingers Left",  # 11
        "Sliding Two Fingers Right",  # 12
        "Sliding Two Fingers Up",  # 13
        "Stop Sign",  # 14
        "Swiping Down",  # 15
        "Swiping Left",  # 16
        "Swiping Right",  # 17
        "Swiping Up",  # 18
        "Thumb Down",  # 19
        "Thumb Up",  # 20
        "Turning Hand Clockwise",  # 21
        "Turning Hand Counterclockwise",  # 22
        "Zooming In With Full Hand",  # 23
        "Zooming In With Two Fingers",  # 24
        "Zooming Out With Full Hand",  # 25
        "Zooming Out With Two Fingers"  # 26
    ]

    n_still_frame = 0

    def process_output(self, idx_, history):
        # idx_: the output of current frame
        # history: a list containing the history of predictions
        if not self.REFINE_OUTPUT:
            return idx_, history

        max_hist_len = 20  # max history buffer

        # mask out illegal action
        if idx_ in [7, 8, 21, 22, 3]:
            idx_ = history[-1]

        # use only single no action class
        if idx_ == 0:
            idx_ = 2
        # history smoothing
        if idx_ != history[-1]:
            if not (history[-1] == history[-2]): #  and history[-2] == history[-3]):
                idx_ = history[-1]

        history.append(idx_)
        history = history[-max_hist_len:]

        return history[-1], history

    def run(self):
        dpg.create_context()
        dpg.create_viewport(title='Action Recognition', width=1600, height=720)
        dpg.setup_dearpygui()

        vid = cv2.VideoCapture(0)
        vid.set(cv2.CAP_PROP_FPS, 60)
        _, frame = vid.read()

        # image size or you can get this from image shape
        data = np.flip(frame, 2)  # because the camera data comes in as BGR and we need RGB
        data = data.ravel()  # flatten camera data to a 1 d stricture
        data = np.asfarray(data, dtype='f')  # change data type to 32bit floats
        texture_data = np.true_divide(data, 255.0)  # normalize image data to prepare for GPU

        t = None
        index = 0
        print("Build transformer...")
        transform = get_transform()
        print("Build Executor...")
        executor = get_executor()
        buffer = (
            tvm.nd.empty((1, 3, 56, 56)),
            tvm.nd.empty((1, 4, 28, 28)),
            tvm.nd.empty((1, 4, 28, 28)),
            tvm.nd.empty((1, 8, 14, 14)),
            tvm.nd.empty((1, 8, 14, 14)),
            tvm.nd.empty((1, 8, 14, 14)),
            tvm.nd.empty((1, 12, 14, 14)),
            tvm.nd.empty((1, 12, 14, 14)),
            tvm.nd.empty((1, 20, 7, 7)),
            tvm.nd.empty((1, 20, 7, 7))
        )
        idx = 0
        history = [2, 2, 2, 2]
        history_logit = []

        i_frame = -1

        with dpg.texture_registry(show=False):
            dpg.add_raw_texture(frame.shape[1], frame.shape[0], texture_data, tag="texture_tag", format=dpg.mvFormat_Float_rgb)

        with dpg.window(label="Example Window"):
            dpg.add_text("Hello, world")
            dpg.add_image("texture_tag")

        dpg.show_metrics()
        dpg.show_viewport()
        while dpg.is_dearpygui_running():
            i_frame += 1

            # updating the texture in a while loop the frame rate will be limited to the camera frame rate.
            # commenting out the "ret, frame = vid.read()" line will show the full speed that operations and updating a texture can run at
            if i_frame % 2 == 0:
                continue
            _, img = vid.read()
            data = np.flip(img, 2)
            data = data.ravel()
            data = np.asfarray(data, dtype='f')
            texture_data = np.true_divide(data, 255.0)
            dpg.set_value("texture_tag", texture_data)
            img_tran = transform([Image.fromarray(img).convert('RGB')])
            input_var = torch.autograd.Variable(img_tran.view(1, 3, img_tran.size(1), img_tran.size(2)))
            img_nd = tvm.nd.array(input_var.detach().numpy())
            inputs: Tuple[tvm.nd.NDArray] = (img_nd,) + buffer
            outputs = executor(inputs)
            feat, buffer = outputs[0], outputs[1:]
            assert isinstance(feat, tvm.nd.NDArray)
            if self.SOFTMAX_THRES > 0:
                feat_np = feat.asnumpy().reshape(-1)
                feat_np -= feat_np.max()
                softmax = np.exp(feat_np) / np.sum(np.exp(feat_np))

                print(max(softmax))
                if max(softmax) > self.SOFTMAX_THRES:
                    idx_ = np.argmax(feat.asnumpy(), axis=1)[0]
                else:
                    idx_ = idx
            else:
                idx_ = np.argmax(feat.asnumpy(), axis=1)[0]

            if self.HISTORY_LOGIT:
                history_logit.append(feat.asnumpy())
                history_logit = history_logit[-12:]
                avg_logit = sum(history_logit)
                idx_ = np.argmax(avg_logit, axis=1)[0]

            idx, history = self.process_output(idx_, history)

            t2 = time.time()
            print(f"{index} {self.catigories[idx]}")

            if t is None:
                    t = time.time()
            else:
                nt = time.time()
                index += 1
                t = nt
            # to compare to the base example in the open cv tutorials uncomment below
            #cv.imshow('frame', frame)

            dpg.render_dearpygui_frame()

        vid.release()
        #cv.destroyAllWindows() # when using upen cv window "imshow" call this also
        dpg.destroy_context() 

def main():
    app = Application()
    app.run()

main()
