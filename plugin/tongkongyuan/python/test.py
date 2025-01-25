import cv2
import argparse
from segmentation_models_pytorch.fpn import FPN



from utils import *


def mask_color(mask: np.ndarray, thre=0.5, color=(255, 0, 0)):
    # mask [h,w]
    mask = mask[..., None].repeat(3, axis=-1)
    color = np.where(mask > thre, color, 0)
    return color.astype(np.uint8)

def normalize(image):
    image = image.astype(np.float32)
    image = image / 255.
    image -= [0.485, 0.456, 0.406]
    image /= [0.229, 0.224, 0.225]
    return np.transpose(image, axes=[2, 0, 1])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type=str, default="",
                        required=True, help="video path")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    torch.set_grad_enabled(False)

    model = FPN(encoder_name='mobilenet_v2', encoder_weights='imagenet')
    model.load_state_dict(torch.load("ckpt/model_avg.pth"))
    model.to(Config.device)
    model.eval()


    cap = cv2.VideoCapture(f"{args.video_path}")

    x_f, y_f = 0, 0 # EMA

    while (True):
        ret, frame = cap.read()  # 读取帧
        if not ret:
            break
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # 灰度化展示
        aug = Config.valid_transforms(image=image)
        image = aug['image']
        image = image[:, 64:448, :]

        image_torch = torch.from_numpy(normalize(image))[None, ...]
        mask_output = model(image_torch.cuda())
        mask_output = torch.sigmoid(mask_output).squeeze().detach().cpu().numpy()



        mask = mask_color(mask_output, thre=0.3, color=(0,255,0))
        image_show = image.copy()
        image_show = cv2.addWeighted(image_show, 1.0, mask, 0.7, 0)

        M = cv2.moments((mask_output>0.3).astype(np.uint8))
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            if x_f == 0:
                x_f = cx
                y_f = cy
            else:
                x_f = int(x_f*0.5 + cx*0.5)
                y_f = int(y_f * 0.5 + cy * 0.5)

            image_show = cv2.circle(image_show, center=(cx, cy), radius=4, color=(255, 0, 0), thickness=-1)

        result = cv2.cvtColor(image_show, cv2.COLOR_RGB2BGR)
        cv2.imshow('frame', result)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # 按‘q’退出
            break

    # 释放资源并关闭窗口
    cap.release()
    cv2.destroyAllWindows()






