
import torch

from segmentation_models_pytorch.fpn import FPN


# mobilenetv2-fpn
merge_lists = [
    ["encoder.features.0.0", "encoder.features.0.1"],
    ["encoder.features.1.conv.0.0", "encoder.features.1.conv.0.1"],
    ["encoder.features.1.conv.1", "encoder.features.1.conv.2"],

    # 2
    ["encoder.features.2.conv.0.0", "encoder.features.2.conv.0.1"],
    ["encoder.features.2.conv.1.0", "encoder.features.2.conv.1.1"],
    ["encoder.features.2.conv.2", "encoder.features.2.conv.3"],
    # 3
    ["encoder.features.3.conv.0.0", "encoder.features.3.conv.0.1"],
    ["encoder.features.3.conv.1.0", "encoder.features.3.conv.1.1"],
    ["encoder.features.3.conv.2", "encoder.features.3.conv.3"],
    # 4
    ["encoder.features.4.conv.0.0", "encoder.features.4.conv.0.1"],
    ["encoder.features.4.conv.1.0", "encoder.features.4.conv.1.1"],
    ["encoder.features.4.conv.2", "encoder.features.4.conv.3"],
    # 5
    ["encoder.features.5.conv.0.0", "encoder.features.5.conv.0.1"],
    ["encoder.features.5.conv.1.0", "encoder.features.5.conv.1.1"],
    ["encoder.features.5.conv.2", "encoder.features.5.conv.3"],
    # 6
    ["encoder.features.6.conv.0.0", "encoder.features.6.conv.0.1"],
    ["encoder.features.6.conv.1.0", "encoder.features.6.conv.1.1"],
    ["encoder.features.6.conv.2", "encoder.features.6.conv.3"],
    # 7
    ["encoder.features.7.conv.0.0", "encoder.features.7.conv.0.1"],
    ["encoder.features.7.conv.1.0", "encoder.features.7.conv.1.1"],
    ["encoder.features.7.conv.2", "encoder.features.7.conv.3"],
    # 8
    ["encoder.features.8.conv.0.0", "encoder.features.8.conv.0.1"],
    ["encoder.features.8.conv.1.0", "encoder.features.8.conv.1.1"],
    ["encoder.features.8.conv.2", "encoder.features.8.conv.3"],
    # 9
    ["encoder.features.9.conv.0.0", "encoder.features.9.conv.0.1"],
    ["encoder.features.9.conv.1.0", "encoder.features.9.conv.1.1"],
    ["encoder.features.9.conv.2", "encoder.features.9.conv.3"],
    # 10
    ["encoder.features.10.conv.0.0", "encoder.features.10.conv.0.1"],
    ["encoder.features.10.conv.1.0", "encoder.features.10.conv.1.1"],
    ["encoder.features.10.conv.2", "encoder.features.10.conv.3"],
    # 11
    ["encoder.features.11.conv.0.0", "encoder.features.11.conv.0.1"],
    ["encoder.features.11.conv.1.0", "encoder.features.11.conv.1.1"],
    ["encoder.features.11.conv.2", "encoder.features.11.conv.3"],
    # 12
    ["encoder.features.12.conv.0.0", "encoder.features.12.conv.0.1"],
    ["encoder.features.12.conv.1.0", "encoder.features.12.conv.1.1"],
    ["encoder.features.12.conv.2", "encoder.features.12.conv.3"],
    # 13
    ["encoder.features.13.conv.0.0", "encoder.features.13.conv.0.1"],
    ["encoder.features.13.conv.1.0", "encoder.features.13.conv.1.1"],
    ["encoder.features.13.conv.2", "encoder.features.13.conv.3"],
    # 14
    ["encoder.features.14.conv.0.0", "encoder.features.14.conv.0.1"],
    ["encoder.features.14.conv.1.0", "encoder.features.14.conv.1.1"],
    ["encoder.features.14.conv.2", "encoder.features.14.conv.3"],
    # 15
    ["encoder.features.15.conv.0.0", "encoder.features.15.conv.0.1"],
    ["encoder.features.15.conv.1.0", "encoder.features.15.conv.1.1"],
    ["encoder.features.15.conv.2", "encoder.features.15.conv.3"],
    # 16
    ["encoder.features.16.conv.0.0", "encoder.features.16.conv.0.1"],
    ["encoder.features.16.conv.1.0", "encoder.features.16.conv.1.1"],
    ["encoder.features.16.conv.2", "encoder.features.16.conv.3"],
    # 17
    ["encoder.features.17.conv.0.0", "encoder.features.17.conv.0.1"],
    ["encoder.features.17.conv.1.0", "encoder.features.17.conv.1.1"],
    ["encoder.features.17.conv.2", "encoder.features.17.conv.3"],
    # 18
    ["encoder.features.18.0", "encoder.features.18.1"],

]

dummy = torch.randn(1, 3, 288, 384)
inp = torch.zeros((1,3,288,384))

model = FPN(encoder_name='mobilenet_v2', encoder_weights='imagenet')
model.load_state_dict(torch.load("ckpt/model_avg.pth"))
model.eval()
model.cpu()

print(model)

model = torch.quantization.fuse_modules(model, merge_lists)
torch.onnx.export(model, dummy, f"export/model_mobilenetv2.onnx", verbose=True, opset_version=11)

print(model(inp).squeeze()[0,:50])

