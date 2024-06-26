import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from torch.utils.data import DataLoader

from data_prepare.mydataset import MyDataset
from utils.confusion_matrix import ConfusionMatrix

from model import resnet_model, alex_model, densenet
from model import efficientnet, googlenet, mobilenet, vgg_model, regnet, shufflenet, convnext
from model import vision_transformer, swin_transformer

# 设置随机数种子，确保结果可重复
torch.manual_seed(1)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"using {device} device.")

test_txt_path = os.path.join("..", "test.txt")

# 数据预处理设置
normMean = [0.67254436, 0.639278, 0.6043134]
normStd = [0.208332, 0.2092541, 0.2310524]
normTransform = transforms.Normalize(normMean, normStd)

# 测试集的预处理
test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normTransform
])

batch_size = 16

test_dataset = MyDataset(test_txt_path, test_transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

class_label = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
confusion = ConfusionMatrix(num_classes=6, class_labels=class_label)

# 构建模型
# net = alex_model.AlexNet(num_classes=6)
# pretrain
# net = models.alexnet(num_classes=6)
# net = models.vgg19(num_classes=6)
# net = models.resnet152(num_classes=6)

# model_name = 'vgg11'
# net = vgg_model.vgg(model_name=model_name, num_classes=6)

# net = resnet_model.resnet50(num_classes=6)
# net = mobilenet.mobilenet_v3_small(num_classes=6)
# net = mobilenet.mobilenet_v3_large(num_classes=6)
# net = regnet.regnet(num_classes=6)
# net = shufflenet.shufflenet_v2_x0_5(num_classes=6)
# net = shufflenet.shufflenet_v2_x1_0(num_classes=6)
# net = efficientnet.efficientnet_b5(num_classes=6)
# net = efficientnet.efficientnetv2_s(num_classes=6)
# net = convnext.convnext_tiny(num_classes=6)
# net = vision_transformer.vit_base_patch16_224(num_classes=6)
net = swin_transformer.swin_tiny_patch4_window7_224(num_classes=6)

net.to(device)

# weights_path = "./AlexNet.pth"
# weights_path = f"./save_weight/{model_name}.pth"
# weights_path = 'save_weight/resnet50.pth'
# weights_path = 'save_weight/pretrain_vgg19.pth'
# weights_path = 'save_weight/pretrain_resnet152.pth'

# weights_path = 'save_weight/mobilenet_v3_small.pth'
# weights_path = 'save_weight/mobilenet_v3_large.pth'
# weights_path = 'save_weight/regnet.pth'
# weights_path = 'save_weight/shufflenet_v2_x0_5.pth'
# weights_path = 'save_weight/shufflenet_v2_x1_0.pth'
# weights_path = 'save_weight/efficientnet_b5.pth'
# weights_path = 'save_weight/efficientnetv2_s.pth'
# weights_path = 'save_weight/convnext_tiny.pth'
# weights_path = 'save_weight/vit_base_patch16_224.pth'
weights_path = 'save_weight/swin_tiny_patch4_window7_224.pth'

assert os.path.exists(weights_path), f"file: '{weights_path}' dose not exist."
net.load_state_dict(torch.load(weights_path))

net.eval()  # 去掉dropout，batch normalization使用全局均值和标准差
with torch.no_grad():
    for test_data in test_loader:
        test_images, test_labels = test_data
        outputs = net(test_images.to(device))
        outputs = torch.softmax(outputs, dim=1)
        outputs = torch.argmax(outputs, dim=1)
        confusion.update(outputs.cpu().numpy(), test_labels.cpu().numpy())
    confusion.summary()
    # confusion.plot()
