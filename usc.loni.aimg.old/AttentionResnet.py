import argparse
import copy
import os
import sys
if not sys.warnoptions:
    import warnings
    warnings.simplefilter('ignore')
import time
from tqdm import tqdm
from collections import OrderedDict
import numpy as np
import pandas as pd
import torch
torch.backends.cudnn.enabled = True
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torchvision.datasets import VisionDataset
import nibabel as nib

def has_file_allowed_extension(filename, extensions):
    return filename.endswith(extensions)

def make_dataset(dir, class_to_idx, extensions=None, is_valid_file=None):
    images = OrderedDict({})
    dir = os.path.expanduser(dir)
    if not (extensions is None) ^ (is_valid_file is None):
        raise ValueError('Both extensions and is_valid_file cannot be None or not None at the same time')
    if extensions is not None:
        def is_valid_file(x):
            return has_file_allowed_extension(x, extensions)
    for target in sorted(class_to_idx.keys()):
        d = os.path.join(dir, str(target).split(',')[0])
        if not os.path.isdir(d):
            pass
        else:
            for root, _, fnames in sorted(os.walk(d)):
                for fname in sorted(fnames):
                    path = os.path.join(root, fname)
                    if is_valid_file(path):
                        if '/'.join(path.split('/')[:-1]) in images:
                            images['/'.join(path.split('/')[:-1])].append((path, class_to_idx[target]))
                        else:
                            images['/'.join(path.split('/')[:-1])] = [(path, class_to_idx[target])]
    return images

class DatasetFolder(VisionDataset):
    def __init__(self, root, csv_dir, loader, file_type=None, filename_label=None, class_label=None, extensions=None, transform=None, target_transform=None, is_valid_file=None, common=64, demographic=None):
        super(DatasetFolder, self).__init__(root)
        self.transform = transform
        self.common = common
        self.filename_label = filename_label
        self.class_label = class_label
        self.target_transform = target_transform
        self.extensions = file_type
        self.demographic = demographic
        if len(self.demographic) > 0:
            classes, class_to_idx, class_to_demo = self._find_classes(csv_dir, root)
        else:
            classes, class_to_idx = self._find_classes(csv_dir, root)
        print('CSV Dir', csv_dir)
        print('Root Dir', root)
        samples = make_dataset(self.root, class_to_idx, self.extensions, is_valid_file)
        print('length of samples', samples)
        if len(samples) == 0:
            raise RuntimeError('Found 0 files in subfolders of: ' + self.root + '\nSupported extensions are: ' + ','.join(extensions))
        self.loader = loader
        self.extensions = extensions
        self.filename_label = filename_label
        self.class_label = class_label
        if len(self.demographic) > 0:
            self.class_to_demo = class_to_demo
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[0][1] for s in samples.values()]
    def _find_classes(self, csv_dir, root):
        df = pd.read_csv(csv_dir)
        df = df.fillna(0)
        df['cat_label'] = df[self.class_label].astype('category')
        df['codes'] = df['cat_label'].cat.codes
        classes = list(df['codes'].unique())
        classes = [i for i in range(len(classes))]
        class_to_idx = {}
        class_to_demo = {}
        for _, row in df.iterrows():
            if str(row[self.filename_label]) in os.listdir(root):
                class_to_idx[row[self.filename_label]] = row['codes']
                if self.demographic:
                    class_to_demo[row[self.filename_label]] = row[self.demographic].values
        if self.demographic:
            return (classes, class_to_idx, class_to_demo)
        else:
            return (classes, class_to_idx)
    def __getitem__(self, index):
        key = list(self.samples.keys())[index]
        all_data = self.samples[key]
        combined = None
        subject = key.split('/')[(-1)]
        if len(self.demographic) > 0:
            demographic_data = torch.from_numpy(np.asarray(self.class_to_demo[subject]).astype(np.float32))
        for path1, target in all_data:
            sample1 = np.float32(nii_loader(path1))
            if self.transform is not None:
                sample1 = self.transform(sample1)
                sample = F.interpolate(sample1.view((1, 1) + sample1.shape), size=self.common, mode='trilinear').view(1, self.common, self.common, self.common)
                if combined is not None:
                    combined = torch.cat((combined, sample), 0)
                else:
                    combined = sample
                if self.target_transform is not None:
                    target = self.target_transform(target)
        if len(self.demographic) > 0:
            return (combined, target, demographic_data, subject)
        else:
            return (combined, target, subject)
    def __len__(self):
        return len(self.samples)

IMG_EXTENSIONS = '.gz'

def nii_loader(path):
    img = nib.load(path)
    img = np.array(img.dataobj)
    return img

class ImageFolder(DatasetFolder):
    def __init__(self, root, csv_dir, transform=None, target_transform=None, loader=nii_loader, is_valid_file=None, filename_label=None, class_label=None, file_type=None, common=64, demographic=[]):
        super(ImageFolder, self).__init__(root, csv_dir, loader, (IMG_EXTENSIONS if file_type is None else file_type), filename_label=filename_label, class_label=class_label, transform=transform,
          target_transform=target_transform,
          is_valid_file=is_valid_file,
          demographic=demographic)
        self.common = common
        self.imgs = self.samples

def train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, device, num_epochs=20, demographic=[]):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    for epoch in tqdm(range(num_epochs)):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('----------')
        for phase in ('train', 'val'):
            if phase == 'train':
                model.train()
            else:
                model.eval()
            running_loss = 0.0
            running_corrects = 0
            if len(demographic) > 0:
                for inputs, labels, dem, subject in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.long().to(device)
                    dem = dem.to(device)
                    optimizer.zero_grad()
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs, dem)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
                            scheduler.step()
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
            else:
                for inputs, labels, subject in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.long().to(device)
                    optimizer.zero_grad()
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
                            scheduler.step()
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            print('{} Classification Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    model.load_state_dict(best_model_wts)
    return model

def predict_model(model, criterion, dataloaders, dataset_sizes, device, demographic=[]):
    since = time.time()
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    phase = 'test'
    if len(demographic) > 0:
        for inputs, labels, dem, subject in dataloaders[phase]:
            inputs = inputs.to(device)
            labels = labels.long().to(device)
            dem = dem.to(device)
            outputs = model(inputs, dem)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            for i in range(len(preds)):
                print(subject[i], int(preds[i].cpu().numpy()))
    else:
        for inputs, labels, subject in dataloaders[phase]:
            inputs = inputs.to(device)
            labels = labels.long().to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            for i in range(len(preds)):
                print(subject[i], int(preds[i].cpu().numpy()))
    epoch_loss = running_loss / dataset_sizes[phase]
    epoch_acc = running_corrects.double() / dataset_sizes[phase]
    print('{} Classification Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
    time_elapsed = time.time() - since
    print('Testing complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

class UpsamplingBilinear3d(nn.modules.Upsample):
    def __init__(self, size=None):
        super(UpsamplingBilinear3d, self).__init__(size, mode='trilinear', align_corners=True)

def conv1x1(in_planes, out_planes, stride=1, groups=1, bias=False):
    return nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride, bias=bias, groups=groups)

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1, bias=False):
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, groups=groups, bias=bias, dilation=dilation)

class ResidualModule(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1, norm_layer=nn.BatchNorm3d, expansion=1):
        super(ResidualModule, self).__init__()
        self.expansion = expansion
        width = int(planes * (base_width / 64.0)) * groups
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class AttentionModule(nn.Module):
    def __init__(self, inplanes, planes, norm_layer=nn.BatchNorm3d, stages=4, size1=32, size2=16, size3=8, size4=4):
        super(AttentionModule, self).__init__()
        self.stages = stages
        self.first_residual_blocks = ResidualModule(inplanes, planes)
        self.trunk_branches = nn.Sequential(
            ResidualModule(inplanes, planes),
            ResidualModule(inplanes, planes)
        )
        self.mpool1 = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.softmax1_blocks = ResidualModule(inplanes, planes)
        self.skip1_connection_residual_block = ResidualModule(inplanes, planes)
        self.mpool2 = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.softmax2_blocks = ResidualModule(inplanes, planes)
        self.skip2_connection_residual_block = ResidualModule(inplanes, planes)
        self.mpool3 = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.softmax3_blocks = ResidualModule(inplanes, planes)
        self.skip3_connection_residual_block = ResidualModule(inplanes, planes)
        self.mpool4 = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.softmax4_blocks = nn.Sequential(
            ResidualModule(inplanes, planes),
            ResidualModule(inplanes, planes)
        )
        self.interpolation4 = UpsamplingBilinear3d(size=size4)
        self.softmax5_blocks = ResidualModule(inplanes, planes)
        self.interpolation3 = UpsamplingBilinear3d(size=size3)
        self.softmax6_blocks = ResidualModule(inplanes, planes)
        self.interpolation2 = UpsamplingBilinear3d(size=size2)
        self.softmax7_blocks = ResidualModule(inplanes, planes)
        self.interpolation1 = UpsamplingBilinear3d(size=size1)
        self.softmax8_blocks = nn.Sequential(
            norm_layer(planes),
            nn.ReLU(inplace=True),
            conv1x1(planes, planes),
            norm_layer(planes),
            nn.ReLU(inplace=True),
            conv1x1(planes, planes),
            nn.Sigmoid()
        )
        self.last_blocks = ResidualModule(inplanes, planes)
    def forward(self, x):
        x = self.first_residual_blocks(x)
        out_trunk = self.trunk_branches(x)
        out_mpool1 = self.mpool1(x)
        out_softmax1 = self.softmax1_blocks(out_mpool1)
        out_skip1_connection = self.skip1_connection_residual_block(out_softmax1)
        x = out_softmax1
        if self.stages < 3:
            out_mpool2 = self.mpool2(x)
            out_softmax2 = self.softmax2_blocks(out_mpool2)
            out_skip2_connection = self.skip2_connection_residual_block(out_softmax2)
            x = out_softmax2
        if self.stages < 2:
            out_mpool3 = self.mpool3(out_softmax2)
            out_softmax3 = self.softmax3_blocks(out_mpool3)
            out_skip3_connection = self.skip3_connection_residual_block(out_softmax3)
            x = out_softmax3
        if self.stages < 1:
            out_mpool4 = self.mpool4(out_softmax3)
            out_softmax4 = self.softmax4_blocks(out_mpool4)
            x = self.interpolation4(out_softmax4) + out_softmax3
            out = x + out_skip3_connection
            x = self.softmax5_blocks(x)
        if self.stages < 2:
            x = self.interpolation3(x) + out_softmax2
            x = x + out_skip2_connection
            x = self.softmax6_blocks(x)
        if self.stages < 3:
            x = self.interpolation2(x) + out_softmax1
            x = x + out_skip1_connection
            x = self.softmax7_blocks(x)
        x = self.interpolation1(x) + out_trunk
        x = self.softmax8_blocks(x)
        x = (1 + x) * out_trunk
        x = self.last_blocks(x)
        return x

class AttentionResNet(nn.Module):
    def __init__(self, in_channels, num_classes, block=ResidualModule, attention=AttentionModule, layers=[1,2,1,2], channels=[64,32,64,32], stride=[1,1,1,1,1], zero_init_residual=False, groups=1, width_per_group=64, replace_stride_with_dilation=None, norm_layer=nn.BatchNorm3d, demographic=[]):
        super(AttentionResNet, self).__init__()
        self._norm_layer = norm_layer
        if len(demographic) > 0:
            self.attn = nn.Linear(len(demographic), channels[(-1)] * block.expansion)
        self.inplanes = 64
        self.dilation = 1
        self.in_channels = int(in_channels)
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False for _ in range(len(layers))]
        self.groups = int(groups)
        self.base_width = int(width_per_group)
        self.conv1 = nn.Conv3d((self.in_channels), (self.inplanes), kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=(stride[0]), padding=1)
        self.layer = []
        for i in range(len(layers)):
            self.layer.append(self._make_layer(block, attention, (channels[i]), (layers[i]), i, stride=(stride[(i + 1)]), dilate=(replace_stride_with_dilation[i])))
        self.layer = (nn.Sequential)(*self.layer)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(channels[(-1)] * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_((m.weight), mode='fan_out', nonlinearity='relu')
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, ResidualModule):
                    nn.init.constant_(m.bn3.weight, 0)
    def _make_layer(self, block, attention, planes, blocks, stages, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(conv1x1(self.inplanes, planes * block.expansion, stride), norm_layer(planes * block.expansion))
        layers = []
        layers.append(block(inplanes=(self.inplanes), planes=planes, stride=stride, downsample=downsample, groups=(self.groups), base_width=(self.base_width),
          dilation=previous_dilation,
          norm_layer=norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(attention(self.inplanes, planes, norm_layer, stages))
        return (nn.Sequential)(*layers)
    def forward(self, x, y=None):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        if y is not None:
            x = x + self.attn(y)
        x = self.fc(x)
        return x

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename_label', required=True, type=str)
    parser.add_argument('--class_label', required=True, type=str)
    parser.add_argument('--data_folder', required=True, type=str)
    parser.add_argument('--data_csv', required=True, type=str)
    parser.add_argument('--demographic', default=None, type=str, nargs='+')
    parser.add_argument('--device', default='cpu', type=str)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--file_type', required=True, type=str, nargs='+')
    parser.add_argument('--epochs', default=30, type=int)
    parser.add_argument('--save', default=None, type=str)
    parser.add_argument('--load', default=None, type=str)
    args = parser.parse_args()
    batch_size = args.batch_size
    data_folder = args.data_folder
    data_csv = args.data_csv
    file_type = tuple(args.file_type)
    num_epochs = args.epochs
    demographic = args.demographic
    if not demographic:
        demographic  = []
    data_transforms = transforms.Compose([transforms.ToTensor()])
    filename_label = args.filename_label
    class_label = args.class_label
    device = args.device
    image_datasets = {x:ImageFolder((os.path.join(data_folder, x)), data_csv, data_transforms, filename_label=filename_label, class_label=class_label, file_type=file_type, demographic=demographic) for x in ('train','val')}
    dataloaders = {x:torch.utils.data.DataLoader((image_datasets[x]), batch_size=(batch_size), shuffle=True, num_workers=1) for x in ('train','val')}
    dataset_sizes = {x:len(image_datasets[x]) for x in ('train', 'val')}
    block = ResidualModule
    attention = AttentionModule
    model = AttentionResNet(in_channels=len(file_type), num_classes=len(set(pd.read_csv(data_csv)[class_label])), block=block, attention=attention).to(device)
    optimizer_ft = optim.Adam(model.parameters(), lr=3e-3, weight_decay=0)
    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
    if args.load:
        try:
            state_dict = torch.load(args.load)
            model.load_state_dict(state_dict)
        except RuntimeError as e:
            print('Ignoring "' + str(e) + '"')
    else:
        model = train_model(model, nn.CrossEntropyLoss(), optimizer_ft, exp_lr_scheduler, dataloaders, dataset_sizes, device, num_epochs, demographic)
        if args.save:
            torch.save(model.state_dict(), args.save)
    image_datasets = {x:ImageFolder((os.path.join(data_folder, x)), data_csv, data_transforms, filename_label=filename_label, class_label=class_label, file_type=file_type, demographic=demographic) for x in ('test', )}
    dataloaders = {x:torch.utils.data.DataLoader((image_datasets[x]), batch_size=batch_size, shuffle=True, num_workers=1) for x in ('test', )}
    dataset_sizes = {x:len(image_datasets[x]) for x in ('test', )}
    predict_model(model, nn.CrossEntropyLoss(), dataloaders, dataset_sizes, device, demographic)
