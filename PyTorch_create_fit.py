from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

class NN(nn.Module):
    def __init__(self, ch_num1, ch_num2):
        super(NN,self).__init__()
        self.conv1 = nn.Conv2d(ch_num1, ch_num2, stride=3) 
        self.conv2 = nn.Conv2d(ch_num2, ch_num2*2, stride=3)
        self.conv3 = nn.Conv2d(ch_num2*2, ch_num2*4, stride=3)
        self.conv4 = nn.Conv2d(ch_num2*4, ch_num2*8, stride=3)
        self.convt1 = nn.ConvTranspose2d(ch_num2*8, ch_num2*4, stride=3)
        self.convt2 = nn.ConvTranspose2d(ch_num2*4, ch_num2*2, stride=3)
        self.convt3 = nn.ConvTranspose2d(ch_num2*2, ch_num2, stride=3)
        self.convt4 = nn.ConvTranspose2d(ch_num2, ch_num1, stride=3)
        self.conv0 = nn.Conv2d(ch_num2*2, 1, stride=3) 
        self.pool_1 = nn.MaxPool2d((2, 2), stride=2)
        self.uppool_1 = nn.Upsample(scale_factor=2, mode='bilinear')
    def forward(self,x):
        n1 = self.conv1(x)
        n1 = nn.BatchNorm2d(ch_num2, affine=False, track_running_stats=False)(n1)
        n1 = nn.ReLU(n1)
        n1 = self.conv1(n1)
        n1 = self.pool_1(n1)

        n2 = self.conv2(n1)
        n2 = nn.BatchNorm2d(ch_num2*2, affine=False, track_running_stats=False)(n2)
        n2 = nn.ReLU(n2)
        n2 = self.conv2(n2)
        n2 = self.pool_1(n2)

        n3 = self.conv3(n2)
        n3 = nn.BatchNorm2d(ch_num2*4, affine=False, track_running_stats=False)(n3)
        n3 = nn.ReLU(n3)
        n3 = self.conv3(n3)
        n3 = self.pool_1(n3)

        n4 = self.conv4(n3)
        n4 = nn.BatchNorm2d(ch_num2*8, affine=False, track_running_stats=False)(n4)
        n4 = nn.ReLU(n4)
        n4 = self.conv4(n4)
        n4 = self.pool_1(n4)

        xx = self.conv7(n4)
        xx = self.conv7(xx)

        m1 = self.convt1(xx)
        m1 = torch.cat([m1, n4], 1)
        m1 = self.conv4(m1)
        m1 = nn.BatchNorm2d(ch_num2*8, affine=False, track_running_stats=False)(m1)
        m1 = nn.ReLU(m1)
        m1 = self.conv4(m1)
        m1 = self.uppool_1(m1)

        m2 = self.convt2(m1)
        m2 = torch.cat([m2, n3], 1)
        m2 = self.conv3(m2)
        m2 = nn.ReLU(m2)
        m2 = self.conv3(m2)
        m2 = self.uppool_1(m2)

        m3 = self.convt3(m2)
        m3 = torch.cat([m3, n2], 1)
        m3 = self.conv2(m3)
        m3 = nn.ReLU(m3)
        m3 = self.conv2(m3)
        m3 = self.uppool_1(m3)

        m4 = self.convt4(m3)
        m4 = torch.cat([m4, n1], 1)
        m4 = self.conv1(m4)
        m4= nn.ReLU(m4)
        m4 = self.conv1(m4)
        m4 = self.uppool_1(m4)

        end = self.conv0(m4)
        return nn.Tanh(end)

list_of_transformations = [
    transforms.Resize((256,256)),
    transforms.ToTensor(),
]
transform = transforms.Compose(list_of_transformations)

#dataset
#L__train
#¦   L__ Cat
#¦   L__ Dog
train_dataset = ImageFolder(root="/dataset/train/", transform=transform)
val_dataset = ImageFolder(root="/dataset/val/", transform=transform)
batch_size = 50
num_workers = 2
load_Train = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, 
             shuffle=True, num_workers=num_workers)
load_Test = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, 
            shuffle = False, num_workers=num_workers)
#картинка - текст




class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, image_pairs,transform):
        self.image_pairs = image_pairs
        self.transform = transform

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        img1_path, img2_path = self.image_pairs[idx]
        img1 = Image.open(img1_path).convert('RGB')  
        img2 = Image.open(img2_path).convert('RGB')
        img1 = self.transform(img1)
        img2 = self.transform(img2)

        return img1, img2

dataset_path = "new_data"
train_path = os.path.join(dataset_path, "train")
valid_path = os.path.join(dataset_path, "test")

train_x = sorted(glob(os.path.join(train_path, "image", "*png")))#возвращает 2 массива путей
train_y = sorted(glob(os.path.join(train_path, "mask", "*png")))
train_x, train_y = shuffle(train_x, train_y, random_state=42) # перемешивание
image_pairs = np.hstack((train_x, train_y))

train_dataset = CustomDataset(image_pairs, transform=transform)
val_dataset = CustomDataset(image_pairs, transform=transform)

load_Train = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, 
             shuffle=True, num_workers=num_workers)
load_Test = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, 
            shuffle = False, num_workers=num_workers)
#картинка-картинка


NN = NN(64, 128)
print(NN)
optimizer = optim.Adam(Gen.parameters(), lr=2e-4, betas=(0.5, 0.999))
epochs = 5

for ep in range(epochs):
    for batch_index, (x_data, y_data) in enumerate(load_Train):
        output = NN(x_data)
        loss = (output - y_data) ** 2
        # loss = nn.NLLLoss(ouptut, y_data)

        total = y_data.size(0)
        _, predicted = torch.max(output.data, 1)
        correct = (predicted == y_data).sum().item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % log_interval == 0:
                print('Train Epoch: [{}/{}], Loss {:.4f}, Accuracy: {:.2f}%'.format
                (ep+1, epochs, loss.data[0], correct / total * 100)
torch.save(NN.state_dict(), 'conv_net_model.ckpt')
