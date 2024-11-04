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
        self.conv5 = nn.Conv2d(ch_num2*8, ch_num2*8, stride=3)
        self.conv6 = nn.Conv2d(ch_num2*8, ch_num2*8, stride=3)
        self.conv7 = nn.Conv2d(ch_num2*8, ch_num2*8, stride=3)
        self.conv8 = nn.Conv2d(ch_num2*8, ch_num2*8, stride=3)
        self.convt1 = nn.ConvTranspose2d(ch_num2*8, ch_num2*8, stride=3)
        self.convt2 = nn.ConvTranspose2d(ch_num2*8*2, ch_num2*8, stride=3)
        self.convt3 = nn.ConvTranspose2d(ch_num2*8*2, ch_num2*8, stride=3)
        self.convt4 = nn.ConvTranspose2d(ch_num2*8*2, ch_num2*8, stride=3)
        self.convt5 = nn.ConvTranspose2d(ch_num2*8*2, ch_num2*4, stride=3)
        self.convt6 = nn.ConvTranspose2d(ch_num2*4*2, ch_num2*2, stride=3)
        self.convt7 = nn.ConvTranspose2d(ch_num2*2*2, ch_num2*1, stride=3)
        self.conv0 = nn.Conv2d(ch_num2*2, 1, stride=3) 
    def forward(self,x):
        n1 = self.conv1(x)
        n1 = nn.ReLU(n1)
        n2 = self.conv2(n1)
        n2 = nn.ReLU(n2)
        n3 = self.conv3(n2)
        n3 = nn.ReLU(n3)
        n4 = self.conv4(n3)
        n4 = nn.ReLU(n4)
        n5 = self.conv5(n4)
        n5 = nn.ReLU(n5)
        n6 = self.conv6(n5)
        n6 = nn.ReLU(n6)
        n7 = self.conv7(n6)
        n7 = nn.ReLU(n7)

        n8 = self.conv8(n7)
        n8 = nn.ReLU(n8)

        m1 = self.convt1(n8)
        m1 = torch.cat([m1, n7], 1)
        m1 = self.conv7(m1)
        m1 = self.conv7(m1)

        m2 = self.convt2(m1)
        m2 = torch.cat([m2, n6], 1)
        m2 = self.conv6(m2)
        m2 = self.conv6(m2)

        m3 = self.convt3(m2)
        m3 = torch.cat([m3, n5], 1)
        m3 = self.conv5(m3)
        m3 = self.conv5(m3)

        m4 = self.convt4(m3)
        m4 = torch.cat([m4, n4], 1)
        m4 = self.conv4(m4)
        m4 = self.conv4(m4)

        m5 = self.convt5(m4)
        m5 = torch.cat([m5, n3], 1)
        m5 = self.conv3(m5)
        m5 = self.conv3(m5)

        m6 = self.convt6(m5)
        m6 = torch.cat([m6, n2], 1)
        m6 = self.conv2(m6)
        m6 = self.conv2(m6)

        m7 = self.convt7(m6)
        m7 = torch.cat([m7, n1], 1)
        m7 = self.conv2(m7)
        m7 = self.conv2(m7)

        m8 = self.conv0(m7)
        return nn.Tanh(m8)

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
#êàðòèíêà - òåêñò




class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, root,transform,,image_pairs):
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

train_x = sorted(glob(os.path.join(train_path, "image", "*png")))#âîçâðàùàåò 2 ìàññèâà ïóòåé
train_y = sorted(glob(os.path.join(train_path, "mask", "*png")))
train_x, train_y = shuffle(train_x, train_y, random_state=42) # ïåðåìåøèâàíèå
image_pairs = np.hstack((train_x, train_y))

train_dataset = PairedImageDataset(path="/dataset/train/", transform=transform, image_pairs)
val_dataset = PairedImageDataset(path="/dataset/val/", transform=transform, image_pairs)

load_Train = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, 
             shuffle=True, num_workers=num_workers)
load_Test = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, 
            shuffle = False, num_workers=num_workers)
#êàðòèíêà-êàðòèíêà


NN = NN(64, 128)
print(NN)
optimizer = optim.Adam(Gen.parameters(), lr=2e-4, betas=(0.5, 0.999))
iter_per_plot = 500
epochs = 5
L1_lambda = 100.0

for ep in range(epochs):
    for batch_index, (x_data, y_data) in enumerate(load_Train):
        output = NN(x_data)
        loss = (output - y_data) ** 2
        # loss = nn.NLLLoss(ouptut, y_data)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total = y_data.size(0)
        _, predicted = torch.max(output.data, 1)
        correct = (predicted == labels).sum().item()

        if batch_idx % log_interval == 0:
                print('Train Epoch: [{}/{}], Loss {:.4f}, Accuracy: {:.2f}%'.format
                (ep+1, epochs, loss.data[0], correct / total * 100)
torch.save(NN.state_dict(), 'conv_net_model.ckpt')
