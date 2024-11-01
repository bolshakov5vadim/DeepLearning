from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

class NN(nn.Module):
    def __init__(self, inst_norm=False):
        super(Gen,self).__init__()
        self.n1 = conv(dim_c, dim_g, 4, 2, 1) 
        self.n2 = conv_n(dim_g, dim_g*2, 4, 2, 1, inst_norm=inst_norm)
        self.n3 = conv_n(dim_g*2, dim_g*4, 4, 2, 1, inst_norm=inst_norm)
        self.n4 = conv_n(dim_g*4, dim_g*8, 4, 2, 1, inst_norm=inst_norm)
        self.n5 = conv_n(dim_g*8, dim_g*8, 4, 2, 1, inst_norm=inst_norm)
        self.n6 = conv_n(dim_g*8, dim_g*8, 4, 2, 1, inst_norm=inst_norm)
        self.n7 = conv_n(dim_g*8, dim_g*8, 4, 2, 1, inst_norm=inst_norm)
        self.n8 = conv(dim_g*8, dim_g*8, 4, 2, 1)
        self.m1 = tconv_n(dim_g*8, dim_g*8, 4, 2, 1, inst_norm=inst_norm)
        self.m2 = tconv_n(dim_g*8*2, dim_g*8, 4, 2, 1, inst_norm=inst_norm)
        self.m3 = tconv_n(dim_g*8*2, dim_g*8, 4, 2, 1, inst_norm=inst_norm)
        self.m4 = tconv_n(dim_g*8*2, dim_g*8, 4, 2, 1, inst_norm=inst_norm)
        self.m5 = tconv_n(dim_g*8*2, dim_g*4, 4, 2, 1, inst_norm=inst_norm)
        self.m6 = tconv_n(dim_g*4*2, dim_g*2, 4, 2, 1, inst_norm=inst_norm)
        self.m7 = tconv_n(dim_g*2*2, dim_g*1, 4, 2, 1, inst_norm=inst_norm)
        self.m8 = tconv(dim_g*1*2, dim_c, 4, 2, 1)
        self.tanh = nn.Tanh()
    def forward(self,x):
        n1 = self.n1(x)
        n2 = self.n2(F.leaky_relu(n1, 0.2))
        n3 = self.n3(F.leaky_relu(n2, 0.2))
        n4 = self.n4(F.leaky_relu(n3, 0.2))
        n5 = self.n5(F.leaky_relu(n4, 0.2))
        n6 = self.n6(F.leaky_relu(n5, 0.2))
        n7 = self.n7(F.leaky_relu(n6, 0.2))
        n8 = self.n8(F.leaky_relu(n7, 0.2))
        m1 = torch.cat([F.dropout(self.m1(F.relu(n8)), 0.5, training=True), n7], 1)
        m2 = torch.cat([F.dropout(self.m2(F.relu(m1)), 0.5, training=True), n6], 1)
        m3 = torch.cat([F.dropout(self.m3(F.relu(m2)), 0.5, training=True), n5], 1)
        m4 = torch.cat([self.m4(F.relu(m3)), n4], 1)
        m5 = torch.cat([self.m5(F.relu(m4)), n3], 1)
        m6 = torch.cat([self.m6(F.relu(m5)), n2], 1)
        m7 = torch.cat([self.m7(F.relu(m6)), n1], 1)
        m8 = self.m8(F.relu(m7))
        return self.tanh(m8)

list_of_transformations = [
    transforms.Resize((256,256)),
    transforms.ToTensor(),
]
transform = transforms.Compose(list_of_transformations)

#dataset
#L__test
#¦   L__ Cat
#¦   L__ Dog
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

train_x = sorted(glob(os.path.join(train_path, "image", "*png")))#возвращает 2 массива путей
train_y = sorted(glob(os.path.join(train_path, "mask", "*png")))
train_x, train_y = shuffle(train_x, train_y, random_state=42) # перемешивание
image_pairs = np.hstack((train_x, train_y))

train_dataset = PairedImageDataset(path="/dataset/train/", transform=transform, image_pairs)
val_dataset = PairedImageDataset(path="/dataset/val/", transform=transform, image_pairs)

load_Train = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, 
             shuffle=True, num_workers=num_workers)
load_Test = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, 
            shuffle = False, num_workers=num_workers)
#картинка-картинка


NN = NN(False)
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