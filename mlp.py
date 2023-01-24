import torch 
import matplotlib.pyplot as plt



with open("data/names.txt") as f:
    words = f.read().splitlines()

chars = ["."] + sorted(list(set(''.join(words))))
vocab_size = len(chars) # 27 in this case

str_to_idx = {s: i for i, s in enumerate(chars)}
idx_to_str = {i: s for s, i in str_to_idx.items()}


context_length = 10

X = []
Y = []

print(min(len(w) for w in words))

for word in words:
    context = [0] * context_length
    for character in word + ".":
        token = str_to_idx[character]
        X.append(context)
        Y.append(token)
        context = context[1:] + [token]

X = torch.tensor(X)
Y = torch.tensor(Y)
print(X.shape)
print(Y.shape)

from torch.utils.data import TensorDataset, DataLoader, random_split

dataset = TensorDataset(X, Y)
train_ds, test_val_ds = random_split(dataset, [0.8, 0.2])
val_ds, test_ds = random_split(test_val_ds, [0.5, 0.5])

batch_size = 1024
train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=True)
test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=True)

from torch import nn
import torch.nn.functional as F
from tqdm import tqdm

class NeuralNet(nn.Module):
    def __init__(self, embedding_size=20):
        super().__init__()
        self.character_embeddings = nn.Embedding(vocab_size, embedding_size)
        self.input_size = context_length * embedding_size
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1000),
            nn.BatchNorm1d(1000),
            nn.LeakyReLU(),
            nn.Linear(1000, 500),
            nn.BatchNorm1d(500),
            nn.LeakyReLU(),
            nn.Linear(500, vocab_size)
        )

    def forward(self, x):
        embeddings = self.character_embeddings(x)
        inpt = embeddings.view(-1, self.input_size)
        return self.layers(inpt)


model = NeuralNet().cuda()
optim = torch.optim.Adam(model.parameters(), lr=0.01)

global_loss = []

for epoch in range(7):

    train_losses = []
    val_losses = []

    with tqdm(train_dl) as dataloader:
        model.train()
        for x, y in dataloader:
            
            x, y = x.cuda(), y.cuda()
            logits = model(x)
            loss = F.cross_entropy(logits, y)

            train_losses.append(loss.item())

            optim.zero_grad()
            loss.backward()
            optim.step()
    
    with torch.no_grad():
        model.eval()
        for x, y in val_dl:
            x, y = x.cuda(), y.cuda()
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            val_losses.append(loss.item())
        
    av_train_loss = sum(train_losses)/len(train_losses)
    av_val_loss = sum(val_losses)/len(val_losses)

    global_loss.append(av_train_loss)

    print(f"Epoch --> {epoch}\ntrain loss: {av_train_loss:.4f}\nval loss: {av_val_loss:.4f}\n")

def evaluate_model(dataset):
    model.eval()
    losses = []
    for X, Y in dataset:
        X, Y = X.cuda(), Y.cuda()

        logits = model(X)
        loss = F.cross_entropy(logits, Y)
        losses.append(loss.item())
    
    print(sum(losses)/len(losses))


evaluate_model(train_dl)
evaluate_model(val_dl)
evaluate_model(test_dl)

x = torch.arange(len(global_loss))

plt.plot(x, global_loss)
plt.xlabel('Epoch')
plt.ylabel('Loss')

@torch.no_grad()
def generate_names(num=20):
    for _ in range(20):
        out = []
        context = [0] * context_length # initialize with all ...
        while True:
            x = torch.tensor([[context]]).cuda()
            logits = model(x)
            probs = F.softmax(logits, dim=1)
            ix = torch.multinomial(probs, num_samples=1).item()
            context = context[1:] + [ix]
            out.append(ix)
            if ix == 0:
                break
        print(''.join(idx_to_str[i] for i in out))

model.eval()
generate_names()

torch.save(model.state_dict(), "model.pt")


