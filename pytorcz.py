import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


class Model(nn.Module):
    #-> Input layer (4 features of flower) 
    # -> Hidden Layer1 (number of neurons) 
    # -> HL2 (n) 
    # -> output (3 classes of iris flowers)
    def __init__(self, in_features=4, h1=8, h2=9, out_features=3):
        super().__init__()
        self.fc1 = nn.Linear(in_features, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.out = nn.Linear(h2, out_features)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
    
        return x
    
#Pick manual seed for randomization
torch.manual_seed(41)
model = Model()


url = 'https://gist.githubusercontent.com/netj/8836201/raw/6f9306ad21398ea43cba4f7d537619d0e07d5ae3/iris.csv'
my_df = pd.read_csv(url)
my_df['variety'] = my_df['variety'].replace('Setosa', 0.0)
my_df['variety'] = my_df['variety'].replace('Versicolor', 1.0)
my_df['variety'] = my_df['variety'].replace('Virginica', 2.0)

# Train Test Split! Set X,y
X = my_df.drop('variety', axis=1)
y = my_df['variety']

#Convert to numpy arrays
X = X.values
y = y.values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=41)

#Convert X features to float Tensors and y labels to Tensord long
X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)

#Set the criterion of model to measure the error - how far off the predictions are from the data
criterion = nn.CrossEntropyLoss()
#Choose Adam Optimizer, lr = learning rate (if error doesn't go down after a bunch of iterations (*EPOCHS*), lower our learning rate)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

#Train our model!
#How many epochs? (1 -> one run through all the training data in our network)
epochs = 100
losses = []
for i in range(epochs):
    #Go forward and get a prediction
    y_pred = model.forward(X_train) #Get predicted results

    #Measuer the loss/error
    loss = criterion(y_pred, y_train) #predicted value vs train value

    #Keep track of our losses
    losses.append(loss.detach().numpy())

    #print every 10 epochs
    if i % 10 == 0:
        print(f"Epoch:{i} and loss:{loss}")
    
    #Do some back propagation: tahe the error rate of forward propagation and feed it back thru the network to fine tune the weights
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

#Graph it out
# plt.plot(range(epochs), losses)
# plt.ylabel("loss/error")
# plt.xlabel("Epoch")
# plt.show()

with torch.no_grad():   #Turn off back propagation
    y_val = model.forward(X_test)  #X_test are features from our test set, y_eval are predictions
    loss = criterion(y_val, y_test)

correct = 0
with torch.no_grad():
    for i, data in enumerate(X_test):
        y_val = model.forward(data)
        print(f'{i+1}.) {str(y_val)} \t {y_test[i]} \t {y_val.argmax().item()}') #Will tell us what type of flower our network thinks it is

        #Correct or not
        if y_val.argmax().item() == y_test[i]:
            correct += 1
print(f'We got {correct} correct')


#DODAJEMY NOWEGO KWIATKA
new_iris = torch.tensor([4.7, 3.2, 1.3, 0.2])
newer_iris = torch.tensor([5.9, 3.0, 5.1, 1.8])
with torch.no_grad():
    model(new_iris)

#Save NN model
torch.save(model.state_dict(), 'Nauka/iris_model.pt')
#Load saved model
new_model = Model()
new_model.load_state_dict(torch.load('Nauka/iris_model.pt', weights_only=True))

print(new_model.eval())