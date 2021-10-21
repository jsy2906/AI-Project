file = open('./regression_data.txt','r')  # open the file with read-only
text = file.readlines()  # read all line texts
file.close()  # close the file

x_data = []
y_data = []

# float타입으로 변환
for idx,s in enumerate(text):
    data = s.split()
    x_data.append(float(data[0]))
    y_data.append(float(data[1]))    

# 넘파이 Array로 변환
x_data = np.asarray(x_data, dtype=np.float32)
y_data = np.asarray(y_data, dtype=np.float32)

print("shape of input data: ", x_data.shape)
print("shape of output data: ", y_data.shape)

# 데이터 분포 그래프로 확인
%matplotlib inline

plt.figure(1)
plt.plot(x_data, y_data, 'ro') # plot data

plt.xlabel('x-axis')  
plt.ylabel('y-axis')
plt.title('My data')

plt.show()

# Hyper-parameters 설정
input_size = 1
output_size = 1
num_epochs = 100
learning_rate = 0.1

# KFold 사용
k = 5
kfold = KFold(n_splits=k, shuffle=True)

# 데이터 shape 변경
print(x_data.shape, y_data.shape)
if len(x_data.shape)==1 and len(y_data.shape)==1:
  x_data = np.expand_dims(x_data, axis=-1)
  y_data = np.expand_dims(y_data, axis=-1)
print(x_data.shape, y_data.shape)

# 모델 생성
n = 0
val_loss_list = []
for train, test in kfold.split(x_data):
  n += 1
  xtrain = np.array([x_data[i] for i in train])
  ytrain = np.array([y_data[i] for i in train])
  xtest = np.array([x_data[j] for j in test])
  ytest = np.array([y_data[j] for j in test])

  # Linear regression model, y = Wx+b
  model = nn.Linear(input_size, output_size) 

  # Loss and optimizer
  criterion = nn.MSELoss()
  optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
  
  # Trainset 학습
  train_loss = []
  for epoch in range(num_epochs):
    inputs = torch.from_numpy(xtrain)
    targets = torch.from_numpy(ytrain)

    outputs = model(inputs)
    loss = criterion(outputs, targets)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    train_loss.append(loss.item())
    if (epoch+1) % 100 == 0:
        print('Epoch [{}/{}], Loss: {:.8f}'.format(epoch+1, num_epochs, loss.item()))

  plt.figure(1)
  plt.plot(train_loss)
  plt.title('Trainset Loss')

# validationset 확인
  input_test = torch.from_numpy(xtest)
  target_test = torch.from_numpy(ytest)
  output_test = model(input_test)
  val_loss = criterion(input_test, target_test)
  print(f'%d. Val loss : %.8f'% (n, val_loss.item()))
  val_loss_list.append(val_loss.item())

 
val_losses = np.array(val_loss_list)
print('Total Validation Loss :', val_losses.mean())

plt.figure(2)
plt.plot(val_loss_list)
plt.title('Validationset Loss')
