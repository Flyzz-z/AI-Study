import random
from myflow.nn import MLP, MSELoss
import myflow.autograd as ag

def gen(n=64):
    data = []
    for _ in range(n):
        x1 = random.uniform(-1.0, 1.0)
        x2 = random.uniform(-1.0, 1.0)
        y = 2 * x1 - 3 * x2 + 0.5
        data.append(([ag.Scalar(x1), ag.Scalar(x2)], [ag.Scalar(y)]))
    return data

model = MLP(2, [8, 1])
loss_fn = MSELoss()
lr = 0.1
epochs = 60

for epoch in range(epochs):
    model.zero_grad()
    batch = gen(64)
    total_loss = ag.Scalar(0.0)
    for x, y_true in batch:
        y_pred = model(x)
        total_loss = total_loss + loss_fn(y_pred, y_true)
    total_loss = total_loss / ag.Scalar(len(batch))
    total_loss.backward()
    for p in model.param():
        p.val -= lr * p.grad
    print(epoch, total_loss.val)

test_x = [ag.Scalar(0.3), ag.Scalar(-0.2)]
pred = model(test_x)[0].val
true = 2 * 0.3 - 3 * (-0.2) + 0.5
print("pred", pred)
print("true", true)
