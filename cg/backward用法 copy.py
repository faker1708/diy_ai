import torch
x = torch.tensor(1.0, requires_grad=True)
y = torch.tensor(2.0, requires_grad=True)
z = x**2+y
z.backward()
print(z, x.grad, y.grad)

# >>> tensor(3., grad_fn=<AddBackward0>) tensor(2.) tensor(1.)

