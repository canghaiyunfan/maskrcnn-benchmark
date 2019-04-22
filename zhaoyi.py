import torch
import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MyMul(nn.Module):
    def forward(self,input):
        out = input*2
        return out

class MyMean(nn.Module):
    def forward(self,input):
        out = torch.pow(input,2)
        return out

def tensor_hook(grad):
    print("tensor hook")
    print("grad:",grad)
    return grad

class MyNet(nn.Module):
    def __init__(self):
        super(MyNet,self).__init__()
        self.f1 = nn.Linear(4,1,bias=True)
        self.f2 = MyMean()
        self.weight_init()

    def forward(self,input):
        self.input = input
        output = self.f1(input)
        output = self.f2(output)
        return output

    def weight_init(self):
        self.f1.weight.data.fill_(8.0)
        self.f1.bias.data.fill_(2)

    def my_hook(self,module,grad_input,grad_output):
        print("doing my_hook")
        print("original grad:",grad_input)
        print("original outgrad:",grad_output)

        return grad_input

if __name__ == "__main__":
    input = torch.tensor([1,2,3,4],dtype=torch.float32 ,device=device,requires_grad=True)

    net = MyNet()
    net = net.to(device)

    net.register_forward_hook(net.my_hook)

    input.register_hook(tensor_hook)

    result = net(input)
    print('result =', result)














