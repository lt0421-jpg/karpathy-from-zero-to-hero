import random
import math
import numpy as np
import matplotlib.pyplot as plt

# 定义"值“类
class Value:
    def __init__(self,data,_children=(),_op='',label=''):  #初始化
        self.data=data
        self.grad=0.0
        self._backward = lambda: None  #a function do nothing
        self._prev=set(_children)
        self._op=_op
        self.label=label

    def __repr__(self):
        return f"Value(data={self.data})" #定义print输出形式
    
    def __add__(self,other):
        other = other if isinstance(other,Value) else Value(other) #other是Value不动，不是则包装成Value
        out=Value(self.data + other.data,(self,other),'+') #定义对象之间的加法

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward

        return out 
     
    def __mul__(self,other):
        other= other if isinstance(other,Value) else Value(other)
        out=Value(self.data * other.data,(self,other),'*') #定义对象之间的加法

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out

    def __pow__(self,other):                                            #other can only be a number
        assert isinstance(other,(int,float)),"only supporting int/floar powers for npw"    #???????
        out=Value(self.data**other,(self,),f'**{other}') 

        def _backward():
            self.grad += other * self.data**(other-1) * out.grad
        out._backward = _backward

        return out
    
    def __rmul__(self,other):    #解决a*2可以 而 2*a不行的问题
        return self * other

    def __neg__(self):
        return -1*self

    def __sub__(self,other):
        return self + (-other)
    
    def tanh(self):
        x = self.data
        t=(math.exp(2*x)-1)/(math.exp(2*x)+1)
        out = Value(t,(self, ),'tanh')

        def _backward():
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward                   #?????
        return out

    def __truediv__(self,other):
        return self * other**-1

    def exp(self):
        x = self.data
        out=Value(math.exp(x),(self, ),'exp')

        def _backward():
            self.grad += out.data * out.grad  #d e**x / d x = e**x为其本身out.data,再乘上链式法则中的out.grad
        out._backward=_backward

        return out
    
    def backward(self):

        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)           #把零散的对象塞到topo[]中

        self.grad = 1.0
        for node in reversed(topo):  
            node._backward()        #从最终结果向前遍历，反向传播
        #backward 所有对象DFS回溯操作
        #_backward 某个对象的回溯


# 定义"神经元"类 神经元 = 线性变换 + 非线性变换
class Neuron:                                                 # 神经元 = 线性变换 + 非线性变换

    def __init__(self,nin):                                   # nin = the number of inputs
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]  #为神经元每一个输入创建一个随机权重,self.w是一个权重列表
        self.b = Value(random.uniform(-1,1))                        #bias偏置，控制神经元整体触发频率

    def __call__(self,x):                                     #feed the x(which is a list) to n(which is a value) to get the out
        '''act=sum(wi*xi)+b'''
        act = sum( (wi*xi for wi,xi in zip(self.w, x)) ,self.b) #self.b为sum的累加起点。 x为上个Layer传来的结果列表
        '''What is zip?
        list1 = [a,b,c]
        list2 = [x,y,z]
        then zip(list1,list2) = [(a,x),(b,y),(c,z)]
        '''

        '''tanh(act)'''
        out = act.tanh()                                        #激活函数非线性变换。类似创建一个结构体{self.w,x}和遍历所有结构体的迭代器
        return out

    def parameters(self):              #清晰呈现参数列表
        return self.w+[self.b]

# 定义"Layer"类 层(Layer) = 一层中所有神经元的集合
class Layer:

    def __init__(self,nin,nout):                #nin输入数量特征，即上一层神经元数量。nout输出数量特征，即这一层神经元数量。
        # 创建神经元列表(创建nout个神经元，每个神经元有nin个输入）：
        self.neurons = [Neuron(nin) for _ in range(nout)]  # _:只关心循环的次数，不关心循环的序号。

    def __call__(self,x):
        #返回 all神经元输出值的列表
        outs = [n(x) for n in self.neurons]  
        return outs[0] if len(outs) == 1 else outs  # 如果列表中只有一个元素，则只返回该元素。(看起来好看)

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()] #[代词 for1,for嵌套循环找每一个p]
# 定义"MLP"类 Mlp = 所有的参的集合及其运算
class MLP:

    def __init__(self,nin,nouts): # nouts: 每一层的神经元数量组成的列表。nin: 初始输入个数，即第一层神经元有几个输入。
        sz = [nin] + nouts        # 在列表nouts前面添加一个元素nin，sz是每一个Layer输入个数的列表
        self.layers = [Layer(sz[i],sz[i+1]) for i in range(len(nouts))]   #把所有的Layer存入MLP对象中的Layers列表

    def __call__(self,x):
        for layer in self.layers:
            x=layer(x)                 #调用 Layer 中的 __call__来一层一层递推神经元的值
        return x                       #return最后一层神经元中的值

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]  #同上，三个premeter彼此嵌套

#构建神经网络
n=MLP(3,[4,4,1])
#给训练样例
xs=[
    [2.0,3.0,-1.0],
    [3.0,-1.0,0.5],
    [0.5,1.0,1.0],
    [1.0,1.0,-1.0],
]
ys = [1.0,-1.0,-1.0,1.0]

#梯度下降过程
for k in range(1000):
    #forward path
    ypre = [n(x) for x in xs] # 推到最后一层Layer 所有 神经元的值的列表
    loss = sum(((yout-ygt)**2 for ygt,yout in zip(ys,ypre)),Value(0)) #Value(0)定义初始值以及明确sum的类型 # sum((生成器)，初始值) 生成器需要单独括号
    # We expected loss to be low

    #backward path
    for p in n.parameters():
       p.grad = 0                  #zero the grads first!!!!
    loss.backward()                #recalculate the grads

    #update
    for p in n.parameters():
        p.data += -0.09*p.grad    #grad = d loss / d 参数。反向变化使得loss更小  0.12 is called learning rate

    #print(k,loss.data)
    if k == 9999:
        print(ypre)

'''
# sum((生成器)，初始值) 生成器需要单独括号
# We expected loss to be low
print(loss)
loss.backward()

print(n.layers[0].neurons[0].w)
print(n.layers[0].neurons[0].w[0].grad)
print("**************************************")
print(n.parameters(),'\n') 


for p in n.parameters():
    p.data += -0.01*p.grad    #grad = d loss / d 参数。反向变化使得loss更小  0.01 is called learning rate

ypre = [n(x) for x in xs]
print(ypre)
loss = sum(((yout-ygt)**2 for ygt,yout in zip(ys,ypre)),Value(0))
print(loss)
'''