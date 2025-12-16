class Scalar:
	def __init__(self, val, _prev=(), _op=' ') -> None:
		# val: 标量的数值，可以是 int 或 float
		self.val = val
		# grad: 该标量对损失函数的梯度，初始化为 0.0
		self.grad = 0.0
    # 自动求导相关属性
		# _backward: 反向传播时调用的函数，初始化为空操作
		self._backward = lambda: None
		# _prev: 产生当前标量的前置标量元组，用于构建计算图
		self._prev = _prev
		# _op: 产生当前标量的操作符字符串，用于调试和可视化
		self._op = _op

	def __repr__(self) -> str:
		return f"Scalar(val={self.val}, grad={self.grad})"

	def __add__(self, other):
		next = Scalar(self.val + other.val, (self, other), '+')
		# 定义加法的反向传播函数
		def _backward():
			self.grad += next.grad
			other.grad += next.grad
		# 将反向传播函数赋值给 next 标量
		next._backward = _backward
		return next
	
	def __mul__(self, other):
		next = Scalar(self.val * other.val, (self, other), '*')
		# 定义乘法的反向传播函数
		def _backward():
			self.grad += next.grad * other.val
			other.grad += next.grad * self.val
		# 将反向传播函数赋值给 next 标量
		next._backward = _backward
		return next

	def __pow__(self, other):
		assert isinstance(other, (int, float)), "Wrong Operand Type"
		next = Scalar(self.val ** other, (self,), f'**{other}')
		# 定义幂运算的反向传播函数
		def _backward():
			self.grad += next.grad * other * self.val ** (other - 1)
		# 将反向传播函数赋值给 next 标量
		next._backward = _backward
		return next
	
	def relu(self):
		next = Scalar(max(0, self.val), (self,), 'ReLU')
		# 定义 ReLU 激活函数的反向传播函数
		def _backward():
			self.grad += next.grad * (self.val > 0)
		# 将反向传播函数赋值给 next 标量
		next._backward = _backward
		return next

	def backward(self):
		topo = []
		visited = set()
		def build_topo(v):
			if v not in visited:
				visited.add(v)
				for p in v._prev:
					build_topo(p)
				topo.append(v)
		build_topo(self)
		# back-propagate the gradients
		self.grad = 1.0
		for v in reversed(topo):
			v._backward()

	def __neg__(self):
			return self * Scalar(-1.0)

	def __radd__(self, other):
			return self + other

	def __rmul__(self, other):
			return self * other

	def __sub__(self, other):
			return self + (-other)

	def __rsub__(self, other):
			return other + (-self)
	
	def __truediv__(self, other):
			return self * (other ** -1)
	
	def __rtruediv__(self, other):
			return other * (self ** -1)
