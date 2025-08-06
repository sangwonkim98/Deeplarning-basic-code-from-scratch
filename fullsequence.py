import numpy as np
import matplotlib.pyplot as plt

# 수치미분 함수
def numerical_derivative(f, x):
    delta_x = 1e-4
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])

    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]

        x[idx] = float(tmp_val) + delta_x
        fx1 = f(x)

        x[idx] = tmp_val - delta_x
        fx2 = f(x)

        grad[idx] = (fx1 - fx2) / (2 * delta_x)
        x[idx] = tmp_val
        it.iternext()

    return grad

# 시그모이드 함수
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# 논리 게이트 클래스
class LogicGate:
    def __init__(self, gate_name, xdata, tdata):
        self.name = gate_name
        self.__xdata = xdata.reshape(4, 2)
        self.__tdata = tdata.reshape(4, 1)

        # 가중치와 편향 초기화
        self.__W2 = np.random.rand(2, 6)
        self.__b2 = np.random.rand(6)
        self.__W3 = np.random.rand(6, 1)
        self.__b3 = np.random.rand(1)

        self.__learning_rate = 1e-2
        
        # 손실 함수 값 기록을 위한 리스트 초기화
        self.__loss_history = []
        
        print(f"{self.name} object is created")

    # 순전파 계산
    def feed_forward(self):
        delta = 1e-7
        z2 = np.dot(self.__xdata, self.__W2) + self.__b2
        a2 = sigmoid(z2)
        z3 = np.dot(a2, self.__W3) + self.__b3
        y = sigmoid(z3)
        return -np.sum(
            self.__tdata * np.log(y + delta) + (1 - self.__tdata) * np.log(1 - y + delta)
        )

    def loss_val(self):
        return self.feed_forward()

    # 학습 함수
    def train(self, max_iter=10001):
        f = lambda x: self.feed_forward()
        print(f"Initial loss value = {self.loss_val()}")

        for i in range(max_iter):
            self.__W2 -= self.__learning_rate * numerical_derivative(f, self.__W2)
            self.__b2 -= self.__learning_rate * numerical_derivative(f, self.__b2)
            self.__W3 -= self.__learning_rate * numerical_derivative(f, self.__W3)
            self.__b3 -= self.__learning_rate * numerical_derivative(f, self.__b3)
            
            # 100번의 반복마다 손실 값 기록
            if i % 100 == 0:
                self.__loss_history.append(self.loss_val())

        print(f"Final loss value = {self.loss_val()}")

    # 예측 함수
    def predict(self, xdata):
        z2 = np.dot(xdata, self.__W2) + self.__b2
        a2 = sigmoid(z2)
        z3 = np.dot(a2, self.__W3) + self.__b3
        y = sigmoid(z3)
        result = 1 if y > 0.5 else 0
        return y, result

    # 손실 함수 값의 변화를 시각화하는 함수 
    def plot_loss_history(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.__loss_history)
        plt.title(f'Loss History for {self.name} Gate')
        plt.xlabel('Iterations (x100)')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.show()

    # 결정 경계를 시각화하는 함수
    def plot_decision_boundary(self):
        plt.figure(figsize=(10, 6))
        
        # 데이터 포인트 그리기
        for i in range(len(self.__xdata)):
            if self.__tdata[i] == 0:
                plt.plot(self.__xdata[i][0], self.__xdata[i][1], 'ro') # Red circle for 0
            else:
                plt.plot(self.__xdata[i][0], self.__xdata[i][1], 'b^') # Blue triangle for 1

        # 결정 경계 그리기
        x_range = np.arange(-0.1, 1.1, 0.01)
        y_range = np.arange(-0.1, 1.1, 0.01)
        xx, yy = np.meshgrid(x_range, y_range)
        
        grid_data = np.c_[xx.ravel(), yy.ravel()]
        
        # 각 그리드 포인트에 대한 예측 값 계산
        predictions = np.array([self.predict(data)[1] for data in grid_data])
        zz = predictions.reshape(xx.shape)
        
        plt.contourf(xx, yy, zz, alpha=0.3, cmap=plt.cm.RdBu)
        plt.title(f'Decision Boundary for {self.name} Gate')
        plt.xlabel('Input 1')
        plt.ylabel('Input 2')
        plt.legend(['Class 0', 'Class 1'])
        plt.grid(True)
        plt.show()


# ========== AND ==========
x_and = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
t_and = np.array([0, 0, 0, 1])

and_gate = LogicGate("AND", x_and, t_and)
and_gate.train()

print("\n[AND Prediction]")
for x in x_and:
    print(and_gate.predict(x))
and_gate.plot_loss_history()
and_gate.plot_decision_boundary()


# ========== OR ==========
x_or = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
t_or = np.array([0, 1, 1, 1])

or_gate = LogicGate("OR", x_or, t_or)
or_gate.train()

print("\n[OR Prediction]")
for x in x_or:
    print(or_gate.predict(x))
or_gate.plot_loss_history()
or_gate.plot_decision_boundary()


# ========== NAND ==========
x_nand = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
t_nand = np.array([1, 1, 1, 0])

nand_gate = LogicGate("NAND", x_nand, t_nand)
nand_gate.train()

print("\n[NAND Prediction]")
for x in x_nand:
    print(nand_gate.predict(x))
nand_gate.plot_loss_history()
nand_gate.plot_decision_boundary()


# ========== XOR ==========
x_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
t_xor = np.array([0, 1, 1, 0])

xor_gate = LogicGate("XOR", x_xor, t_xor)
xor_gate.train(max_iter=20001) # XOR은 더 많은 학습이 필요할 수 있습니다.

print("\n[XOR Prediction]")
for x in x_xor:
    print(xor_gate.predict(x))
xor_gate.plot_loss_history()
xor_gate.plot_decision_boundary()
