import numpy as np
import matplotlib.pyplot as plt

def softmax(logits, T=1.0):
    logits = np.array(logits)
    exp = np.exp(logits / T)
    return exp / exp.sum()

# 假设 logits（未归一化的得分）
logits = np.array([2.0, 1.0, 0.5])

T_values = [0.1, 0.5, 1.0, 2.0, 5.0]
x = np.arange(len(logits))

plt.figure(figsize=(6, 4))
for T in T_values:
    probs = softmax(logits, T)
    plt.plot(x, probs, marker='o', label=f'T={T}')

plt.title('Softmax with different temperature T')
plt.xlabel('Class index')
plt.ylabel('Probability')
plt.ylim(0, 1)
plt.legend()
plt.grid(alpha=0.3)
plt.show()
