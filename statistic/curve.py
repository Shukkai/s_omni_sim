import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# FPE 
# x_data = np.array([16, 64, 128, 256, 1024])
# y_data = np.array([0.444, 0.841, 1.136, 1.413, 1.764]) 

# Omni WS
# x_data = np.array([32, 64, 128, 256, 1024])
# y_data = np.array([0.864, 1.136, 1.367, 1.524, 1.681]) 

# Omni OS-V
# x_data = np.array([32, 64, 128, 256, 1024])
# y_data = np.array([1.602, 1.769, 1.813, 1.856, 1.82]) 

# FIGLUT 32*4
# x_data = np.array([16, 64, 128, 256])
# y_data = np.array([0.497, 0.9, 1.07, 1.199]) 

# FIGLUT 16*2
x_data = np.array([16, 64, 128, 256, 1024])
y_data = np.array([0.247, 0.397, 0.446, 0.484, 0.51]) 

# Tender 
# x_data = np.array([16, 64, 128, 256, 1024])
# y_data = np.array([0.205, 0.288, 0.346, 0.398, 0.46]) 

# 定義你的回歸模型 (使用反比例模型)
def rational_model(x, P_max, A, k):
    return P_max - (A * np.exp(-k * x))

# 執行 Curve Fitting
# p0 是給予參數的初始猜測值，幫助演算法收斂 [P_max猜測值, A猜測值, k猜測值]
initial_guess = [2.0, 500.0, 0.01] 
popt, pcov = curve_fit(rational_model, x_data, y_data, p0=initial_guess)

P_max_fitted, A_fitted, k_fitted = popt

print(f"預測的極限收斂功耗 (P_max): {P_max_fitted:.4f}")

# 如果需要，還可以畫圖確認擬合狀況
x_fit = np.linspace(64, 2048, 100)
y_fit = rational_model(x_fit, P_max_fitted, A_fitted, k_fitted)
plt.scatter(x_data, y_data, label='Hardware Data')
plt.plot(x_fit, y_fit, color='red', label=f'Fitted Curve (Asymptote={P_max_fitted:.2f})')
plt.legend()
plt.show()