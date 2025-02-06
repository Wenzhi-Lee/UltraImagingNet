import numpy as np
import scipy.io
from matplotlib import pyplot as plt
N = 10416
sampling_rate = 236.72e6  # 采样率
totalscal = 64
fc = 2e6  # Morlet小波的中心频率

# 创建时间轴
time_axis = np.arange(0, N) / sampling_rate

# 设置新的采样长度
new_length = 1000  # 目标采样点数

# 计算新的采样率
new_sampling_rate = sampling_rate * new_length / N

def load_and_print_shape(mat_file):
    # 加载 .mat 文件
    data = scipy.io.loadmat(mat_file)

    # 获取数据字典中的 'final_data' 键对应的数据
    final_data = data['final_data'][0, 0]  # 获取内部字典

    # 打印 'wave' 数据的 shape
    print(f"Shape of 'wave': {final_data['wave'].shape}")
    wave_data = final_data['wave']  # Access the 'wave' data from the final results

    # For example, extract the first sensor (0) and the first transducer (0), and all frequency bins (64) and time steps (1000)
    i, j = 0, 16  # Specify the sensor and transducer indices
    cwt_data = wave_data[i, j // 16, :, :]  # Extract the CWT result for the given indices

    # Create the time and frequency axes
    time_axis = np.arange(0, cwt_data.shape[0]) / new_sampling_rate  # Time axis for 1000 time steps
    frequencies = np.linspace(fc - fc * 0.5, fc + fc * 0.5, totalscal)  # Frequency axis for 64 scales

    # Plotting the CWT result (contour plot for time vs frequency)
    plt.figure(figsize=(10, 6))
    plt.contourf(frequencies, time_axis,  np.abs(cwt_data))
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time (s)')
    plt.title(f"Time-Frequency Plot for Sensor {i}, Transducer {j // 16}")
    plt.show()

    # 打印 'phase' 数据的 shape
    print(f"Shape of 'phase': {final_data['phase'].shape}")

    # 打印 'defeat' 数据的 shape
    print(f"Shape of 'defeat': {final_data['defeat'].shape}")

    # 打印 final_data 本身的 shape
    print(f"Shape of final_data: {final_data.shape}")


# 调用函数并传入生成的 MAT 文件路径
load_and_print_shape(r'C:\Users\user\FORT-main\output\final_data_sensor_data_simulated')
