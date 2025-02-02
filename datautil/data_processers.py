import numpy as np
import scipy.io
import os
from scipy.signal import cwt, morlet2, hilbert
import matplotlib.pyplot as plt
import pywt
from scipy.signal import resample
from joblib import Parallel, delayed

# 参数设置
sampling_rate = 236.72e6  # 采样率
fc = 2e6  # Morlet小波的中心频率
totalscal = 64  # 设置 CWT 参数
output_dir = 'output'  # 结果保存的文件夹
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


# 读取 .mat 文件并处理数据
def load_data(file_path):
    data = scipy.io.loadmat(file_path)
    sensor_data = data['data']
    return sensor_data


# CWT计算
def compute_cwt_for_sensor(i, j, sensor_data, scales, fc, sampling_rate):
    extracted_data = sensor_data[i, j, :]

    # 重新采样数据为1000个点
    new_length = 1000
    resampled_data = resample(extracted_data, new_length)  # 使用 scipy.signal.resample 进行重采样
    sr = sampling_rate * new_length / extracted_data.shape[0]  # 新的采样率

    # 执行小波变换
    cwtmatr, frequencies = pywt.cwt(resampled_data, scales, 'morl', 1.0 / sr)

    # 压缩小波变换结果
    indices = np.linspace(0, cwtmatr.shape[1] - 1, new_length, dtype=int)
    compressed_wavelet_transform = cwtmatr[:, indices]

    return i, j // 16, compressed_wavelet_transform


def compute_cwt(sensor_data, num, sensor):
    N = sensor_data.shape[2]
    time_axis = np.arange(0, N) / sampling_rate  # 创建时间轴

    # 设置 CWT 参数
    new_length = 1000
    new_sampling_rate = sampling_rate * new_length / N
    print(f"New sampling rate: {new_sampling_rate} Hz")

    # 设置尺度
    fc = pywt.central_frequency('morl')  # 计算小波函数的中心频率
    cparam = 2 * fc * totalscal  # 常数c
    scales = cparam / np.arange(totalscal+1, 1, -1)  # 为使转换后的频率序列是一等差序列，尺度序列必须取为这一形式（也即小波尺度）

    # 并行计算每个传感器数据的小波变换
    results = Parallel(n_jobs=-1)(delayed(compute_cwt_for_sensor)(i, j, sensor_data, scales, fc, sampling_rate)
                                  for i in range(num) for j in range(0, sensor, 16))

    WaveletData = np.zeros((num, num, totalscal, new_length), dtype=np.complex64)
    for i, j, compressed_wavelet_transform in results:
        WaveletData[i, j, :, :] = compressed_wavelet_transform

    return WaveletData


# 相位计算
def compute_phase(sensor_data, num_transducers, num_sensor):
    PhaseData = np.zeros((num_transducers, num_sensor, sensor_data.shape[2]), dtype=np.float32)

    for i in range(num_transducers):
        for j in range(num_sensor):
            extracted_data = sensor_data[i, j, :]
            analytic_signal = hilbert(extracted_data)
            phase = np.angle(analytic_signal)
            PhaseData[i, j, :] = phase

    return PhaseData


# 计算相位差
def compute_phase_difference(PhaseData, x_points, y_points, transducer_positions, sensor_positions):
    phase_diff = np.zeros((len(transducer_positions), len(sensor_positions), 64, 64), dtype=np.float32)
    v = 5918  # 声速 (单位：m/s)

    for i in range(len(transducer_positions)):
        grid_x, grid_y = np.meshgrid(x_points, y_points)
        tx_pos = transducer_positions[i]
        d_t = np.sqrt((grid_x - tx_pos) ** 2 + grid_y ** 2)

        for x in range(len(sensor_positions)):
            rx_pos_x = sensor_positions[x]
            d_rx = np.sqrt((grid_x - rx_pos_x) ** 2 + grid_y ** 2)
            t_total_x = (d_t + d_rx) * 0.001 / v  # 计算传播时间
            t_sample_x = np.round(t_total_x * sampling_rate).astype(int)

            signal_values_x = PhaseData[i, x, t_sample_x]
            phase_diff[i, x, :, :] = signal_values_x

    return phase_diff


# 保存计算结果到文件
def save_data(output_file, data_dict):
    scipy.io.savemat(output_file, data_dict)
    print(f"数据已成功保存到 {output_file}")


# 处理文件
def process_file(file_name):
    # 获取当前文件的 defeat_matrix
    defeat_matrix = defeat_dict.get(file_name)

    # 读取数据
    mat_file = r'C:\Users\user\Desktop\Matlab\bin\wave\{}'.format(file_name)
    sensor_data = load_data(mat_file)
    num = 32
    sensor = 512

    # CWT 计算
    WaveletData = compute_cwt(sensor_data, num, sensor)

    # 保存 CWT 结果
    save_data(f"{output_dir}/WaveletData_{file_name}", {'WaveletData': WaveletData})

    # 计算相位
    sensor_data = sensor_data[:, ::16, :]
    num_transducers = sensor_data.shape[0]
    num_sensor = sensor_data.shape[1]
    PhaseData = compute_phase(sensor_data, num_transducers, num_sensor)

    # 计算相位差
    x_points = np.linspace(-25.6, 25.6, 64)
    y_points = np.linspace(0, 51.2, 64)
    transducer_positions = np.linspace(-1.5, 1.6, num_transducers)
    sensor_positions = np.linspace(-25.6, 25.6, num_transducers)
    phase_diff = compute_phase_difference(PhaseData, x_points, y_points, transducer_positions, sensor_positions)

    # 保存相位差数据
    #save_data(f"{output_dir}/phase_data_{file_name}", {'phase_diff': phase_diff})

    # 创建最终数据字典
    final_data = {'wave': WaveletData, 'phase': phase_diff, 'defeat': defeat_matrix}

    # 保存最终数据
    save_data(f"{output_dir}/final_data_{file_name}", {'final_data': final_data})

    # 输出 final_data 各部分的 shape
    final_data_shapes = {key: value.shape for key, value in final_data.items()}
    print(f"Final data shapes for {file_name}: {final_data_shapes}")

    # 返回最终数据
    return final_data


# 执行处理
file_names = [

'sensor_data_x346y346'
]


# 生成 defeat_matrix 的函数
def generate_defeat_matrix(num_defects):
    # 创建一个 10x3 的全零矩阵
    defeat_matrix = np.zeros((10, 3))

    # 循环输入每个缺陷的 3 个维度
    for i in range(num_defects):
        print(f"请输入第 {i + 1} 个缺陷的 3 个维度：")
        # 输入缺陷的 3 个维度，假设你会输入 3 个数值
        x, y, radius = map(int, input("输入格式：x y radius").split())

        # 将输入的缺陷数据填入矩阵中
        defeat_matrix[i, :] = [x, y, radius]

    return defeat_matrix


# 为每个文件指定对应的 defeat_matrix
defeat_dict = {}

# 输入每个文件对应的缺陷数量，并生成对应的 defeat_matrix
for file_name in file_names:
    print(f"请输入文件 {file_name} 的缺陷数量：")
    num_defects = int(input("缺陷数量："))

    # 为每个文件生成对应的 defeat_matrix
    defeat_matrix = generate_defeat_matrix(num_defects)

    # 将生成的 defeat_matrix 存储到 defeat_dict 中
    defeat_dict[file_name] = defeat_matrix

for file_name in file_names:
    final_data = process_file(file_name)
    print(f"处理后的最终数据：{file_name}")