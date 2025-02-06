import numpy as np
import scipy.io
import os
from scipy.spatial import KDTree
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import matlab.engine
from scipy.signal import resample
import pywt
from scipy.signal import hilbert



# ---------------------------- 全局配置 ----------------------------
GRID_SIZE = 512  # 网格尺寸
MAX_DEFECTS = 10  # 最大缺陷数量
MIN_DEFECTS = 1  # 最小缺陷数量
MIN_RADIUS = 5  # 最小缺陷半径
MAX_RADIUS = 50  # 最大缺陷半径
SAFE_MARGIN = 20  # 缺陷间安全间距
SIMULATION_TIMES = 2  # 仿真次数
COVERAGE_THRESHOLD = 0.8  # 覆盖率目标
sampling_rate = 236.72e6  # 采样率
totalscal = 64  # 设置 CWT 参数
output_dir = 'output'  # 结果保存的文件夹
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


# ---------------------------- 核心功能模块 ----------------------------
class CoverageOptimizer:
    def __init__(self):
        self.coverage_map = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)
        self.history_defects = []

    def update_coverage(self, defects):
        """更新覆盖图并记录历史缺陷"""
        for x, y, r in defects:
            x_min = max(0, int(x - r - SAFE_MARGIN))
            x_max = min(GRID_SIZE, int(x + r + SAFE_MARGIN))
            y_min = max(0, int(y - r - SAFE_MARGIN))
            y_max = min(GRID_SIZE, int(y + r + SAFE_MARGIN))
            self.coverage_map[x_min:x_max, y_min:y_max] = True

        self.history_defects.extend(defects)

    def get_coverage_rate(self):
        """计算当前覆盖率"""
        return np.mean(self.coverage_map)

    def generate_defects(self):
        """生成优化后的缺陷配置"""
        num_defects = np.random.randint(MIN_DEFECTS, MAX_DEFECTS + 1)
        defects = []

        # 优先填补未覆盖区域
        uncovered = np.argwhere(~self.coverage_map)
        if len(uncovered) > 0:
            kdtree = KDTree(uncovered)
            for _ in range(num_defects):
                if len(defects) > 0:
                    last_pos = defects[-1][:2]
                    d, idx = kdtree.query(last_pos, k=10)  # 查找最近未覆盖点
                    candidates = uncovered[idx]
                else:
                    candidates = uncovered

                if len(candidates) > 0:
                    idx = np.random.choice(len(candidates))
                    x, y = candidates[idx]
                    r = np.random.randint(MIN_RADIUS, MAX_RADIUS)
                    defects.append((x, y, r))
                    # 更新临时占用区域
                    x_min = max(0, x - r - SAFE_MARGIN)
                    x_max = min(GRID_SIZE, x + r + SAFE_MARGIN)
                    y_min = max(0, y - r - SAFE_MARGIN)
                    y_max = min(GRID_SIZE, y + r + SAFE_MARGIN)
                    self.coverage_map[x_min:x_max, y_min:y_max] = True
                else:
                    break

        # 补充随机缺陷
        while len(defects) < num_defects:
            x = np.random.randint(0, GRID_SIZE)
            y = np.random.randint(0, GRID_SIZE)
            r = np.random.randint(MIN_RADIUS, MAX_RADIUS)

            # 检查碰撞
            collision = False
            for (ex, ey, er) in defects:
                distance = np.sqrt((x - ex) ** 2 + (y - ey) ** 2)
                if distance < (r + er + SAFE_MARGIN):
                    collision = True
                    break
            if not collision:
                defects.append((x, y, r))

        return defects


# ---------------------------- MATLAB 交互模块 ----------------------------
def run_matlab_simulation(defects):
    """运行 MATLAB 仿真并返回传感器数据"""
    eng = matlab.engine.start_matlab()

    # 构建 MATLAB 结构体
    matlab_defects = []
    for x, y, r in defects:
        defect = eng.struct(
        'x_pos', float(x),
        'y_pos', float(y),
        'radius', float(r)
        )
        matlab_defects.append(defect)

    # 将Python列表转换为MATLAB的cell数组
    ml_cell = matlab_defects if len(matlab_defects) > 0 else eng.cellarray([])

    # 调用仿真函数
    try:
        sensor_data_ml = eng.run_simulation(ml_cell)
        sensor_data = np.array(sensor_data_ml, dtype=np.float32).transpose(2, 1, 0)
    except Exception as e:
        print(f"MATLAB仿真失败: {str(e)}")
        sensor_data = np.zeros((32, 512, 1000), dtype=np.float32)
    finally:
        eng.quit()

    return sensor_data


# ---------------------------- 数据处理管道 ----------------------------
def process_simulation(sim_id, optimizer):
    """单次仿真处理流程"""

    try:
        # 生成缺陷配置
        defects = optimizer.generate_defects()
        print(f"生成缺陷: {defects}")  # 调试输出

        # 运行MATLAB仿真
        sensor_data = run_matlab_simulation(defects)

        # 处理数据
        final_data = process_file(
            file_name=f"sim_{sim_id:04d}",
            sensor_data=sensor_data,
            defeat_matrix=np.array(defects)
        )

        # 更新覆盖图
        optimizer.update_coverage(defects)

        return final_data
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None

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

    # 修改1：初始化时交换最后两个维度
    WaveletData = np.zeros((num, num, new_length, totalscal), dtype=np.complex64)  # (32,32,1000,64)
    for i, j, compressed_wavelet_transform in results:
        WaveletData[i, j, :, :] = compressed_wavelet_transform.T

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

# ---------------------------- 修改后的处理函数 ----------------------------
def process_file(file_name, sensor_data, defeat_matrix):
    """处理内存数据版本的函数"""
    num_transducers =32
    x_points = np.linspace(-25.6, 25.6, 64)
    y_points = np.linspace(0, 51.2, 64)
    transducer_positions = np.linspace(-1.5, 1.6, num_transducers)
    sensor_positions = np.linspace(-25.6, 25.6, num_transducers)
    # CWT计算
    wavelet_data = compute_cwt(sensor_data, num=32, sensor=512)

    # 相位计算（假设已有实现）
    phase_data = compute_phase(sensor_data[:, ::16, :],32,32)

    # 相位差计算（假设已有实现）
    phase_diff = compute_phase_difference(phase_data,x_points, y_points, transducer_positions, sensor_positions)

    # 保存结果
    final_data = {
        'wave': wavelet_data,
        'phase': phase_diff,
        'defeat': defeat_matrix
    }
    save_data(f"{output_dir}/final_data_{file_name}", {'final_data': final_data})
    return final_data

# 保存计算结果到文件
def save_data(output_file, data_dict):
    scipy.io.savemat(output_file, data_dict)
    print(f"数据已成功保存到 {output_file}")

# ---------------------------- 主执行流程 ----------------------------
if __name__ == "__main__":
    # 初始化覆盖优化器
    coverage_optimizer = CoverageOptimizer()

    # 并行执行仿真
    results = Parallel(n_jobs=-1)(
        delayed(process_simulation)(i, coverage_optimizer)
        for i in range(SIMULATION_TIMES)
    )

    # 监控覆盖进度
    coverage_history = []
    for i, result in enumerate(results):
        if result is not None:
            coverage = coverage_optimizer.get_coverage_rate()
            coverage_history.append(coverage)
            print(f"Simulation {i + 1}/{SIMULATION_TIMES} completed，current coverage: {coverage:.2%}")

            if coverage >= COVERAGE_THRESHOLD:
                print(f"Cverage {COVERAGE_THRESHOLD:.0%}，stopped early")
                break

    # 可视化覆盖情况
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(coverage_optimizer.coverage_map.T, origin='lower')
    plt.title(f"最终覆盖图 (覆盖率: {coverage_optimizer.get_coverage_rate():.2%})")

    plt.subplot(1, 2, 2)
    plt.plot(coverage_history)
    plt.xlabel('仿真次数')
    plt.ylabel('覆盖率')
    plt.title('覆盖率增长曲线')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'coverage_report.png'))