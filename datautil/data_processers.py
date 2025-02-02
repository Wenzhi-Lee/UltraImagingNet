import numpy as np
import scipy.io
import os
import re
from scipy.signal import resample, hilbert
import pywt
from joblib import Parallel, delayed

# ========================
# 全局参数配置
# ========================
CONFIG = {
    'sampling_rate': 236.72e6,   # 采样率 (Hz)
    'central_freq': 2e6,         # Morlet小波中心频率 (Hz)
    'total_scales': 64,          # CWT尺度总数
    'output_dir': 'output',      # 输出目录
    'resample_length': 1000,     # 重采样点数
    'sound_speed': 5918,         # 声速 (m/s)
    'grid_size': (64, 64),       # 相位差网格尺寸
    'transducer_range': (-1.5, 1.6),
    'sensor_range': (-25.6, 25.6),
    'data_path': r'C:\Users\user\Desktop\Matlab\bin\wave'  # 数据存储路径
}

# ========================
# 工具函数
# ========================
def create_directory(path):
    """创建输出目录"""
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"创建目录: {path}")

def load_matfile(file_path):
    """加载MATLAB数据文件"""
    try:
        data = scipy.io.loadmat(file_path)
        return data['data']
    except FileNotFoundError:
        print(f"错误: 文件 {file_path} 不存在")
        return None
    except KeyError:
        print(f"错误: 文件中未找到 'data' 字段")
        return None

def save_matfile(output_path, data_dict):
    """保存数据到MAT文件"""
    scipy.io.savemat(output_path, data_dict)
    print(f"数据已保存至: {output_path}")

# ========================
# 核心处理函数
# ========================
def compute_cwt_for_sensor(i, j, sensor_data, scales, sampling_rate):
    """单个传感器的连续小波变换计算"""
    # 数据重采样
    raw_data = sensor_data[i, j, :]
    resampled = resample(raw_data, CONFIG['resample_length'])
    
    # 计算新采样率
    new_sr = sampling_rate * CONFIG['resample_length'] / raw_data.shape[0]
    
    # 执行小波变换
    coefficients, _ = pywt.cwt(resampled, scales, 'morl', 1.0 / new_sr)
    
    # 压缩结果
    indices = np.linspace(0, coefficients.shape[1]-1, CONFIG['resample_length'], dtype=int)
    return i, j//16, coefficients[:, indices].T  # 直接返回转置后的数据

def compute_cwt_pipeline(sensor_data, num_transducers=32, num_sensors=512):
    """CWT计算流水线"""
    # 参数初始化
    N = sensor_data.shape[2]
    scales = _calculate_scales()
    
    # 并行计算
    results = Parallel(n_jobs=-1)(
        delayed(compute_cwt_for_sensor)(i, j, sensor_data, scales, CONFIG['sampling_rate'])
        for i in range(num_transducers) 
        for j in range(0, num_sensors, 16)
    )

    # 结果重组
    wavelet_data = np.zeros(
        (num_transducers, num_transducers, CONFIG['resample_length'], CONFIG['total_scales']),
        dtype=np.complex64
    )
    
    for i, j, transform in results:
        wavelet_data[i, j] = transform
    
    return wavelet_data

def _calculate_scales():
    """计算小波尺度序列"""
    fc = pywt.central_frequency('morl')
    cparam = 2 * fc * CONFIG['total_scales']
    return cparam / np.arange(CONFIG['total_scales']+1, 1, -1)

# ========================
# 相位处理模块
# ========================
def compute_phase(sensor_data):
    """计算相位信息"""
    num_t, num_s, _ = sensor_data.shape
    phase_data = np.zeros_like(sensor_data, dtype=np.float32)
    
    for i in range(num_t):
        for j in range(num_s):
            analytic = hilbert(sensor_data[i, j, :])
            phase_data[i, j, :] = np.angle(analytic)
    
    return phase_data

def compute_phase_difference(phase_data, num_transducers):
    """计算相位差矩阵"""
    # 生成空间网格
    x = np.linspace(*CONFIG['sensor_range'], CONFIG['grid_size'][0])
    y = np.linspace(0, 51.2, CONFIG['grid_size'][1])
    tx_positions = np.linspace(*CONFIG['transducer_range'], num_transducers)
    
    phase_diff = np.zeros((num_transducers, num_transducers, *CONFIG['grid_size']))
    
    for i, tx_pos in enumerate(tx_positions):
        grid_x, grid_y = np.meshgrid(x, y)
        d_t = np.hypot(grid_x - tx_pos, grid_y)
        
        for j, rx_pos in enumerate(x):
            d_r = np.hypot(grid_x - rx_pos, grid_y)
            travel_time = (d_t + d_r) * 1e-3 / CONFIG['sound_speed']
            sample_idx = np.round(travel_time * CONFIG['sampling_rate']).astype(int)
            
            # 边界处理
            np.clip(sample_idx, 0, phase_data.shape[2]-1, out=sample_idx)
            phase_diff[i, j] = phase_data[i, j, sample_idx]
    
    return phase_diff

# ========================
# 数据生成管道
# ========================
def process_file(file_name, defect_dict):
    """单个文件处理流水线"""
    # 初始化输出目录
    create_directory(CONFIG['output_dir'])
    
    # 加载数据
    file_path = os.path.join(CONFIG['data_path'], file_name)
    sensor_data = load_matfile(file_path)
    if sensor_data is None:
        return None

    # 小波变换处理
    wavelet_data = compute_cwt_pipeline(sensor_data)
    
    # 相位处理
    phase_data = compute_phase(sensor_data[:, ::16, :])
    phase_diff = compute_phase_difference(phase_data, num_transducers=32)
    
    # 结果打包
    final_data = {
        'wave': wavelet_data,
        'phase': phase_diff,
        'defect': defect_dict[file_name]
    }
    
    # 保存结果
    output_path = os.path.join(CONFIG['output_dir'], f'final_{file_name}.mat')
    save_matfile(output_path, final_data)
    
    # 打印形状信息
    shapes = {k: v.shape for k, v in final_data.items()}
    print(f"处理完成: {file_name}\n数据维度: {shapes}")
    
    return final_data

# ========================
# 缺陷参数解析模块
# ========================
def parse_defect_parameters(file_names):
    """自动解析文件名生成缺陷参数"""
    pattern = r'x(\d+)y(\d+)(?:_(\d+))?$'
    defect_dict = {}
    
    for fn in file_names:
        matrix = np.zeros((10, 3))
        
        # 特殊处理无坐标文件
        if fn in ('sensor_data_1', 'sensor_data_2'):
            _handle_special_case(matrix, fn)
        else:
            # 正则解析参数
            if match := re.search(pattern, fn):
                x = int(match[1])
                y = int(match[2])
                radius = int(match[3]) if match[3] else 20
                matrix[0] = [x, y, radius]
        
        defect_dict[fn] = matrix
    
    return defect_dict

def _handle_special_case(matrix, file_name):
    """处理特殊文件案例"""
    if file_name == 'sensor_data_1':
        offsets = [(160 + i*60, 346 - i*90) for i in range(4)]
    else:  # sensor_data_2
        offsets = [(160 + i*60, 76 + i*90) for i in range(4)]
    
    for i, (x, y) in enumerate(offsets):
        matrix[i] = [x, y, 20]

# ========================
# 主程序
# ========================
if __name__ == "__main__":
    # 文件列表
    FILE_LIST = [
        'sensor_data_x346y346', 'sensor_data_x346y346_10', 'sensor_data_x346y346_40',
        'sensor_data_x406y76_10', 'sensor_data_x406y121_40', 'sensor_data_1',
        'sensor_data_2', 'sensor_data_x106y346_10', 'sensor_data_x160y346',
        'sensor_data_x166y76', 'sensor_data_x166y76_10', 'sensor_data_x166y76_40',
        'sensor_data_x166y391', 'sensor_data_x206y256_10', 'sensor_data_x206y301_40',
        'sensor_data_x220y256', 'sensor_data_x226y166', 'sensor_data_x226y166_10',
        'sensor_data_x226y166_40', 'sensor_data_x280y166', 'sensor_data_x286y256',
        'sensor_data_x286y256_10', 'sensor_data_x286y256_40', 'sensor_data_x306y166_10',
        'sensor_data_x306y211_40', 'sensor_data_x340y76'
    ]

    # 生成缺陷参数
    defect_params = parse_defect_parameters(FILE_LIST)

    # 处理所有文件
    for file in FILE_LIST:
        print(f"\n{'='*40}\n正在处理文件: {file}\n{'='*40}")
        result = process_file(file, defect_params)
