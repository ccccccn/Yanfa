# -*- coding:UTF-8 -*-
"""
 @Author: CNN
 @FileName: analysis_core.py
 @DateTime: 2025-04-16 16:24
 @SoftWare: PyCharm
"""
import math
# analysis_core.py

import os
import subprocess

import numpy as np
import pandas as pd
from datetime import datetime

from deractor import timing


@timing
def process_folder(folder_path, input_fields, pre_data_list,
                   progress_callback=lambda x: None,
                   file_callback=lambda x: None):
    files = []
    # for root, dirs, filenames in os.walk(folder_path):
    #     for file in filenames:
    #         if file.endswith('.csv') and not file.endswith('_result.xlsx'):
    #             files.append(os.path.join(root, file))

    if os.path.isfile(folder_path):
        if folder_path.endswith('.csv'):
            files.append(folder_path)
        else:
            print("这是一个文件，但不是 CSV 文件")
    elif os.path.isdir(folder_path):
        for root, dirs, filenames in os.walk(folder_path):
            for file in filenames:
                if file.endswith('.csv') and not file.endswith('_result.xlsx'):
                    files.append(os.path.join(root, file))
    else:
        print("路径无效或不存在")



    total_files = len(files)

    for idx, file in enumerate(files):
        file_callback(f"当前处理文件：{os.path.basename(file)}")
        try:
            process_single_file(file, input_fields, pre_data_list)
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise Exception(f"处理文件 {file} 时出错：{e}") from e

        progress_callback(int((idx + 1) / total_files * 100))

# 定义函数：根据每行时间计算 Pn 和 Pn'

@timing
def process_single_file(file_path, input_fields, pre_data_list):
    df = pd.read_csv(file_path, skiprows=6, usecols=[0, 1],encoding='gbk')
    # df = fast_read_excel(file_path, skiprows=6, usecols=[0,1])
    # df = pd.read_excel(file_path, skiprows=6, usecols="A:B")

    # 处理频率偏差与实时功率
    df['频率偏差'] = 50 - df['电网频率（Hz）']
    station_capacity = float(input_fields['电站容量（MW）'].text())
    regulation_rate = float(input_fields['调差率（%）'].text())

    df['实时功率'] = (1 / (regulation_rate / 100)) * df['频率偏差'] / 50 * station_capacity

    try:
        df['时间戳'] = pd.to_datetime(df['时间戳'], errors='coerce')
    except Exception as e:
        print("时间戳转化错误：", e)
        # 如果时间列不是字符串格式，需要格式化
        # dt = pd.to_datetime(df['时间戳'].iloc[0])
        df['时间戳'] = df['时间戳'].astype(str)

    @timing
    def compute_pn_vectorized(df, pre_data_list, station_capacity, bid_capacity, time_col='时间戳'):
        """
        向 DataFrame 中添加向量化计算后的 Pn 和 Pn' 列。

        参数:
            df: pandas.DataFrame，包含时间戳列
            pre_data_list: 长度为 96 的列表，对应每个 15 分钟区间的值
            station_capacity: 机组容量
            bid_capacity: 报名容量
            time_col: 时间戳列名（默认 "时间戳"）
        返回:
            原始 df，新增两列 'Pn' 和 "Pn`"
        """

        # 转为 datetime 类型
        timestamps = pd.to_datetime(df[time_col], errors='coerce')

        # 计算每条记录对应的 index（15分钟一个点）
        pn_idx = timestamps.dt.hour * 4 + timestamps.dt.minute // 15

        # 构建有效性掩码
        pre_data_arr = np.array(pre_data_list)
        valid_mask = (pn_idx >= 0) & (pn_idx < len(pre_data_arr))

        # 初始化结果列
        pn_values = np.full(len(df), np.nan)
        pn_prime_values = np.full(len(df), np.nan)

        # 向量化赋值
        pn_values[valid_mask] = pre_data_arr[pn_idx[valid_mask]]
        pn_prime_values[valid_mask] = (station_capacity * pn_values[valid_mask]) / bid_capacity

        # 写回 DataFrame
        df['Pn'] = pn_values
        df["Pn`"] = pn_prime_values

        return df

    # Pn_value = pre_data_list[Pn_idx - 1]
    bid_capacity = float(input_fields['中标容量(MW)'].text())
    df = compute_pn_vectorized(df, pre_data_list, station_capacity, bid_capacity)
    print('Pn,Pn`处理完成')
    # 后时间续将进行响应分析
    results = analyze_response(df, input_fields)
    statistic_result = statistic_response_count(results)
    save_analysis_result(file_path, results, statistic_result)


def save_analysis_result(original_file, result_data, statistic_result):
    if not result_data:
        print(f"{original_file} 无有效分析结果，跳过保存。")
        return

    output_df = pd.DataFrame(result_data, columns=[
        '调频计数', '开始时间', '响应时长', '间隔时间', '△f', '积分电量',
        '调频里程', '△P',
        # '响应时长(30s)', '积分电量(30s)', '调频里程(30s)',
        # '响应时长(总)', '积分电量(总)', '调频里程(总)'
    ])
    output_df2 = pd.DataFrame(statistic_result, columns=[
        '分布区间', '统计个数', '越限总时间'
    ])
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_name = os.path.splitext(os.path.basename(original_file))[0] + f'_{current_time}_result.xlsx'
    output_path = os.path.join(os.path.dirname(original_file), output_name)

    with pd.ExcelWriter(path=output_path, engine='openpyxl') as writer:
        output_df.to_excel(writer, sheet_name='Sheet1', index=False, startrow=0, startcol=0)
        output_df2.to_excel(writer, sheet_name='Sheet1', index=False, startrow=0, startcol=11)
    # output_df.to_excel(output_path, index=False)
    print(f"输出已保存：{output_path}")


@timing
def analyze_response(df, input_fields):
    """
    使用向量化逻辑优化的一次调频响应分析，适合处理大规模数据。v2.1
    """
    valid_exceed_time = int(input_fields['有效越限周期(ms)'].text())
    sample_cycle_ms = int(input_fields['采样周期（ms）'].text()) \
        if input_fields['采样周期（ms）'].isEnabled() else 1000  # 默认1秒
    dead_zone = float(input_fields['一次调频死区（Hz）'].text())
    regulation_rate = float(input_fields['调差率（%）'].text())

    df = df.copy()
    df['时间戳'] = pd.to_datetime(df['时间戳'])
    df['方向'] = np.where(df['频率偏差'] >= dead_zone, 'up',
                          np.where(df['频率偏差'] <= -dead_zone, 'down', 'none'))

    # 找出所有响应区段（方向不为 none）
    df['响应标记'] = (df['方向'] != 'none').astype(int)
    df['响应切换'] = df['响应标记'].diff().fillna(0).ne(0)

    # 标记切换点的行号
    switch_indices = df.index[df['响应切换']].tolist()

    results = []
    response_id = 0

    for i in range(0, len(switch_indices), 2):
        start_idx = switch_indices[i]

        if i + 1 < len(switch_indices):
            end_idx = switch_indices[i + 1]
            segment = df.iloc[start_idx:end_idx + 1]  # 包含 end_idx 行
        else:
            segment = df.iloc[start_idx:]

        # 保证响应段中有有效方向数据
        # non_none_segment = segment[segment['方向'] != 'none']
        # if non_none_segment.empty:
        #     continue

        direction = segment['方向'].iloc[0]
        start_time = segment['时间戳'].iloc[0]
        end_time = segment['时间戳'].iloc[-1]
        duration_ms = (end_time - start_time).total_seconds() * 1000

        if duration_ms < valid_exceed_time:
            continue

        response_id += 1
        f_dev_vals = segment['频率偏差'].values
        p_vals = segment['实时功率'].values

        max_f_dev = max(f_dev_vals) if direction == 'up' else -min(f_dev_vals)
        energy = np.sum(p_vals) * (sample_cycle_ms / 1000) / 3600  # MW·h
        response_time = duration_ms / 1000
        mileage = 3.6 * energy / response_time
        delta_P = (1 / regulation_rate) * (max_f_dev / 50) * segment['Pn`'].iloc[0]

        # 计算间隔时间
        if results:
            last_time = pd.to_datetime(results[-1][1])
            last_duration = results[-1][2]
            interval_time = (start_time - last_time).total_seconds() - last_duration
        else:
            interval_time = 0.0

        results.append([
            response_id, start_time, response_time, interval_time,
            max_f_dev, energy, mileage, delta_P
        ])

    return results


# @timing
# def analyze_response(df, input_fields):
#     """
#     分析一次调频响应区间及各项指标
#     返回：结果行组成的列表
#     """
#     results = []
#     valid_exceed_time = int(input_fields['有效越限周期(ms)'].text())
#     sample_cycle_ms = int(input_fields['采样周期（ms）'].text()) \
#         if input_fields['采样周期（ms）'].isEnabled() else None
#     dead_zone = float(input_fields['一次调频死区（Hz）'].text())
#     flywheel_time = int(input_fields['飞轮标称时长（s）'].text())
#     station_capacity = float(input_fields['电站容量（MW）'].text())
#     regulation_rate = float(input_fields['调差率（%）'].text())
#
#     cnt_limit = 1000 // sample_cycle_ms if sample_cycle_ms else None
#     time_format = '%Y-%m-%d %H:%M:%S.000'
#
#     # 初始化统计变量
#     response_id = 0
#     total_response_time = 0
#     total_energy = 0
#     total_mileage = 0
#     in_response = False
#     response_direction = None  # "up" or "down"
#     record_start_time = None
#     power_sum = 0
#     buffer_f_list = []
#     buffer_p_list = []
#
#     for i in range(len(df)):
#         f_dev = df.at[i, '频率偏差']
#         power = df.at[i, '实时功率']
#         current_time = df.at[i, '时间戳']
#         # current_time = datetime.strptime(df.at[i, '时间'], time_format)
#
#         if not in_response:
#             if f_dev >= dead_zone:
#                 # 正向响应起始
#                 in_response = True
#                 response_direction = "up"
#                 record_start_time = df.at[i, '时间戳']
#                 buffer_f_list = [f_dev]
#                 buffer_p_list = [power]
#             elif f_dev <= -dead_zone:
#                 # 反向响应起始
#                 in_response = True
#                 response_direction = "down"
#                 record_start_time = df.at[i, '时间戳']
#                 buffer_f_list = [f_dev]
#                 buffer_p_list = [power]
#             continue
#
#         # 在响应区间中，是否还持续超限？
#         still_valid = (
#                 (response_direction == "up" and f_dev >= dead_zone) or
#                 (response_direction == "down" and f_dev <= -dead_zone)
#         )
#
#         if still_valid:
#             buffer_f_list.append(f_dev)
#             buffer_p_list.append(power)
#             continue
#         else:
#             # 响应中断，不满足持续性，重置
#             in_response = False
#
#         # 在响应区间中
#         buffer_f_list.append(f_dev)
#         buffer_p_list.append(power)
#         # 计算当前响应时间
#         if (df.at[i, '时间戳'] - record_start_time).total_seconds() * 1000 < 0:
#             delta_ms = (df.at[i, '时间戳'] - record_start_time).total_seconds() * 1000 + 60 * 60 * 1000
#         else:
#             delta_ms = (df.at[i, '时间戳'] - record_start_time).total_seconds() * 1000
#         # 判断是否满足结束条件（超过 1000ms）
#         if delta_ms >= valid_exceed_time:
#             # 分析结果
#             response_id += 1
#             max_f_dev = max(buffer_f_list) if response_direction == "up" else -min(buffer_f_list)
#             energy = sum(buffer_p_list) * 10 / 3600  # 积分能量
#             response_time = delta_ms / 1000
#             mileage = 3.6 * energy / response_time
#             delta_P = (1 / regulation_rate) * (max_f_dev / 50) * df.at[i, 'Pn`']
#             # 累加总量
#             total_response_time += response_time
#             total_energy += energy
#             total_mileage += mileage
#
#             if len(results) > 0:
#                 trigger_time_2 = record_start_time  # 第二次触发时间
#                 trigger_time_1 = pd.to_datetime(results[response_id - 2][1])  # 第一次触发时间
#                 response_duration_1 = pd.to_timedelta(results[response_id - 2][2], unit='s')  # 第一次响应持续时间
#                 Interval_time = ((trigger_time_2 - trigger_time_1) - response_duration_1).total_seconds()
#             else:
#                 Interval_time = pd.Timedelta(seconds=0)
#
#             # 存储当前响应数据
#             results.append([
#                 response_id, record_start_time, response_time, Interval_time,  # 间隔时间暂设为0，后面可以计算
#                 max_f_dev, energy, mileage, delta_P,
#                 # response_time, energy, mileage,
#                 # total_response_time, total_energy, total_mileage
#             ])
#             # 重置状态
#             in_response = False
#             response_direction = None
#             buffer_f_list = []
#             buffer_p_list = []
#             buffer_f_list.clear()
#             buffer_p_list.clear()
#         else:
#             buffer_f_list.clear()
#             buffer_p_list.clear()
#             continue
#
#     return results


def statistic_response_count(result):
    response_time = [row[2] for row in result]
    max_response_time = math.floor(max(response_time))
    response_interval = [0, 0.8, 3, 6, 10, 15, 30]
    for num in range(60, max_response_time + 60, 60):
        response_interval.append(num)
    bin_index = np.digitize(response_time, response_interval, right=False)
    count_per_bin = np.bincount(bin_index, minlength=len(response_interval) + 1)[1:-1]
    sum_per_bin = np.bincount(bin_index, weights=response_time, minlength=len(response_interval) + 1)[1:-1]
    # for i in range(len(response_interval) - 1):
    #     print(f"[{response_interval[i]}, {response_interval[i + 1]}): count={count_per_bin[i]}, sum={sum_per_bin[i]:.3f}")
    statistic_result = []
    for num, cnt, sum in zip(response_interval, count_per_bin, sum_per_bin):
        statistic_result.append([(num, response_interval[response_interval.index(num) + 1]), cnt, sum])
    return statistic_result
