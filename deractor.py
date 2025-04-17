# -*- coding:UTF-8 -*-
"""
 @Author: CNN
 @FileName: deractor.py
 @DateTime: 2025-04-17 9:35
 @SoftWare: PyCharm
"""
import time
from functools import wraps


def timing(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"函数 {func.__name__} 执行耗时：{end_time - start_time:.4f} 秒")
        return result

    return wrapper
