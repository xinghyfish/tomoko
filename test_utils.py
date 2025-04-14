import time
from cProfile import label

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties

font = "/home/magus/xinghy-workspace/tomoko/black.ttf"
font_prop = FontProperties(fname=font)


def time_measurement(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        execution_time = end - start
        print(f"{func.__name__} execution time: {execution_time}")
        return result
    return wrapper


def plot_results(results, label, task_type, min_val, max_val):
    plt.figure(dpi=300)
    x = np.linspace(0, 7, 8)
    colors = ['red', 'green', 'blue', 'cyan', 'magenta', 'yellow', 'black']
    for i in range(len(results)):
        plt.plot(x, results[i], color=colors[i])
    plt.title(task_type + "中轴对齐测试", fontproperties=font_prop)
    plt.xlabel("迭代次数", fontproperties=font_prop)
    plt.ylabel(label, fontproperties=font_prop)
    plt.ylim(ymin=min_val, ymax=max_val)
    plt.grid(True)
    plt.show()


def test_object_detect():
    results = [
        [624, 348, 316, 316, 324, 320, 320, 320],
        [192, 322, 322, 320, 320, 320, 320, 320],
        [592, 276, 324, 320, 320, 320, 320, 320],
        [32, 312, 319, 320, 320, 320, 320, 320],
        [412, 325, 322, 322, 322, 320, 320, 320],
        [386, 318, 318, 316, 318, 316, 318, 318],
        [70, 324, 324, 318, 320, 320, 320, 320]
    ]

    plot_results(results, "检测框中心水平像素值", "目标检测", 0, 640)
    for i in range(len(results)):
        for j in range(len(results[i])):
            results[i][j] = abs(results[i][j] - 320)
    plot_results(results, "水平偏移量", "目标检测", -5, 320)


def test_object_segment():
    results = [
        [324, 320, 320, 320, 320, 323, 323, 322],
        [609, 326, 322, 322, 322, 322, 322, 322],
        [509, 477, 321, 321, 321, 321, 321, 321],
        [98, 318, 321, 320, 320, 321, 320, 320],
        [633, 340, 320, 320, 320, 320, 320, 320],
        [38, 323, 323, 321, 321, 321, 321, 321],
        [566, 321, 321, 320, 320, 320, 320, 320],
    ]

    plot_results(results, "掩码中心水平像素值", "语义分割", 0, 640)
    for i in range(len(results)):
        for j in range(len(results[i])):
            results[i][j] = abs(results[i][j] - 320)
    plot_results(results, "水平偏移量", "语义分割", -5, 320)


if __name__ == '__main__':
    test_object_detect()
    test_object_segment()
