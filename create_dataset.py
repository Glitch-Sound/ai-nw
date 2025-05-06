import os
import json
import random
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import LineString, Polygon
from shapely.affinity import rotate

random.seed(42)
np.random.seed(42)


def generate_sample(index, is_train=True):
    # 中心ラインを生成（曲がった形状）
    num_points = random.randint(5, 10)
    x = np.linspace(0.1, 0.9, num_points)
    y = 0.5 + 0.2 * np.sin(np.linspace(0, np.pi, num_points)) * random.uniform(-1, 1)
    center_line = np.stack([x, y], axis=1)

    # 筒形状（青ポリゴン）を作成
    offset = 0.005 + random.uniform(0.005, 0.01)
    dx = np.gradient(x)
    dy = np.gradient(y)
    norms = np.stack([-dy, dx], axis=1)
    norms = norms / np.linalg.norm(norms, axis=1, keepdims=True)
    upper = center_line + offset * norms
    lower = center_line - offset * norms
    polygon = np.concatenate([upper, lower[::-1]], axis=0)

    # 自己交差チェック
    if not Polygon(polygon).is_valid:
        return None

    # 緑ノード（両端を少し外側）
    head = center_line[0] - (center_line[1] - center_line[0]) * 0.1
    tail = center_line[-1] + (center_line[-1] - center_line[-2]) * 0.1
    nodes = [head.tolist(), tail.tolist()]

    # 赤ポリライン（中心線 + ノードで置き換え）
    polyline = center_line.copy()
    polyline[0] = head
    polyline[-1] = tail

    # 0〜1 範囲に正規化
    all_coords = np.concatenate([polygon, polyline, np.array(nodes)])
    min_xy = all_coords.min(axis=0)
    max_xy = all_coords.max(axis=0)
    scale = max_xy - min_xy
    polygon = (polygon - min_xy) / scale
    polyline = (polyline - min_xy) / scale
    nodes = (np.array(nodes) - min_xy) / scale

    # 保存先
    prefix = "train" if is_train else "test"
    os.makedirs(f"data/{prefix}", exist_ok=True)
    os.makedirs(f"data_check/{prefix}", exist_ok=True)
    fname = f"sample_{index:03d}"

    # JSON 保存
    with open(f"data/{prefix}/{fname}.json", "w") as f:
        json.dump({
            "polygon": polygon.tolist(),
            "nodes": nodes.tolist(),
            "polyline": polyline.tolist()
        }, f, indent=2)

    # PNG 保存
    plt.figure(figsize=(4, 4))
    plt.fill(*polygon.T, color='blue', alpha=0.3, label='Polygon')
    plt.plot(*np.array(polyline).T, color='red', label='Polyline')
    plt.scatter(*np.array(nodes).T, color='green', label='Nodes')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.gca().set_aspect('equal')
    plt.axis('off')
    plt.legend()
    plt.savefig(f"data_check/{prefix}/{fname}.png", bbox_inches='tight')
    plt.close()


def safe_generate(index, is_train=True, max_retry=20):
    for attempt in range(max_retry):
        result = generate_sample(index, is_train)
        if result is not None:
            return True
    return False

if __name__ == '__main__':
    for i in range(500):
        safe_generate(i, is_train=True, max_retry=100)
    for i in range(10):
        safe_generate(i, is_train=False, max_retry=100)
    print("Dataset generation complete.")
