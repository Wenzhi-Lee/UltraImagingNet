def compute_intersectionArea(x1, y1, r1, x2, y2, r2):
    import math
    d = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    
    # 判断两个圆的位置关系
    if d > r1 + r2:
        # 两个圆不相交
        return 0
    elif d < abs(r1 - r2):
        # 一个圆完全包含在另一个圆内
        return math.pi * min(r1, r2)**2
    else:
        # 计算相交面积
        theta1 = math.acos((d**2 + r1**2 - r2**2) / (2 * d * r1))
        theta2 = math.acos((d**2 + r2**2 - r1**2) / (2 * d * r2))
        area = (r1**2 * theta1) + (r2**2 * theta2) - (0.5 * r1**2 * math.sin(2 * theta1)) - (0.5 * r2**2 * math.sin(2 * theta2))
        return area

def iou(x1, y1, r1, x2, y2, r2):
    from math import pi
    s1 = pi * r1 ** 2
    s2 = pi * r2 ** 2

    intersection = compute_intersectionArea(x1, y1, r1, x2, y2, r2)
    union = s1 + s2 - intersection

    return intersection / union