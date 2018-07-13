import colorsys
from six.moves.urllib.parse import urlparse


GOLDEN_RATIO = 0.618033988749895
NOTIFICATION_KEYS = ('class_name', 'score')


def generate_colors(n, max_value=255):
    colors = []
    h = 0.1
    s = 0.5
    v = 0.95
    for i in range(n):
        h = 1 / (h + GOLDEN_RATIO)
        colors.append([c*max_value for c in colorsys.hsv_to_rgb(h, s, v)])

    return colors


def format_predictions(predicts):
    return ', '.join('{class_name}: {score:.2f}'.format(**p) for p in predicts)


def format_notification(predicts):
    result = []
    for p in predicts:
        result.append({key: p[key] for key in NOTIFICATION_KEYS})

    return result

def intersection_over_union(r1, r2):
    x1, y1, x12, y12 = r1
    w1 = abs(y1-y12)
    h1 = abs(x1-x12)
    x2, y2, x21, y21 = r2
    w2 = abs(y2-y21)
    h2 = abs(x2-x21)
    and_x1, and_y1 = max(x1, x2), max(y1, y2)
    and_x2, and_y2 = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
    and_w = and_x2 - and_x1
    and_h = and_y2 - and_y1
    if and_w <= 0 or and_h <= 0:
        return 0
    and_area = and_w * and_h
    area1 = w1 * h1
    area2 = w2 * h2
    or_area = area1 + area2 - and_area

    return and_area / or_area

def frame_size(f):
    x1, y1, x2, y2 = f
    w = abs(y1-y2)
    h = abs(x1-x2)
    return w*h

def find_class_by_name(name, modules):
    modules = [getattr(module, name, None) for module in modules]
    return next(a for a in modules if a)


def is_url(path):
    try:
        result = urlparse(path)
        return result.scheme and result.netloc and result.path
    except:
        return False
