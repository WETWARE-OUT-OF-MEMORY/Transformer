import os
from datetime import datetime


def run_dir(base="output"):
    """脚本启动时调用，返回 output/YYYYMMDD/ 目录（自动创建）"""
    d = os.path.join(base, datetime.now().strftime("%Y%m%d"))
    os.makedirs(d, exist_ok=True)
    return d


def find_latest_ckpt(base="output", name="transformer.pt"):
    """在 base 下按日期目录名倒序找最新的 <name>，返回 (路径, 日期目录名)。

    找不到返回 (None, None)
    """
    if not os.path.isdir(base):
        return None, None
    date_dirs = sorted(
        (d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d))),
        reverse=True,
    )
    for d in date_dirs:
        ckpt = os.path.join(base, d, name)
        if os.path.exists(ckpt):
            return ckpt, d
    return None, None
