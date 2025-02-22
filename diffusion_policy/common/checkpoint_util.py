from typing import Optional, Dict
import os


"""
python 常见的map：
1. 字典：key-value
2. map函数：key-value
# 这是Python的内置函数，用于对可迭代对象中的每个元素应用一个函数
numbers = [1, 2, 3]
squared = map(lambda x: x*x, numbers)  # 返回map对象
"""

class TopKCheckpointManager:
    """
    管理模型训练过程中的检查点文件,只保留性能最好的k个检查点。
    支持最大值(max)和最小值(min)两种模式。
    """
    def __init__(self,
            save_dir,              # 检查点保存目录
            monitor_key: str,      # 用于监控的指标名称(如'train_loss')
            mode='min',           # 模式:'min'保留最小值,'max'保留最大值
            k=1,                  # 保留的检查点数量
            format_str='epoch={epoch:03d}-train_loss={train_loss:.3f}.ckpt' # 检查点文件名格式
        ):
        # 验证参数
        assert mode in ['max', 'min']
        assert k >= 0

        # 初始化属性
        self.save_dir = save_dir
        self.monitor_key = monitor_key
        self.mode = mode
        self.k = k
        self.format_str = format_str
        self.path_value_map = dict()  # 存储检查点路径与对应指标值的映射
    
    def get_ckpt_path(self, data: Dict[str, float]) -> Optional[str]:
        """
        根据监控指标决定是否需要保存新的检查点。
        
        Args:
            data: 包含监控指标的字典,如{'epoch': 10, 'train_loss': 0.5}
            
        Returns:
            str or None: 如果需要保存检查点,返回保存路径;否则返回None
        """
        # 如果k=0,不保存任何检查点
        if self.k == 0:
            return None

        # 获取监控指标值和检查点保存路径
        value = data[self.monitor_key]
        ckpt_path = os.path.join(
            self.save_dir, self.format_str.format(**data))
        
        # 如果当前保存的检查点数量小于k,直接保存
        if len(self.path_value_map) < self.k:
            self.path_value_map[ckpt_path] = value
            return ckpt_path
        
        # 已达到最大保存数量k,需要判断是否替换现有检查点
        sorted_map = sorted(self.path_value_map.items(), key=lambda x: x[1])
        min_path, min_value = sorted_map[0]     # 获取最小值的检查点
        max_path, max_value = sorted_map[-1]    # 获取最大值的检查点

        # 根据模式决定是否需要删除现有检查点
        delete_path = None
        if self.mode == 'max':
            # max模式:新值大于最小值时,删除最小值检查点
            if value > min_value:
                delete_path = min_path
        else:
            # min模式:新值小于最大值时,删除最大值检查点
            if value < max_value:
                delete_path = max_path

        # 如果不需要删除任何检查点,返回None
        if delete_path is None:
            return None
        else:
            # 更新检查点映射
            del self.path_value_map[delete_path]
            self.path_value_map[ckpt_path] = value

            # 确保保存目录存在
            if not os.path.exists(self.save_dir):
                os.mkdir(self.save_dir)

            # 删除旧检查点文件
            if os.path.exists(delete_path):
                os.remove(delete_path)
            return ckpt_path
