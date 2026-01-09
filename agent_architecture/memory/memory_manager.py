import time
from typing import List, Dict

from llm.call_llm import  init_model_client

# 实例化千问模型客户端
client = init_model_client(provider= "qwen")


# 使用OpenAI API生成总结
def generate_summary(messages: List[Dict[str, str]]) -> str:
    try:
        # 将消息转换为适合摘要的格式
        combined_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])

        # 调用OpenAI API生成摘要
        response = client.chat.completions.create(
            model="qwen-plus",  # 模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
            messages=[
                {"role": "system",
                 "content": "你是一个专业的对话摘要器。请根据对话内容生成一个简洁的摘要，突出关键信息和主题。"},
                {"role": "user", "content": f"请总结以下对话：\n{combined_text}"}
            ],
            temperature=0.3,
            max_tokens=100
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        # 如果API调用失败，返回默认摘要
        return "（摘要生成失败，使用简化版本）"


class MemoryManager:
    """记忆管理基类"""

    def __init__(self, max_length: int = 10):
        self.max_length = max_length
        self.full_history: List[Dict[str, str]] = []

    def add_message(self, role: str, content: str, **kwargs) -> None:
        message = {"role": role, "content": content, **kwargs}
        self.full_history.append(message)

    def get_context(self) -> List[Dict[str, str]]:
        """获取用于模型输入的上下文"""
        raise NotImplementedError("子类必须实现此方法")

    def get_display_history(self) -> List[Dict[str, str]]:
        """获取用于显示的完整历史"""
        return self.full_history

    def get_context_token_count(self) -> int:
        """简单估算当前上下文token数量"""
        total_tokens = 0
        messages = self.get_context()
        for message in messages:
            content = message.get("content", "")
            # 简单估算：每4个字符算1个token
            tokens = len(content) // 4
            # 每个消息额外加5个token（用于角色和格式）
            total_tokens += tokens + 5
        return total_tokens

    def clear(self) -> None:
        """清空记忆"""
        self.full_history = []


# Trim记忆管理类 - 只保留最近的对话
class TrimMemoryManager(MemoryManager):

    def add_message(self, role: str, content: str,**kwargs) -> None:
        message = {"role": role, "content": content,**kwargs}
        self.full_history.append(message)
        # 保持窗口大小固定
        if len(self.full_history) > self.max_length:
            self.full_history = self.full_history[-self.max_length:]

    def get_context(self) -> List[Dict[str, str]]:
        # 获取当前上下文
        return self.full_history


# Summarize记忆管理类 - 总结历史对话
class SummarizeMemoryManager(MemoryManager):
    def __init__(self, max_length: int = 10):
        super().__init__(max_length)
        self.summary = ""

    def add_message(self, role: str, content: str,**kwargs) -> None:
        super().add_message(role, content,**kwargs)
        # 当消息数量超过max_length时，使用模型生成总结
        if len(self.full_history) > self.max_length:
            self._update_summary()

    def _update_summary(self) -> None:
        """更新对话总结"""
        # recent_messages = self.full_history[-self.max_length:]
        # self.full_history 列表里取出除了末尾 self.max_length 个记录之外的所有记录。
        oldest_messages = self.full_history[:-self.max_length]
        self.summary = generate_summary(oldest_messages)

    def get_context(self) -> List[Dict[str, str]]:
        if len(self.full_history) <= self.max_length:
            return self.full_history
        else:
            # 返回总结和最近的self.max_length个消息
            return [{"role": "system", "content": f"对话总结: {self.summary}"}] + self.full_history[-self.max_length:]


# Trim+Summarize记忆管理类 - 结合裁剪和总结
class TrimSummarizeMemoryManager(MemoryManager):
    def __init__(self, max_length: int = 10):
        super().__init__(max_length)
        self.summary = "对话开始"

    def add_message(self, role: str, content: str,**kwargs) -> None:
        super().add_message(role, content,**kwargs)
        # 当消息数量超过2倍max_length时，总结并裁剪
        if len(self.full_history) > 2 * self.max_length:
            self._update_summary()
            # 保留总结和最近的max_length条消息
            self.full_history = [{"role": "system", "content": f"对话总结: {self.summary}"}] + self.full_history[
                                                                                               -self.max_length:]

    def _update_summary(self) -> None:
        """更新对话总结"""
        # 总结超出max_length的部分
        old_messages = self.full_history[:-self.max_length]
        if old_messages:
            self.summary = generate_summary(old_messages)

    def get_context(self) -> List[Dict[str, str]]:
        return self.full_history


# 定期总结Summarize记忆管理类
class PeriodSummarizeMemoryManager(MemoryManager):
    def __init__(self, max_length: int = 10):
        super().__init__(max_length)
        self.summaries = []
        self.message_count_since_last_summary = 0

    def add_message(self, role: str, content: str,**kwargs) -> None:
        super().add_message(role, content,**kwargs)
        self.message_count_since_last_summary += 1

        # 达到总结间隔时进行总结
        if self.message_count_since_last_summary >= self.max_length:
            self._update_summary()
            self.message_count_since_last_summary = 0

    def _update_summary(self) -> None:
        """更新对话总结"""
        # 总结最近的消息
        recent_messages = self.full_history[-self.max_length:]
        summary = generate_summary(recent_messages)
        # 保留最近的记录
        self.full_history = recent_messages

        # 添加时间戳,把总结的消息添加到summaries中
        timestamp = time.strftime("%H:%M:%S")
        self.summaries.append(f"[{timestamp}] {summary}")

    def get_context(self) -> List[Dict[str, str]]:
        # 返回所有总结和最近的消息
        context = []
        for summary in self.summaries:
            context.append({"role": "system", "content": summary})

        context += self.full_history[-self.max_length:]
        return context


# HierarchicalMemory记忆管理类 - 分层记忆
class HierarchicalMemoryManager(MemoryManager):
    def __init__(self, short_term_length: int = 5, long_term_length: int = 5, summary_threshold: int = 10):
        super().__init__(short_term_length + long_term_length)
        self.short_term_length = short_term_length
        self.long_term_length = long_term_length
        self.summary_threshold = summary_threshold
        self.long_term_memory = []
        self.summary = "对话开始"

    def add_message(self, role: str, content: str,**kwargs) -> None:
        super().add_message(role, content,**kwargs)

        # 当短期记忆超过阈值时，将最旧的消息移到长期记忆
        if len(self.full_history) > self.short_term_length:
            old_message = self.full_history.pop(0)
            self.long_term_memory.append(old_message)

            # 保持长期记忆在限制范围内
            if len(self.long_term_memory) > self.long_term_length + self.summary_threshold:
                self._update_summary()

    def _update_summary(self) -> None:
        """更新长期记忆总结"""

        if self.long_term_memory:
            # 总结summary_threshold条记录.
            self.summary = generate_summary(self.long_term_memory[:self.summary_threshold])
            # 将总结结果加上未参加总结的长期记忆更新为新的长期记忆long_term_memory
            self.long_term_memory = ([{"role": "system", "content": f"长期记忆总结: {self.summary}"}] +
                                     self.long_term_memory[self.summary_threshold:])

    def get_context(self) -> List[Dict[str, str]]:
        # 返回总结、长期记忆和短期记忆
        context = []
        context += self.long_term_memory
        context += self.full_history[-self.short_term_length:]
        return context
