from src.rewards import reward

@reward("multiturn_reward")
def multiturn_reward(
    messages,
    maximal_turns=5,
    minimal_turns=1,
    **kwargs
    ) -> float:
    '''
    这是一个用于强化学习（RL）或大模型对齐训练中的多轮对话（或多步工具调用）奖励函数。其核心目的是：鼓励模型在指定的交互回合数（turns）内完成任务，防止模型过早退出或陷入无限死循环。
    '''
    token_messages = [msg for msg in messages if msg["kind"] == "TokenEvent"]
    num_turns = len(token_messages)
    if (num_turns >= minimal_turns) and (num_turns <= maximal_turns):
        return 1.0
    return 0.0