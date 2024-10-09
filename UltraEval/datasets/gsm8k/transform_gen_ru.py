import random

from UltraEval.tasks.postprocess import GSM8KPost


def transform(data, num_sample: int, r: random.Random, dataset_name: str):
    text = f"[INST] <<SYS>>\nТы являешься русским ассистентом искусственного интеллекта, твоя ответственность - помогать пользователям решать их проблемы. При отвечании на вопросы, для того, чтобы русскоязычные пользователи лучше понимали твои ответы, ты можешь использовать только русский язык. Старайся, чтобы твои ответы были максимально полезными и читаемыми, и не допускай публикации вредоносного контента. Пожалуйста, не используй английский язык для ответа.\n<</SYS>>\n\n{data['question']} [/INST]"
    correct_answer = data["answer"]
    gsm8kp = GSM8KPost()
    _, processed_correct_answer = gsm8kp([], correct_answer)
    return {
        "input": text,
        "output": correct_answer,
        "processed_output": processed_correct_answer,
    }
