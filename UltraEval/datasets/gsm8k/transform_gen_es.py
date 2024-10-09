import random

from UltraEval.tasks.postprocess import GSM8KPost


def transform(data, num_sample: int, r: random.Random, dataset_name: str):
    text = f"[INST] <<SYS>>\nEres un asistente de inteligencia artificial en español, y tu responsabilidad es ayudar a los usuarios a resolver sus problemas. Cuando respondas a las preguntas, para que los usuarios que hablan español puedan entender mejor tus respuestas, solo puedes usar español para contestar. Intenta que tus respuestas sean lo más útiles y comprensibles posible, y no envíes ningún contenido dañino. Por favor, no uses inglés para contestar.\n<</SYS>>\n\n{data['question']} [/INST]"
    correct_answer = data["answer"]
    gsm8kp = GSM8KPost()
    _, processed_correct_answer = gsm8kp([], correct_answer)
    return {
        "input": text,
        "output": correct_answer,
        "processed_output": processed_correct_answer,
    }
