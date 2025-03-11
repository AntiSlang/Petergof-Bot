import json
import re
import math
import pymorphy2
from nltk.stem.snowball import RussianStemmer
from yandex_cloud_ml_sdk import YCloudML
from chromadb.api import EmbeddingFunction
from chroma import create_or_update_chroma_collection, init_chroma
from dotenv import load_dotenv
from os import getenv
import os
from rout_suggestion import get_route_suggestion

load_dotenv()

YANDEX_FOLDER_ID = getenv('FOLDER')
YANDEX_AUTH = getenv('AUTH')

def is_museum_question(question: str, dialog_history: str) -> bool:

    YANDEX_FOLDER_ID = os.getenv('FOLDER')
    YANDEX_AUTH = os.getenv('AUTH')

    sdk = YCloudML(
        folder_id=YANDEX_FOLDER_ID,
        auth=YANDEX_AUTH,
    )
    model = sdk.models.completions(model_name="yandexgpt")
    model = model.configure(temperature=0.3)

    prompt = f"""
Твоя задача — определить, требует ли следующий вопрос обращения к базе данных за информацией по отдельным объектам музея (например, конкретные экспонаты, памятники, маршруты, выставки), или же он носит общий характер про музей, его время работы или последние новости и не требует такого запроса в базу по объектам.

Если вопрос пользователя требует получить подробные сведения об отдельных объектах музея для формирования ответа, выведи "True". Если же вопрос является общим, например, касается общей истории, значимости или описания музея, и не предполагает запрос к базе данных за информацией по конкретным объектам, выведи "False".

Вопрос: "{question}"
История диалога: {dialog_history}
"""


    result = model.run(prompt)
    answer = result.alternatives[0].text.strip().lower()

    if "true" in answer or "да" in answer:
        return True
    return False


def answer_from_news(question: str, dialog_history:str, news_file_path="news.json") -> str:
    try:
        with open(news_file_path, "r", encoding="utf-8") as f:
            news_data = json.load(f)
    except Exception as e:
        return f"Ошибка при чтении новостного файла: {e}"

    news_list = news_data.get("news", [])
    news_content = "\n".join(news_list) if news_list else "Нет актуальных новостей."

    museum_facts = ""

    prompt = f"""
У тебя есть следующие последние новости:
{news_content}

Также у тебя есть информация о музее:
{museum_facts}

Ты должен ответить на вопрос пользователя исходя из того, что ты бот помощник по музею и возможно используя информацию из новостей:
Вопрос пользователя: "{question}"
История диалога: {dialog_history}
"""

    sdk = YCloudML(
        folder_id=YANDEX_FOLDER_ID,
        auth=YANDEX_AUTH,
    )
    model = sdk.models.completions(model_name="yandexgpt")
    model = model.configure(temperature=0.5)
    result = model.run(prompt)
    return result.alternatives[0].text.strip()


def classify_and_answer(user_question: str, user_dialogues: list, data_chunks: list) -> str:

    if is_museum_question(user_question, user_dialogues):
        route_description, _ = get_route_suggestion(user_dialogues, data_chunks)
        return route_description
    else:
        return answer_from_news(user_question, user_dialogues)



if __name__ == "__main__":
    user_question = input("Введите ваш вопрос: ")

    user_dialogues = [
        {"user": "Расскажи, какие объекты музея сейчас работают?"}
    ]
    data_chunks = [
        {
            "name": "Павильон Большой Фонтан",
            "description": "Интересное здание с уникальными фонтанами...",
            "coordinates": {"lat": "59.8920", "lon": "29.9135"},
            "score": 0.95
        },

    ]

    answer = classify_and_answer(user_question, user_dialogues, data_chunks)
    print("Ответ бота:")
    print(answer)
