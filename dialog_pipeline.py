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

GREETING_PATTERNS = {
    'ru': re.compile(r'\b(привет|здравствуй|добрый день|доброе утро|добрый вечер|здравствуйте|приветствую)\b',
                     re.IGNORECASE),
    'en': re.compile(r'\b(hello|hi|greetings|good morning|good day|good evening)\b', re.IGNORECASE)
}


def is_greeting_in_message(text, language='ru'):
    pattern = GREETING_PATTERNS.get(language, GREETING_PATTERNS['en'])
    return bool(pattern.search(text))


def classify_question_type(question: str, dialog_history: str) -> str:
    sdk = YCloudML(
        folder_id=YANDEX_FOLDER_ID,
        auth=YANDEX_AUTH,
    )
    model = sdk.models.completions(model_name="yandexgpt", model_version="rc")
    model = model.configure(temperature=0.1)

    prompt = f"""
Задача: Определить тип вопроса пользователя по отношению к музею-заповеднику "Петергоф".

Возможные типы вопросов:
1. "museum" - вопрос о конкретных объектах музея (фонтаны, павильоны, экспонаты, скульптуры) не связанные с их режимом работы, расписанием, открытием
2. "route" - вопрос о маршрутах или явная просьба помочь с составлением маршрута
3. "general" - общие вопросы о музее, режиме работы, билетах, новостях и т.д.

Примеры вопросов типа "museum":
- "Расскажи о Большом каскаде в Петергофе"
- "Где находится павильон Эрмитаж?"
- "Какие экспонаты можно увидеть в музее Императорских яхт?"
- "Что представляет собой фонтан Самсон?"

Примеры вопросов типа "route":
- "Помоги составить маршрут по парку"
- "Как лучше обойти все основные фонтаны за 3 часа?"
- "Предложи маршрут для посещения дворцов"
- "Какой маршрут посоветуешь для первого посещения?"
- "Составь для меня план прогулки по парку"
- "Куда лучше пойти в первую очередь?"
- "Что стоит посмотреть в первую очередь?"

Примеры вопросов типа "general":
- "Расписание работы объекта?"
- "Открыт ли фонтан/музей/дворец?"
- "Режим работы объекта?"
- "Когда открывается музей?"
- "Сколько стоит вход в парк?"
- "Есть ли сегодня какие-то мероприятия?"
- "Где находится туалет в парке?"
- "Можно ли с собакой в парк?"
- "Есть ли в музее кафе?"

Проанализируй следующий вопрос и историю диалога. Ответь только "museum", "route" или "general".

Вопрос: "{question}"
История диалога: {dialog_history}
"""

    result = model.run(prompt)
    answer = result.alternatives[0].text.strip().lower()
    print(f"Классификация вопроса {question}: {answer}")

    if "museum" in answer:
        return "museum"
    elif "route" in answer:
        return "route"
    else:
        return "general"


def is_museum_question(question: str, dialog_history: str) -> bool:
    question_type = classify_question_type(question, dialog_history)
    return question_type == "museum"


def is_route_question(question: str, dialog_history: str) -> bool:
    question_type = classify_question_type(question, dialog_history)
    return question_type == "route"


def answer_from_news(question: str, dialog_history: str, greeting_style="friendly", news_file_path="news.json", tickets_file_path="tickets.json") -> str:
    try:
        with open(news_file_path, "r", encoding="utf-8") as f:
            news_data = json.load(f)
    except Exception as e:
        return f"Ошибка при чтении новостного файла: {e}"

    news_list = news_data.get("news", [])
    news_content = '\n'.join([f'{i + 1}) {j}' for i, j in enumerate(news_list)]) if news_list else "Нет актуальных новостей."

    try:
        with open(tickets_file_path, "r", encoding="utf-8") as f:
            tickets_data = json.load(f)
    except Exception as e:
        return f"Ошибка при чтении файла билетов: {e}"

    tickets_list = tickets_data.get("data", [])
    tickets_content = "\n".join(tickets_list) if tickets_list else "Нет актуальной информации по билетам и расписаниям."

    greeting_instruction = ""
    if greeting_style == "none":
        greeting_instruction = """
            Важно: не используй никаких приветствий в начале сообщения, если пользователь не поздоровался первым 
            Начинай ответ сразу с информации по существу вопроса.
            Не используй фразы типа "Здравствуйте", "Привет", "Добрый день" и другие формы приветствий, если пользователь не поздоровался первым 
            """
    elif greeting_style == "brief":
        greeting_instruction = """
            Используй только очень краткое приветствие, если это первое сообщение в диалоге.
            В последующих сообщениях не используй приветствий вообще, если пользователь не поздоровался первым 
            Избегай чрезмерной эмоциональности и длинных фраз вежливости.
            """
    elif greeting_style == "friendly":
        greeting_instruction = """
            Используй дружелюбный, но профессиональный тон в ответе. 
            Если пользователь поприветствовал тебя, обязательно ответь на приветствие.
            Если это первое взаимодействие или пользователь задаёт новую тему, можешь использовать краткое
            приветствие, чтобы установить дружелюбный тон.
            """
    elif greeting_style == "very_friendly":
        greeting_instruction = """
            Используй очень дружелюбный и тёплый тон. Обязательно начни с приветствия.
            """

    is_first_message = dialog_history.count('user:') <= 1

    continuity_instruction = ""
    if not is_first_message:
        continuity_instruction = """
        Это продолжение диалога. Не представляйся заново и не повторяй информацию,
        которую ты уже сообщал ранее. Отвечай по существу нового вопроса.
        """

    user_greeting = is_greeting_in_message(question)
    greeting_response = ""
    if user_greeting:
        greeting_response = """
        Пользователь поприветствовал тебя, обязательно ответь на приветствие в начале своего сообщения.
        """

    prompt = f"""
    У тебя есть следующие последние новости - всегда прикладывай ссылку на используемую новость!:
    {news_content}
    
    Также у тебя есть информация по билетам и расписаниям - если по объекту есть конкретное расписание на сегодняшнее число, то упомяни это:
    {tickets_content}

    Ты бот-помощник по музею-заповеднику "Петергоф". Твоя задача - отвечать на вопросы посетителей.

    {greeting_instruction}
    {greeting_response if user_greeting else ""}
    {continuity_instruction}

    Отвечай лаконично и по существу, избегая ненужных длинных введений, но сохраняя дружелюбный тон.
    Предоставляй точную и полезную информацию, которая требуется пользователю. Если ты используешь новость, то обязательно приложи ссылку на неё!

    Вопрос пользователя: "{question}"
    История диалога: {dialog_history}
    """

    sdk = YCloudML(
        folder_id=YANDEX_FOLDER_ID,
        auth=YANDEX_AUTH,
    )
    model = sdk.models.completions(model_name="yandexgpt", model_version="rc")
    model = model.configure(temperature=0.4)
    result = model.run(prompt)
    return result.alternatives[0].text.strip()