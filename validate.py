import asyncio
from yandex_cloud_ml_sdk import YCloudML
from os import getenv, remove
from dotenv import load_dotenv
import json

load_dotenv()

YANDEX_FOLDER_ID = getenv('FOLDER')
YANDEX_AUTH = getenv('AUTH')

def parse_json_output(text):
    text = text.strip()
    if '```json' in text:
        text = text[text.rfind('```json') + len('```json'): text.rfind('```')]
    else:
        text = text[text.find('```') + len('```'): text.rfind('```')]

    return text.strip()

async def judge_classification(llm_model, question: str, dialog_history, classifier_label: str) -> str:
    prompt = f"""
Задача: Оценить корректность классификации вопроса по отношению к музею-заповеднику "Петергоф".

<Вопрос>
{question}
</Вопрос>

<Диалог>
{dialog_history}
</Диалог>

Классификация, предложенная классификатором:
{classifier_label}

Возможные типы вопросов:
1. "museum" – вопрос о конкретных объектах музея (фонтаны, павильоны, экспонаты, скульптуры), не связанные с режимом работы, расписанием или открытием.
2. "route" – вопрос о маршрутах или просьба помочь с составлением маршрута.
3. "general" – общие вопросы о музее, режиме работы, билетах, новостях и т.д.

Правила:
1. Если классификация корректна — ответить "Верно".
2. Если классификация неверна — ответить "Неверно. Правильная классификация: <museum/route/general>".

Формат ответа (необходимо вернуть строго JSON):
json```
{{
    "thoughts": "<Опиши в 5 предложениях, почему выбрана та или иная классификация>",
    "rating": "<Строка: 'Верно' или 'Неверно'>"
}}
```
"""
    result = await asyncio.to_thread(llm_model.run, prompt)
    output = result.alternatives[0].text.strip()

    try:
        json_objects = json.loads(parse_json_output(output))
    except json.JSONDecodeError:
        return f"Ошибка парсинга JSON: {output}"

    rating = json_objects.get("rating", "").strip()
    # print(f"Судья оценил классификацию '{classifier_label}' для вопроса '{question}' как: {rating}")
    return rating

async def evaluate_answer_news(llm_model,
                               question: str,
                               dialog_history,
                               bot_answer: str,
                               news_data: dict,
                               tickets_data: dict) -> str:
    news_str = json.dumps(news_data, ensure_ascii=False, indent=2)
    tickets_str = json.dumps(tickets_data, ensure_ascii=False, indent=2)

    prompt = f"""
Задача: Оценить полезность ответа бота по музею-заповеднику "Петергоф" на вопрос пользователя, используя только данные из двух файлов.

Шкала оценки:
0 – Ответ абсолютно не полезен. Если в ответе присутствует информация, которой нет в файлах или модель придумала ложные факты, оцени как 0.
1 – Ответ по теме, но не полностью решает задачу пользователя (требуется уточнение).
2 – Идеальный ответ: пользователь получил всю требуемую информацию из предоставленных данных.

Даны:
<Вопрос>
{question}
</Вопрос>

<Диалог>
{dialog_history}
</Диалог>

<Ответ>
{bot_answer}
</Ответ>

<NEWS>
{news_str}
</NEWS>

<TICKETS>
{tickets_str}
</TICKETS>

Формат ответа (строго JSON):
json```
{{
    "thoughts": "<Опиши в 5 предложениях, как ответ соответствует данным новостей и расписаний>",
    "rating": "<Одно число: 0, 1 или 2>"
}}
```
"""
    result = await asyncio.to_thread(llm_model.run, prompt)
    output = result.alternatives[0].text.strip()

    try:
        json_objects = json.loads(parse_json_output(output))
    except json.JSONDecodeError:
        return f"Ошибка парсинга JSON: {output}"

    rating = str(json_objects.get("rating", "")).strip()
    # print(f"Оценка полезности ответа (новости/билеты): {rating}")
    return rating


async def evaluate_contextual_answer_usefulness(llm_model,
                                                question: str,
                                                dialog_history,
                                                relevant_context: str,
                                                bot_answer: str) -> str:
    prompt = f"""
Задача: Оценить полезность ответа бота по музею-заповеднику "Петергоф" на вопрос пользователя на основе предоставленного контекста.

Шкала:
0 – Ответ абсолютно не полезен. Если в ответе содержатся сведения, которых нет в контексте или информация выдумана, оцени как 0.
1 – Ответ по теме, но не полностью раскрывает требуемую информацию (может потребоваться уточнение).
2 – Идеальный ответ: полностью отвечает на вопрос и включает всю информацию, содержащуюся в контексте.

Важно: Если пользователь просит составить или изменить маршрут с объектами для просмотра, необходимо оценивать строго – если недостаёт нужных объектов или добавлены лишние, оценка 0.

Даны:
<Вопрос>
{question}
</Вопрос>

<Диалог>
{dialog_history}
</Диалог>

<КОНТЕКСТ>
{relevant_context}
</КОНТЕКСТ>

<Ответ>
{bot_answer}
</Ответ>

Формат ответа (строго JSON):
json```
{{
    "thoughts": "<Опиши в 5 предложениях, чего хочет пользователь, проведи анализ текущего маршрута, обозначь недостающие элементы и перечисли объекты для добавления>",
    "new_objects": "<Перечисли названия объектов через запятую или пустая строка, если добавлять не нужно>",
    "rating": "<Одно число: 0, 1 или 2>"
}}
```
"""
    result = await asyncio.to_thread(llm_model.run, prompt)
    output = result.alternatives[0].text.strip()

    try:
        json_objects = json.loads(parse_json_output(output))
    except json.JSONDecodeError:
        return f"Ошибка парсинга JSON: {output}"

    rating = str(json_objects.get("rating", "")).strip()
    # print(f"Оценка полезности ответа с контекстом: {rating}")
    return rating


async def main():
    sdk = YCloudML(
        folder_id=YANDEX_FOLDER_ID,
        auth=YANDEX_AUTH,
    )
    llm_model_low_temp = sdk.models.completions(model_name="yandexgpt", model_version="rc")
    llm_model_low_temp = llm_model_low_temp.configure(temperature=0.0)

    # Пример использования для judge_classification
    classification = await judge_classification(
        llm_model_low_temp,
        question="Расскажи о Большом каскаде в Петергофе",
        dialog_history="user: Привет, расскажи подробнее.\nbot: Конечно, расскажу.",
        classifier_label="museum"
    )
    print("Классификация:", classification)

    # Пример данных для evaluate_answer_news
    news_data = {
        "news": [
            "Выставка современных художников открыта. Подробнее: https://example.com/exhibition",
            "Новые экспозиции в музее императорских яхт."
        ]
    }
    tickets_data = {
        "data": [
            "Расписание: с 10:00 до 18:00",
            "Билеты: https://example.com/tickets"
        ]
    }
    rating_news = await evaluate_answer_news(
        llm_model_low_temp,
        question="Какая новость сегодня по поводу выставки в Петергофе?",
        dialog_history="user: Расскажи новости музея.\nbot: Вот последние новости.",
        bot_answer="Сегодня в Петергофе открыта выставка современных художников. Подробности: https://example.com/exhibition",
        news_data=news_data,
        tickets_data=tickets_data
    )
    print("Оценка ответа (новости):", rating_news)

    contextual_rating = await evaluate_contextual_answer_usefulness(
        llm_model_low_temp,
        question="Составь маршрут для осмотра главных фонтанов",
        dialog_history="user: Мне нужен маршрут по парку.\nbot: Расскажу подробнее.",
        relevant_context=("Фонтан Самсон расположен в центре парка. "
                          "Фонтан Непокорённых находится у входа. "
                          "Дополнительная информация доступна на https://example.com/fountains"),
        bot_answer="Рекомендуется посетить фонтан Самсон и фонтан Непокорённых. Дополнительно можно добавить фонтан Мудрости."
    )
    print("Оценка ответа (контекст):", contextual_rating)


if __name__ == "__main__":
    asyncio.run(main())
