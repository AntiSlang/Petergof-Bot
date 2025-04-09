import asyncio
import json
import random
import os
from datetime import datetime
from pathlib import Path
import aiohttp
import warnings
import pandas as pd
from typing import List, Dict, Any, Optional
from yandex_cloud_ml_sdk import YCloudML
from dotenv import load_dotenv
from os import getenv
import sys
import traceback

load_dotenv()

from validate import (
    judge_classification,
    evaluate_answer_news,
    evaluate_contextual_answer_usefulness
)

from dialog_pipeline import classify_question_type, answer_from_news
from chroma import init_chroma
from rout_suggestion import get_route_suggestion
from utils import create_json_chunks

SOY_TOKEN = getenv('SOY_TOKEN')
TEST_SIZE = 30
RESULTS_DIR = "test_results"
DETAILED_LOGS = True


async def get_chatgpt_response(messages: list,
                               model: str = 'gpt-4o',
                               temperature: float = 0.1,
                               timeout: int = 600,
                               max_tokens: int = 16000) -> str:
    headers = {'Authorization': f'OAuth {SOY_TOKEN}'}

    model_family = 'openai'
    if 'deepseek' in model.lower():
        model_family = 'together'
        model = 'deepseek-ai/deepseek-r1'

    if 'meta-llama' in model.lower():
        model_family = 'together'

    url = f'http://api.eliza.yandex.net/{model_family}/v1/chat/completions'

    data = {
        'model': model,
        'temperature': temperature,
        'messages': messages,
        'max_tokens': max_tokens
    }
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(url, json=data, headers=headers, timeout=timeout) as response:
                result = await response.json()
                if response.status == 200:
                    answer = result['response']['choices'][0]['message']['content']
                    return answer
                else:
                    error_msg = result.get('response', {}).get('error', {}).get('message', 'Unknown error')
                    print(f"API Error: {error_msg}")
                    return ''
        except asyncio.exceptions.TimeoutError:
            warnings.warn('Timeout Exceeded. Return empty value')
            return ''
        except Exception as e:
            print(f'Error in API call: {e}')
            raise e


async def generate_test_dialogs(count_per_type: int = 10) -> List[Dict]:
    type_prompts = {
        "museum": """
        Создай {count} разных вопросов о конкретных достопримечательностях музея-заповедника "Петергоф".
        Вопросы должны касаться фонтанов, павильонов, дворцов, экспонатов и других объектов.
        НЕ включай вопросы о режиме работы, билетах или расписаниях.

        Примеры:
        - "Что такое Большой каскад в Петергофе?"
        - "Расскажи историю павильона Эрмитаж"
        - "Какие статуи находятся возле фонтана Самсон?"
        """,

        "route": """
        Создай {count} разных вопросов, связанных с составлением маршрутов по музею-заповеднику "Петергоф".
        Вопросы должны касаться планирования посещения, рекомендуемых маршрутов и последовательности осмотра объектов.

        Примеры:
        - "Посоветуй маршрут для посещения парка с детьми"
        - "Как лучше спланировать осмотр всех главных фонтанов за 3 часа?"
        - "Составь план посещения Петергофа для первого раза"
        """,

        "general": """
        Создай {count} разных общих вопросов о музее-заповеднике "Петергоф".
        Включи вопросы о режиме работы, билетах, расписании, правилах посещения, инфраструктуре и новостях.

        Примеры:
        - "Когда открываются фонтаны в Петергофе?"
        - "Сколько стоит билет в Нижний парк?"
        - "Можно ли приходить с домашними животными?"
        """
    }

    all_dialogs = []

    for q_type, prompt_template in type_prompts.items():
        prompt = prompt_template.format(count=count_per_type)

        messages = [{"role": "user", "content": prompt}]
        response = await get_chatgpt_response(messages)

        if not response:
            print(f"Failed to generate questions for {q_type}")
            continue

        questions = [q.strip() for q in response.split('\n') if q.strip() and not q.startswith('-')]

        questions = [q[q.find(' ') + 1:] if q[0].isdigit() and q[1] in '.) ' else q for q in questions]
        questions = [q[2:] if q.startswith('- ') else q for q in questions]

        for q in questions[:count_per_type]:
            if random.random() < 0.3:
                history_prompt = f"Придумай короткий диалог между пользователем и ботом (1-2 реплики), который мог бы предшествовать вопросу: '{q}'"
                history_response = await get_chatgpt_response([{"role": "user", "content": history_prompt}])
                history = history_response.strip() if history_response else ""
            else:
                history = ""

            all_dialogs.append({
                "question": q,
                "history": history,
                "expected_type": q_type
            })

    random.shuffle(all_dialogs)
    return all_dialogs


async def test_classification(dialog: Dict, llm_model) -> Dict:
    result = {
        "question": dialog["question"],
        "history": dialog["history"],
        "expected_type": dialog["expected_type"],
        "classification": {
            "predicted_type": "",
            "validation": "",
            "is_correct": False
        }
    }

    try:
        predicted_type = classify_question_type(dialog["question"], dialog["history"])
        result["classification"]["predicted_type"] = predicted_type

        validation = await judge_classification(
            llm_model,
            question=dialog["question"],
            dialog_history=dialog["history"],
            classifier_label=predicted_type
        )
        result["classification"]["validation"] = validation
        result["classification"]["is_correct"] = validation == "Верно"
    except Exception as e:
        traceback.print_exc()
        result["classification"]["error"] = str(e)

    return result


async def test_news_answer(dialog: Dict, llm_model, news_data: Dict, tickets_data: Dict) -> Dict:
    result = {
        "question": dialog["question"],
        "history": dialog["history"],
        "expected_type": dialog["expected_type"],
        "news_answer": {
            "bot_answer": "",
            "evaluation": "",
            "rating": None
        }
    }

    try:
        bot_answer = answer_from_news(
            dialog["question"],
            dialog["history"],
            greeting_style="none"
        )
        result["news_answer"]["bot_answer"] = bot_answer

        evaluation = await evaluate_answer_news(
            llm_model,
            question=dialog["question"],
            dialog_history=dialog["history"],
            bot_answer=bot_answer,
            news_data=news_data,
            tickets_data=tickets_data
        )
        result["news_answer"]["evaluation"] = evaluation

        try:
            result["news_answer"]["rating"] = int(evaluation) if evaluation.isdigit() else None
        except:
            pass
    except Exception as e:
        traceback.print_exc()
        result["news_answer"]["error"] = str(e)

    return result


async def test_rag_answer(dialog: Dict, llm_model, chroma_collection) -> Dict:
    result = {
        "question": dialog["question"],
        "history": dialog["history"],
        "expected_type": dialog["expected_type"],
        "rag_answer": {
            "bot_answer": "",
            "context": "",
            "evaluation": "",
            "rating": None
        }
    }

    try:
        query_results = chroma_collection.query(
            query_texts=[dialog["question"]],
            n_results=5
        )
        retrieved_docs = query_results.get('documents', [[]])[0]
        context = "\n\n".join(retrieved_docs)
        result["rag_answer"]["context"] = context

        prompt = f"""
        Ты - виртуальный помощник по музею-заповеднику Петергоф.

        Вопрос пользователя: "{dialog["question"]}"

        Релевантный контекст для ответа:
        {context}

        Дай краткий и точный ответ на основе предоставленного контекста.
        """
        llm_result = llm_model.run(prompt)
        bot_answer = llm_result.alternatives[0].text.strip()
        result["rag_answer"]["bot_answer"] = bot_answer

        evaluation = await evaluate_contextual_answer_usefulness(
            llm_model,
            question=dialog["question"],
            dialog_history=dialog["history"],
            relevant_context=context,
            bot_answer=bot_answer
        )
        result["rag_answer"]["evaluation"] = evaluation

        try:
            result["rag_answer"]["rating"] = int(evaluation) if evaluation.isdigit() else None
        except:
            pass
    except Exception as e:
        traceback.print_exc()
        result["rag_answer"]["error"] = str(e)

    return result


async def test_route_answer(dialog: Dict, llm_model) -> Dict:
    result = {
        "question": dialog["question"],
        "history": dialog["history"],
        "expected_type": dialog["expected_type"],
        "route_answer": {
            "bot_answer": "",
            "evaluation": "",
            "rating": None
        }
    }

    try:
        data_chunks = create_json_chunks()
        coordinates = ['59.891802', '29.913220']

        user_dialogues = [{"user": dialog["question"]}]
        if dialog["history"]:
            history_lines = dialog["history"].split('\n')
            for line in history_lines:
                if line.lower().startswith(('user:', 'пользователь:')):
                    user_dialogues.append({"user": line.split(':', 1)[1].strip()})
                elif line.lower().startswith(('bot:', 'бот:')):
                    user_dialogues.append({"bot": line.split(':', 1)[1].strip()})

        answer_text, route_json = get_route_suggestion(
            user_dialogues,
            data_chunks,
            initial_coordinates=coordinates
        )
        bot_answer = answer_text.replace('*', '')
        result["route_answer"]["bot_answer"] = bot_answer

        route_context = json.dumps(data_chunks[:2], ensure_ascii=False)

        evaluation = await evaluate_contextual_answer_usefulness(
            llm_model,
            question=dialog["question"],
            dialog_history=dialog["history"],
            relevant_context=route_context,
            bot_answer=bot_answer
        )
        result["route_answer"]["evaluation"] = evaluation

        try:
            result["route_answer"]["rating"] = int(evaluation) if evaluation.isdigit() else None
        except:
            pass
    except Exception as e:
        traceback.print_exc()
        result["route_answer"]["error"] = str(e)

    return result


async def run_tests() -> Dict:
    os.makedirs(RESULTS_DIR, exist_ok=True)

    sdk = YCloudML(folder_id=getenv('FOLDER'), auth=getenv('AUTH'))
    llm_model = sdk.models.completions(model_name="yandexgpt", model_version="rc")
    llm_model = llm_model.configure(temperature=0.1)

    print("Initializing Chroma collection...")
    chroma_collection = init_chroma(remote=True)

    print("Loading data...")
    try:
        with open('news.json', 'r', encoding='utf-8') as f:
            news_data = json.load(f)
    except Exception as e:
        print(f"Error loading news data: {e}")
        news_data = {"news": [], "number": 1}

    try:
        with open('tickets.json', 'r', encoding='utf-8') as f:
            tickets_data = json.load(f)
    except Exception as e:
        print(f"Error loading tickets data: {e}")
        tickets_data = {"data": []}

    print(f"Generating {TEST_SIZE} test dialogs...")
    dialogs = await generate_test_dialogs(count_per_type=TEST_SIZE // 3)

    all_results = []
    for i, dialog in enumerate(dialogs):
        print(f"Testing dialog {i + 1}/{len(dialogs)}: {dialog['question'][:50]}...")

        dialog_result = {
            "id": i + 1,
            "question": dialog["question"],
            "history": dialog["history"],
            "expected_type": dialog["expected_type"]
        }

        classification_result = await test_classification(dialog, llm_model)
        dialog_result["classification"] = classification_result["classification"]

        if dialog["expected_type"] == "general":
            news_result = await test_news_answer(dialog, llm_model, news_data, tickets_data)
            dialog_result["answer_test"] = news_result["news_answer"]
            dialog_result["answer_type"] = "news"
        elif dialog["expected_type"] == "museum":
            rag_result = await test_rag_answer(dialog, llm_model, chroma_collection)
            dialog_result["answer_test"] = rag_result["rag_answer"]
            dialog_result["answer_type"] = "rag"
        elif dialog["expected_type"] == "route":
            route_result = await test_route_answer(dialog, llm_model)
            dialog_result["answer_test"] = route_result["route_answer"]
            dialog_result["answer_type"] = "route"

        all_results.append(dialog_result)

        if DETAILED_LOGS:
            log_file = os.path.join(RESULTS_DIR, f"dialog_{i + 1}.json")
            with open(log_file, 'w', encoding='utf-8') as f:
                json.dump(dialog_result, f, ensure_ascii=False, indent=2)

        await asyncio.sleep(1)

    return all_results


def analyze_results(results: List[Dict]) -> Dict:
    total_tests = len(results)

    classification_correct = sum(1 for r in results if r.get("classification", {}).get("is_correct", False))
    classification_accuracy = (classification_correct / total_tests) * 100 if total_tests > 0 else 0

    answer_ratings = [r.get("answer_test", {}).get("rating") for r in results]
    valid_ratings = [r for r in answer_ratings if r is not None]

    answer_stats = {
        "rating_0": sum(1 for r in valid_ratings if r == 0),
        "rating_1": sum(1 for r in valid_ratings if r == 1),
        "rating_2": sum(1 for r in valid_ratings if r == 2),
        "average": sum(valid_ratings) / len(valid_ratings) if valid_ratings else 0
    }

    # Type-specific stats
    type_stats = {}
    for q_type in ["museum", "route", "general"]:
        type_results = [r for r in results if r.get("expected_type") == q_type]
        if not type_results:
            continue

        type_correct = sum(1 for r in type_results if r.get("classification", {}).get("is_correct", False))
        type_accuracy = (type_correct / len(type_results)) * 100 if type_results else 0

        type_ratings = [r.get("answer_test", {}).get("rating") for r in type_results]
        valid_type_ratings = [r for r in type_ratings if r is not None]

        type_stats[q_type] = {
            "count": len(type_results),
            "classification_accuracy": type_accuracy,
            "answer_quality": {
                "rating_0": sum(1 for r in valid_type_ratings if r == 0),
                "rating_1": sum(1 for r in valid_type_ratings if r == 1),
                "rating_2": sum(1 for r in valid_type_ratings if r == 2),
                "average": sum(valid_type_ratings) / len(valid_type_ratings) if valid_type_ratings else 0
            }
        }

    return {
        "total_tests": total_tests,
        "classification": {
            "correct": classification_correct,
            "accuracy": classification_accuracy
        },
        "answer_quality": answer_stats,
        "by_type": type_stats,
        "timestamp": datetime.now().isoformat()
    }


async def main():
    print(f"Starting test run with {TEST_SIZE} dialogs")
    start_time = datetime.now()

    try:
        results = await run_tests()

        stats = analyze_results(results)

        timestamp = start_time.strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(RESULTS_DIR, f"all_results_{timestamp}.json")
        stats_file = os.path.join(RESULTS_DIR, f"stats_{timestamp}.json")

        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

        df = pd.DataFrame(results)
        df['classification_correct'] = df['classification'].apply(lambda x: x.get('is_correct', False))
        df['answer_rating'] = df['answer_test'].apply(lambda x: x.get('rating', None))
        df_simple = df[['id', 'question', 'expected_type', 'answer_type', 'classification_correct', 'answer_rating']]
        csv_file = os.path.join(RESULTS_DIR, f"results_{timestamp}.csv")
        df_simple.to_csv(csv_file, index=False, encoding='utf-8')

        elapsed = (datetime.now() - start_time).total_seconds()
        print("\n" + "=" * 60)
        print("TEST RESULTS SUMMARY")
        print("=" * 60)
        print(f"Total tests: {stats['total_tests']}")
        print(f"Classification accuracy: {stats['classification']['accuracy']:.2f}%")
        print(f"Average answer quality: {stats['answer_quality']['average']:.2f}/2")
        print(f"Answer ratings distribution: 0: {stats['answer_quality']['rating_0']}, " +
              f"1: {stats['answer_quality']['rating_1']}, 2: {stats['answer_quality']['rating_2']}")

        print("\nResults by question type:")
        for q_type, type_stat in stats["by_type"].items():
            print(f"\n{q_type.upper()} ({type_stat['count']} tests):")
            print(f"- Classification accuracy: {type_stat['classification_accuracy']:.2f}%")
            print(f"- Average answer quality: {type_stat['answer_quality']['average']:.2f}/2")
            print(f"- Ratings: 0: {type_stat['answer_quality']['rating_0']}, " +
                  f"1: {type_stat['answer_quality']['rating_1']}, 2: {type_stat['answer_quality']['rating_2']}")

        print("\n" + "=" * 60)
        print(f"Test completed in {elapsed:.2f} seconds")
        print(f"Full results saved to {results_file}")
        print(f"Stats saved to {stats_file}")
        print(f"CSV summary saved to {csv_file}")

    except Exception as e:
        print(f"Error in test execution: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())