import pymorphy2
import math
import re
import nltk
from nltk.stem.snowball import RussianStemmer
from yandex_cloud_ml_sdk import YCloudML
from chromadb.api import EmbeddingFunction
from os import getenv, remove
from chroma import create_or_update_chroma_collection, init_chroma
from dotenv import load_dotenv
from typing import List

load_dotenv()

YANDEX_FOLDER_ID = getenv('FOLDER')
YANDEX_AUTH = getenv('AUTH')
def update_route(user_message, route_json, relevant_objects, model):

    objects_names = ", ".join([obj["name"] for obj in route_json])

    # print("mentioned objects:")
    # print(objects_names)

    prompt = f"""Ты умный гид музея Петергоф, и твоя задача — понять, какие объекты пользователь хочет увидеть, а какие нет.

Ты получишь список объектов, которые были предложены пользователю, и его последнее сообщение, в котором он хочет что-то поменять или добавить. Также тебе будет предложено пару дополнительных объектов с их описаниями, чтобы заменить какие-то из списка (если надо) или просто добавить к имеющимся.

ЗАПОМНИ: ты можешь говорить только про объекты из сообщения.

Желание пользователя может включать в себя:

1) Удалить объект из маршрута. Если пользователю не нравятся определённые объекты, это может быть явно выражено или ясно из контекста. В этом случае нужно удалить такие объекты из списка и вернуть его в том же формате, в котором они были даны.

2) Добавить объект в список. Если пользователь изъявляет желание увидеть что-то ещё, такие как "Добавь музей карт", добавь указанные объекты к текущему маршруту и не удаляй существующие. Важно: не переходи к заменам и не удаляй текущие объекты, если об этом нет явного указания.

3) Удалить объект и добавить ещё что-то в маршрут. Если же пользователь говорит о нежелательных объектах и высказывает желание добавить другие, удали упомянутые объекты и добавь указанные новые, сохраняя оставшиеся.

Примеры работы:
- Если пользователь пишет "Добавь музей карт", текущий список должен остаться прежним, просто с добавлением "музей карт".
- При отсутствии конкретного указания на удаление, ничто не должно удаляться.

Определи, какой из трех случаев наиболее вероятен, следуй инструкциям по нему и выведи только одну строку с итоговыми названиями объектов.
- Важно избегать любых отсылок к другим темам или ресурсам, кроме органов управления объектами.

ВАЖНО! Понять из сообщения пользователя, какие объекты нужно удалить, а какие добавлять, чтобы его маршрут стал более релевантным.
ВАЖНО! Если пользователь не попросил уменьшить количество мест, то обязательно выведи 5 мест, если по контексту не ясно, что пользователь не хочет их видеть
ВАЖНО! Добавляй ответов всегда побольше чтобы пользователь мог выбрать сам (если только он не указал, что ему нужно меньше объектов)

Выведи только итоговый список мест для посещения. Если подходящих мест нет, предложи самые подходящие альтернативы

## Возможные релевантные объекты для пользователя: ##
{relevant_objects}

## Последнее сообщение пользователя: ##
{user_message}

## Объекты нашего музея, которые были рекомендованы к посещению ##
{objects_names}

Обязательно ответь списком, как я тебе дал на входе
"""

    result = model.run(prompt)
    return result.alternatives[0].text


def calculate_distance(coord1, coord2):
    lat1, lon1 = map(float, coord1)
    lat2, lon2 = map(float, coord2)
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    R = 6371000
    return R * c

def normilize_text(text):
    text = text.replace('ё', 'е').replace('Ё', 'е')
    text = re.sub(r'[^A-Za-zА-Яа-я]', '', text)
    return stem_text(text.lower())

def stem_text(text):
    stemmer = RussianStemmer()
    return ' '.join(stemmer.stem(word) for word in text.split())

def sort_json_by_distance(initial, json_list):
    return sorted(
        json_list,
        key=lambda obj: calculate_distance(
            initial, (float(obj['coordinates']['lat']), float(obj['coordinates']['lon']))
        )
    )

def filter_route_by_distance(route, max_distance=5000):
    if not route:
        return []
    filtered_route = [route[0]]
    for obj in route[1:]:
        obj_coords = (float(obj['coordinates']['lat']), float(obj['coordinates']['lon']))
        if any(calculate_distance(obj_coords, (float(existing['coordinates']['lat']), float(existing['coordinates']['lon']))) <= max_distance
               for existing in filtered_route):
            filtered_route.append(obj)
    return filtered_route

def generate_yandex_maps_route_url(coordinates, start_location_coordinates):

    route_param = '%2C'.join(start_location_coordinates) + '~'
    for coord in coordinates:
        route_param += (coord[0] + '%2C' + coord[1] + '~')
    base_url = "(https://yandex.ru/maps/?rtext="
    route_url = f"{base_url}{route_param[:-1]}&rtt=auto)"
    print(route_url)
    return f"[Ссылка Яндекс Карты]{route_url}"

def lemmatize_text(text):
    morph = pymorphy2.MorphAnalyzer()
    return ' '.join(morph.parse(word)[0].normal_form for word in text.split())

def suggest_route_by_gpt(model, route_description, link, last_message=None):
    prompt = f"""
Ты умный бот помощник по музею Петергоф. Тебя попросили подобрать оптимальный маршрут исходя из диалога и лучших объектов по рейтингу.
Ты подобрал следующие объекты:
{route_description}

Теперь очень кратко расскажи про маршрут и спроси, нужно ли что-то изменить (по 1 предложению на каждый объект).
Также сообщи, что маршрут можно изучить по ссылке:
{link}
"""
    if last_message is not None:
        prompt += f"Ранее ты уже предлагал маршрут но пользователь решил его поменять. Вот его по: {last_message}"
    result = model.run(prompt)
    return result.alternatives[0].text

def change_route_by_gpt(model, route_description, link, last_message, user_dialog):
    prompt = f"""
Ты умный бот помощник по музею Петергоф. Тебя попросили подобрать оптимальный маршрут исходя из диалога и лучших объектов по рейтингу.
Ранее ты уже предлагал маршрут, но пользователь захотел что-то поменять. 
Вот диалог с пользователем:
{user_dialog}

Вот последнее сообщение пользователя: {last_message}
Теперь ты изменил маршрут на следующий:
{route_description}

Теперь кратко расскажи пользователю, что ты поменял в его маршруте и о самом новом маршруте. Спроси нужно ли что-то изменить
Также сообщи, что маршрут можно изучить по ссылке:
{link}
"""

    result = model.run(prompt)
    return result.alternatives[0].text


def check_on_positivity(model, user_dialogues, mentioned_objects):
    dialogue_str = "\n".join([d.get("user", d.get("bot", "")) for d in user_dialogues])
    objects_str = ", ".join(mentioned_objects)
    prompt = f"""Ты умный гид музея Петергоф. Проанализируй диалог и объекты, про которые говорилось:
Диалог:
{dialogue_str}
Объекты: {objects_str}
Выведи объекты через запятую, убрав те, которые пользователь не хочет видеть.
"""
    result = model.run(prompt)
    return result.alternatives[0].text.split(", ")

def cluster_by_distance(objects, max_distance=5000):
    clusters = []
    for obj in objects:
        added = False
        for cluster in clusters:
            if any(calculate_distance(
                    (float(obj['coordinates']['lat']), float(obj['coordinates']['lon'])),
                    (float(existing['coordinates']['lat']), float(existing['coordinates']['lon']))
            ) <= max_distance for existing in cluster):
                cluster.append(obj)
                added = True
                break
        if not added:
            clusters.append([obj])
    return clusters

def select_cluster(clusters, min_objects):
    suitable = [cluster for cluster in clusters if len(cluster) >= min_objects]
    if suitable:
        return max(suitable, key=lambda c: len(c))
    return max(clusters, key=lambda c: len(c))


def get_route_suggestion(user_dialogues, data_chunks, initial_coordinates=["59.891802", "29.913220"],
                         objects_number=5, initial_max_distance=1000, max_distance_limit=10000, distance_increment=500):
    sdk = YCloudML(
        folder_id=YANDEX_FOLDER_ID,
        auth=YANDEX_AUTH,
    )
    model = sdk.models.completions(model_name="llama")
    model = model.configure(temperature=0.1)

    mentioned_objects = []
    all_objects_names = []
    for dialogue in user_dialogues:
        dialog_key = 'user' if "user" in dialogue else 'bot'
        lemmatized_text = normilize_text(dialogue[dialog_key])
        for chunk in data_chunks:
            lemmatized_name = normilize_text(chunk["name"])
            if lemmatized_name in lemmatized_text:
                all_objects_names.append(lemmatized_name)
                mentioned_objects.append(chunk)

    good_objects = check_on_positivity(model, user_dialogues, all_objects_names)
    mentioned_objects = list(filter(lambda x: normilize_text(x["name"]) in good_objects, mentioned_objects))
    bad_objects = set(all_objects_names) - set(good_objects)

    additional_objects = sorted(data_chunks, key=lambda x: x["score"], reverse=True)
    additional_objects = list(filter(lambda x: normilize_text(x["name"]) not in bad_objects, additional_objects))
    additional_objects = [obj for obj in additional_objects if obj not in mentioned_objects][:objects_number]

    candidate_route = mentioned_objects + additional_objects
    candidate_route = sort_json_by_distance(initial_coordinates, candidate_route)

    current_threshold = initial_max_distance
    clusters = cluster_by_distance(candidate_route, max_distance=current_threshold)
    selected_cluster = select_cluster(clusters, objects_number)

    selected_cluster = sort_json_by_distance(initial_coordinates, selected_cluster)

    route_description = ""
    for index, obj in enumerate(selected_cluster):
        route_description += f"{index+1}. {obj['name']} - {obj['description'][:250]}...\n"
    coordinates_list = [[obj["coordinates"]["lat"], obj["coordinates"]["lon"]] for obj in selected_cluster]
    link = generate_yandex_maps_route_url(coordinates_list, initial_coordinates)
    route_description = suggest_route_by_gpt(model, route_description, link)

    return route_description, selected_cluster


def select_new_routes_by_gpt(model, retrieved_context, current_old_names, user_message, user_dialog):

    prompt = f"""
## Основная инструкция:
Ты умный гид музея Петергоф. Твоя цель – формировать оптимальный маршрут, учитывая, что в новом маршруте уже присутствуют следующие объекты: {", ".join(current_old_names)}.
На основе истории диалога и последнего сообщения пользователя определи, нужно ли добавлять дополнительные объекты.
Если пользователь просит уменьшить общее количество объектов маршрута или если текущих объектов уже достаточно для удовлетворения запроса, то не добавляй новые объекты и выведи пустую строку.
Если же пользователь запрашивает дополнение маршрута новыми интересными объектами, выбери из списка новых кандидатов объекты для добавления (выведи только их названия через запятую).

## История диалога:
{user_dialog}

## Последнее сообщение пользователя:
{user_message}

## Новые кандидаты для маршрута:
{retrieved_context}

Учти, что если запрос пользователя подразумевает сокращение маршрута или текущие объекты уже отвечают его запросу, выводи пустую строку.
ВАЖНО: Внимательно следи за нужным количество новых объектов и учти, что уже есть {len(current_old_names)} объектов, так что тебе нужно добавить на {len(current_old_names)} меньше, чем в итоге хочет пользователь 
ВАЖНО: Выводи только названия объектов через запятую или пустую строку (''), других рассуждений или пояснений не надо
ВАЖНО: Максимальное количество объектов может быть 10 (только если пользователь не попросил больше 10 объектов)
"""
    result = model.run(prompt)
    return result.alternatives[0].text

def filter_unwanted_objects(message, current_route_json, model):
    objects_names = ", ".join([obj["name"] for obj in current_route_json])

    prompt = f"""
Ты выступаешь в роли опытного гида музея Петергоф.
Твоя задача — проанализировать последнее сообщение пользователя и текущий список объектов маршрута, чтобы определить, какие из них остаются актуальными для маршрута.

Пользователь написал: "{message}"
Текущий маршрут содержит следующие объекты: {objects_names}

Требования:
1. Выведи через запятую только те названия объектов, которые должны остаться в маршруте.
2. Убирай объекты только если в сообщении пользователя или в истории диалога явно указано, что их следует исключить.
3. По последнему сообщению пойми насколько пользователю нравится текущие объекты и удали те, чтобы можно было улучшить маршрут
4. Если пользователь задал конкретное число объектов для маршрута, соблюдай это требование: если требуется сократить количество — оставь ровно столько, если увеличить — выведи все текущие объекты.
5. Ответ должен содержать только список названий через запятую, без дополнительного пояснения.
"""
    result = model.run(prompt)
    return result.alternatives[0].text



def change_route_by_message(message, current_route_json, data_chunks, user_dialog, initial_coordinates=["59.891802", "29.913220"],
                            objects_number=20, initial_max_distance=500, max_distance_limit=1000, distance_increment=100):
    sdk = YCloudML(
        folder_id=YANDEX_FOLDER_ID,
        auth=YANDEX_AUTH,
    )
    model = sdk.models.completions(model_name="llama")
    model = model.configure(temperature=0.5)

    chroma_collection = init_chroma()
    collection_count = chroma_collection.count()
    if collection_count == 0:
        print("Creating a new collection")
        create_or_update_chroma_collection(chroma_collection)

    retriever_results = chroma_collection.query(
        query_texts=[message],
        n_results=20
    )
    retrieved_docs = retriever_results.get('documents', [[]])[0]
    # Обрезаем каждый документ до 400 символов, чтобы уменьшить объём контекста
    retrieved_docs = [doc[:400] for doc in retrieved_docs]
    retrieved_context = "\n\n".join(retrieved_docs)
    # print('Current obj----------------')
    # for data in current_route_json:
    #     print(data['name'])
    filtered_old_names = filter_unwanted_objects(message, current_route_json, model)
    # print("Good places (старые объекты):")
    # print(filtered_old_names)
    norm_old_names = normilize_text(filtered_old_names)
    old_objects = []
    for chunk in data_chunks:
        lemmatized_chunk_name = normilize_text(chunk["name"])
        if lemmatized_chunk_name in norm_old_names:
            old_objects.append(chunk)
    current_old_names = [obj["name"] for obj in old_objects]
    # print("Старые объекты, которые останутся в маршруте:")
    # print(current_old_names)

    new_objects_response = select_new_routes_by_gpt(model, retrieved_context, current_old_names, message, user_dialog)
    # print("Ответ на выбор новых объектов:")
    # print(new_objects_response)
    norm_new_names = normilize_text(new_objects_response)
    new_objects = []
    for chunk in data_chunks:
        lemmatized_chunk_name = normilize_text(chunk["name"])
        # Добавляем только новые объекты, которых ещё нет среди старых
        if (lemmatized_chunk_name in norm_new_names and
                lemmatized_chunk_name not in normilize_text(" ".join(current_old_names))):
            new_objects.append(chunk)

    num_needed = max(0, objects_number - len(old_objects))
    filtered_new_objects = []
    if num_needed > 0 and new_objects:
        current_threshold = initial_max_distance
        clusters = cluster_by_distance(new_objects, max_distance=current_threshold)
        selected_cluster_new = select_cluster(clusters, num_needed)
        while len(selected_cluster_new) < num_needed and current_threshold < max_distance_limit:
            current_threshold += distance_increment
            clusters = cluster_by_distance(new_objects, max_distance=current_threshold)
            selected_cluster_new = select_cluster(clusters, num_needed)
        filtered_new_objects = sort_json_by_distance(initial_coordinates, selected_cluster_new)
    else:
        filtered_new_objects = []

    final_route = old_objects + filtered_new_objects

    route_description = ""
    for index, obj in enumerate(final_route):
        route_description += f"{index + 1}. {obj['name']} - {obj['description'][:150]}...\n"
    coordinates_list = [[obj["coordinates"]["lat"], obj["coordinates"]["lon"]] for obj in final_route]
    link = generate_yandex_maps_route_url(coordinates_list, initial_coordinates)
    route_description = change_route_by_gpt(model, route_description, link, message, user_dialog)

    return route_description, final_route
