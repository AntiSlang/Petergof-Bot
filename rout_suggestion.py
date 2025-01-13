import pymorphy2
import math
from yandex_cloud_ml_sdk import YCloudML

def calculate_distance(coord1, coord2):
    return math.sqrt((float(coord1[0]) - float(coord2[0])) ** 2 + (float(coord1[1]) - float(coord2[1])) ** 2)


def sort_json_by_distance(initial, json_list):

    sorted_json_list = sorted(json_list, key=lambda obj: calculate_distance(
        initial, (float(obj['coordinates']['lat']), float(obj['coordinates']['lon']))
    ))

    return sorted_json_list


def generate_yandex_maps_route_url(coordinates,start_location_coordinates):
    """
    Генерация URL маршрута в Яндекс.Картах с использованием текстовых адресов.
    """

    route_param = '%2С'.join(start_location_coordinates) + '~'
    for coord in coordinates:
        route_param += (coord[0] + '%2C' + coord[1] + '~')


    base_url = "https://yandex.ru/maps/?rtext="
    route_url = f"{base_url}{route_param[:-1]}&rtt=auto"
    return route_url


def lemmatize_text(text):
    morph = pymorphy2.MorphAnalyzer()
    return ' '.join(morph.parse(word)[0].normal_form for word in text.split())


def suggest_route_by_gpt(route_description, link, model_name="yandexgpt", temperature=0.2):

    sdk = YCloudML(
        folder_id="b1g10f66fjjfuqg9ehje",
        auth="AQVN0zMfZzvnaQ_qeJz4mtiu3yYeTKJe2aupo1z5",
    )

    model = sdk.models.completions(model_name)
    model = model.configure(temperature=temperature)

    prompt = f"""
Ты умный бот помощник по музею Петергоф. Тебя попросили подобрать оптимальный маршрут исходя из диалога и лучших объектов по рейтингу, которые сейчас работают
Ты не должен здороваться, так как уже ведешь диалог
Ты подобрал следующие объекты:
{route_description}

Теперь тебе нужно вкратце рассказать про маршрут пользователю и спросить нужно ли что-то изменить. 
Если он хочет посмотреть конкретные объекты, то пусть он напишет и мы добавим их в маршрут (или уберем не интересующие)
Также скажи, что он может изучить маршрут по ссылке ниже:
{link}
    """

    result = model.run(prompt)

    return result.alternatives[0].text

def get_route_suggestion(user_dialogues, data_chunks, initial_coordinates = ["59.891802", "29.913220"], objects_number = 5):

    mentioned_objects = []
    for dialogue in user_dialogues:

        dialog_key ='user'
        if "user" not in dialogue:
            dialog_key = 'bot'

        lemmatized_user_text = lemmatize_text(dialogue[dialog_key].lower())
        for chunk in data_chunks:
            lemmatized_chunk_name = lemmatize_text(chunk["name"].lower())
            if lemmatized_chunk_name in lemmatized_user_text:
                mentioned_objects.append(chunk)

    additional_objects = sorted(data_chunks, key=lambda x: x["score"], reverse=True)
    additional_objects = [obj for obj in additional_objects if obj not in mentioned_objects][:objects_number]

    route = mentioned_objects + additional_objects

    route_sorted = sort_json_by_distance(initial_coordinates, route)

    route_description = ""
    for index, obj in enumerate(route_sorted):
        route_description += f"{index + 1}. {obj['name']} - {obj['description'][:250]}...\n"

    coordinates_list = [[obj["coordinates"]["lat"], obj["coordinates"]["lon"]] for obj in route_sorted]

    link = generate_yandex_maps_route_url(coordinates_list, initial_coordinates)

    route_description = suggest_route_by_gpt(route_description, link)

    return route_description






