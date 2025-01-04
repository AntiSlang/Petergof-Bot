import json


def create_chunks(file_path):

    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    docs = []

    for place in data['places']:
        all_info_text = f"""
        Название: 
        {place['title']}

        Информация об объекта: 
        {place['context']}

        Оценка объекта (по 100 бальной шкале):
        {place['rate']}

        Ссылка на объект:
        {place['full_url']}
        """

        docs.append(all_info_text)

    return docs

