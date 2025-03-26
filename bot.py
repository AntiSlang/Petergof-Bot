import asyncio
import datetime
import json
import linecache
import random
import re
import sys
from pathlib import Path
from traceback import format_exception
import pytz
from aiogram import types
from aiogram import Bot, Dispatcher
from aiogram.contrib.fsm_storage.memory import MemoryStorage
from aiogram.dispatcher import FSMContext
from aiogram.dispatcher.filters.state import StatesGroup, State
from aiogram.types import (InlineKeyboardButton, InlineKeyboardMarkup, KeyboardButton, ReplyKeyboardMarkup,
                           ReplyKeyboardRemove, MediaGroup)
from googletrans import Translator
from rout_suggestion import get_route_suggestion, change_route_by_message
from utils import create_json_chunks
from dotenv import load_dotenv
from os import getenv, remove
from speechkit import model_repository, configure_credentials, creds
from speechkit.stt import AudioProcessingType
from chroma import init_chroma, get_links, create_or_update_chroma_collection
from yandex_cloud_ml_sdk import YCloudML
import requests
from bs4 import BeautifulSoup
from datetime import date
from dialog_pipeline import answer_from_news, is_museum_question, is_greeting_in_message, classify_question_type

load_dotenv()
bot = Bot(token=getenv('TOKEN'))
dp = Dispatcher(bot, storage=MemoryStorage())
bot.user_settings = {}
bot.texts = {}
bot.states = {}
bot.route_data = {}
bot.sdk = None
bot.files = None
CIS_COUNTRIES = ['ru', 'ua', 'by', 'kz', 'kg', 'am', 'uz', 'tj', 'az', 'md']
admin_chat = -1002411793280
COLLECTION_NAME = "peterhof_docs"
bot.chroma_collection = None


'''async def files_delete():
    async for file in bot.sdk.files.list():
        await file.delete()


async def files_create():
    sdk = YCloudML(
        folder_id=getenv('FOLDER'),
        auth=getenv('AUTH'),
    )

    embd = sdk.models.text_embeddings('doc')
    model = sdk.models.completions('yandexgpt')

    path = 'data.json'
    de = DataEnlarger(llm=model, embd=embd, data_path=path)
    docs = de.chunks

    files = []
    for i, doc in enumerate(docs):
        print(len(doc))
        file_name = f'temp_doc_{i}.txt'
        with open(file_name, 'w', encoding='utf-8') as f:
            f.write(doc)
        file = await bot.sdk.files.upload(file_name)
        files.append(file)
        remove(file_name)
    return files


async def files_create():
    with open('data.json', 'r', encoding='utf-8') as file:
        data = json.load(file)
    docs = [f'{place["context"]}\n\nimage_url для {place["title"]}: {place["image_url_v2"]}' for place in data['places']]
    files = []
    for i, doc in enumerate(docs):
        file_name = f'temp_doc_{i}.txt'
        with open(file_name, 'w', encoding='utf-8') as f:
            f.write(doc)
        file = await bot.sdk.files.upload(file_name)
        files.append(file)
        remove(file_name)
    return files'''


def translation(user_id, key):
    return bot.texts[bot.user_settings[str(user_id)]['language']][key]


async def ru_to_en(text):
    async with Translator() as translator:
        return (await translator.translate(text, src='ru', dest='en')).text.replace('Image_url', 'image_url')


async def get_answer_prompt(question, sdk, prompt_original=True, greeting_style="brief"):
    results = bot.chroma_collection.query(
        query_texts=[question],
        n_results=3
    )
    retrieved_docs = results.get('documents', [[]])[0]
    relevant_context = "\n\n".join(retrieved_docs)
    links = get_links(relevant_context)
    llm_model = sdk.models.completions("yandexgpt")

    random_route = '' if random.randint(1,
                                        5) != 1 else 'Также обязательно предложите пользователю составить индивидуальный маршрут и упомяните точную команду: /route'

    greeting_instruction = ""
    if greeting_style == "none":
        greeting_instruction = """
            Важно: не используй никаких приветствий в начале сообщения, если пользователь не поздоровался первым 
            Начинай ответ сразу с информации по существу вопроса.
            Не используй фразы типа "Здравствуйте", "Привет", "Добрый день" и другие формы приветствий, если пользователь не поздоровался первым 
            """
    elif greeting_style == "brief":
        greeting_instruction = """
            не используй приветствий, если пользователь не поздоровался первым 
            """

    if prompt_original:
        prompt = f'''
        Контекст и цель:

        Вы являетесь виртуальным помощником для посетителей музея-заповедника Петергоф.

        Цель:

        Предоставлять исчерпывающие ответы на вопросы пользователей относительно объектов музея, маршрутов, билетов, сайта и других аспектов посещения, основываясь на доступной базе данных. Стремитесь поддерживать интерес посетителя к посещению музея.

        {greeting_instruction}

        Отвечай лаконично и по существу, избегая ненужных длинных введений.

        Релевантный контекст для ответов:

        {relevant_context}

        {random_route}
        '''.strip()
    else:
        prompt = f'''
        Контекст и цель:
        Вы являетесь виртуальным помощником для посетителей музея-заповедника Петергоф.

        {greeting_instruction}

        Отвечай лаконично и по существу.

        Релевантный контекст для ответов:
        {relevant_context}
        '''.strip()

    result = llm_model.run(prompt)
    answer_text = result.alternatives[0].text
    return answer_text, links


async def get_answer(question: str, user_id: int) -> tuple:
    sdk = YCloudML(folder_id=getenv('FOLDER'), auth=getenv('AUTH'))

    memory = bot.user_settings[str(user_id)]['memory']

    user_dialogues = []
    for i in range(min(3, len(memory['questions']))):
        if memory['questions'][i] != '-':
            user_dialogues.append({'user': memory['questions'][i]})
        if memory['answers'][i] != '-':
            user_dialogues.append({'bot': memory['answers'][i]})

    user_dialogues.append({'user': question})
    dialog_history = "\n".join([f"{k}: {v}" for d in user_dialogues for k, v in d.items()])

    is_first_interaction = all(q == '-' for q in memory['questions'])

    user_language = bot.user_settings[str(user_id)]['language']
    user_greeted = is_greeting_in_message(question, user_language)

    if is_first_interaction:
        greeting_style = "very_friendly"
    elif user_greeted:
        greeting_style = "friendly"
    else:
        greeting_style = "friendly"

    links = []
    try:
        question_type = classify_question_type(question, dialog_history)

        if question_type == "museum":
            answer_text, links = await get_answer_prompt(question, sdk, True, greeting_style=greeting_style)
        elif question_type == "route":
            from utils import create_json_chunks
            data_chunks = create_json_chunks()
            coordinates = ['59.891802', '29.913220']

            if bot.route_data.get(user_id) is not None and bot.route_data[user_id]['geo'][0] is not None:
                coordinates = bot.route_data[user_id]['geo']

            answer_text, current_route_json = get_route_suggestion(user_dialogues, data_chunks,
                                                                   initial_coordinates=coordinates)

            if bot.route_data.get(user_id) is None:
                bot.route_data[user_id] = {'geo': coordinates, 'request': question, 'json': current_route_json}
            else:
                bot.route_data[user_id]['request'] = question
                bot.route_data[user_id]['json'] = current_route_json

            answer_text = answer_text.replace('*', '')
        else:
            answer_text = answer_from_news(question, dialog_history, greeting_style=greeting_style)
    except Exception as e:
        answer_text, links = await get_answer_prompt(question, sdk, False, greeting_style=greeting_style)
        print(f"Ошибка в pipeline: {e}, использован запасной ответ")

    bot.user_settings[str(user_id)]['memory']['questions'] = [
        question,
        memory['questions'][0],
        memory['questions'][1]
    ]
    bot.user_settings[str(user_id)]['memory']['answers'] = [
        answer_text,
        memory['answers'][0],
        memory['answers'][1]
    ]
    write_dictionary(bot.user_settings)

    result_text = answer_text.replace('**', '')
    if bot.user_settings[str(user_id)]['language'] == 'en':
        result_text = await ru_to_en(result_text)

    return result_text, links


async def is_news_useful(question: str) -> str:
    sdk = YCloudML(folder_id=getenv('FOLDER'), auth=getenv('AUTH'))
    llm_model = sdk.models.completions("yandexgpt")
    result = llm_model.run(f'Оцени полезность новости для посетителя музея от 1 до 10. В ответ напиши ТОЛЬКО число без лишнего текста. Полезность заключается в том, поможет ли эта новость непосредственному посетителю музея в Санкт-Петербурге, она не должна касаться каких-то людей и других мест/городов. Новость: {question}')
    return result.alternatives[0].text


def load_dictionary(path='users.json'):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_dictionary(dictionary, path='users.json'):
    Path(path).write_text(json.dumps(dictionary, ensure_ascii=False, sort_keys=False, indent=4), encoding='utf-8')


@dp.edited_message_handler(lambda message: message.chat.type == 'private', commands=['help'])
@dp.message_handler(lambda message: message.chat.type == 'private', commands=['help'])
async def help_command(message: types.Message):
    await message.reply(translation(message.from_user.id, 'help'))


def escape_text_except_links(text):
    links = re.findall(r'\[([^\]]+)\]\((https?:\/\/[^\)]+)\)', text)
    placeholder = "LINK_PLACEHOLDER_{}"
    link_map = {}
    for i, (link_text, link_url) in enumerate(links):
        placeholder_key = placeholder.format(i)
        link_map[placeholder_key] = f"[{link_text}]({link_url})"
        text = text.replace(f"[{link_text}]({link_url})", placeholder_key)
    text = re.sub(r'([.\-!()])', r'\\\1', text)
    for placeholder_key, original_link in link_map.items():
        text = text.replace(placeholder_key, original_link)
    return text


async def get_route(user_id: int, request: str = None, latitude: str = None, longitude: str = None):
    data_chunks = create_json_chunks()
    coordinates = ['59.891802' if latitude is None else latitude, '29.913220' if longitude is None else longitude]
    dialogue_user = bot.user_settings[str(user_id)]['memory']['questions'][0]
    dialogue_bot = bot.user_settings[str(user_id)]['memory']['answers'][0]
    user_dialogues = [
        {'user': dialogue_user if dialogue_user != '-' else dialogue_user},
        {'bot': dialogue_bot if dialogue_bot != '-' else dialogue_bot}
    ]
    if request is None:
        res, current_route_json = get_route_suggestion(user_dialogues, data_chunks, initial_coordinates=coordinates)
    else:
        res, current_route_json = change_route_by_message(request, bot.route_data[user_id]['json'], data_chunks, user_dialogues, initial_coordinates=coordinates)
    user_dialogues.append({'bot': res})
    if bot.route_data.get(user_id) is None:
        bot.route_data[user_id] = {'geo': [None, None], 'request': None, 'json': None}
    bot.route_data[user_id]['json'] = current_route_json
    res_final = res.replace('*', '')
    if bot.user_settings[str(user_id)]['language'] == 'en':
        res_translated = await ru_to_en(res_final)
        res_final = ''
        for i in res_translated.split('\n'):
            res_final += f'{i}\n' if 'yandex.ru' not in i else i.replace(' ', '')
    res_final = escape_text_except_links(res_final).replace(r'%2С', r'%2C')
    print(res_final)
    return res_final


@dp.edited_message_handler(lambda message: message.chat.type == 'private', commands=['route'])
@dp.message_handler(lambda message: message.chat.type == 'private', commands=['route'])
async def route(message: types.Message):
    msg = await message.reply(get_route_text(message.from_user.id), disable_web_page_preview=True)
    await msg.edit_text(await get_route(message.from_user.id), parse_mode='MarkdownV2')
    await msg.edit_reply_markup(get_route_keyboard())


async def news_task():
    while True:
        now = datetime.datetime.now(pytz.timezone("Europe/Moscow"))
        target_time = now.replace(hour=9, minute=0, second=0, microsecond=0)
        if now >= target_time:
            target_time += datetime.timedelta(days=1)
        sleep_time = (target_time - now).total_seconds()
        await asyncio.sleep(sleep_time)
        await add_news()


async def add_news():
    data = load_dictionary('news.json')
    b = True
    while b:
        try:
            news = f'https://peterhofmuseum.ru/news/{date.today().year}/{data["number"]}'
            html = requests.get(news).text
            soup = BeautifulSoup(html, "html.parser")
            article = soup.find("article")
            for img in article.find_all("img"):
                img.decompose()
            text = article.get_text(separator="\n", strip=True)
            try:
                usefulness = int(await is_news_useful(text))
            except ValueError:
                usefulness = 5
            print(f'{news}: {usefulness}')
            if usefulness >= 5:
                data['news'] = [text] + ([data['news']][:9] if len(data['news']) == 10 else data['news'])
            data["number"] += 1
        except Exception:
            b = False
    print('method add_news ended')
    write_dictionary(data, 'news.json')


@dp.edited_message_handler(lambda message: message.chat.type == 'private', commands=['start'])
@dp.message_handler(lambda message: message.chat.type == 'private', commands=['start'])
async def start(message: types.Message):
    if str(message.from_user.id) not in bot.user_settings.keys():
        user_country = message.from_user.language_code if message.from_user.language_code else 'en'
        bot.user_settings[str(message.from_user.id)] = {
            'language': 'ru' if user_country in CIS_COUNTRIES else 'en',
            'menu': 'off',
            'memory': {'questions': ['-', '-', '-'], 'answers': ['-', '-', '-']},
            'tickets': {}
        }
        write_dictionary(bot.user_settings)
    await message.reply(translation(message.from_user.id, 'start'))


@dp.edited_message_handler(lambda message: message.chat.type == 'private', commands=['settings'])
@dp.message_handler(lambda message: message.chat.type == 'private', commands=['settings'])
async def settings(message: types.Message):
    keyboard = get_settings_keyboard(message.from_user.id)
    await message.reply(translation(message.from_user.id, 'settings'), reply_markup=keyboard)


class SupportForm(StatesGroup):
    name = State()


class RouteForm(StatesGroup):
    name = State()


class GeoForm(StatesGroup):
    name = State()


def crop(text: str):
    return text if len(text) <= 10 else f'{text[:10]}…'


def get_ticket_answer_keyboard(user_id, ticket_id):
    button1 = InlineKeyboardButton(translation(user_id, 'see'), callback_data=f'ticket_answer_{ticket_id}')
    return InlineKeyboardMarkup().add(button1)


def get_reply_keyboard():
    return ReplyKeyboardMarkup(resize_keyboard=True, is_persistent=True, keyboard=[[KeyboardButton('/help'), KeyboardButton('/settings'), KeyboardButton('/support'), KeyboardButton('/route')]])


def get_settings_keyboard(user_id: int):
    button1 = InlineKeyboardButton(translation(user_id, 'language'), callback_data='toggle_language')
    button2 = InlineKeyboardButton(translation(user_id, 'menu') + ('✅' if bot.user_settings[str(user_id)]['menu'] == 'on' else '❌'), callback_data='toggle_menu')
    return InlineKeyboardMarkup().add(button1).add(button2)


def get_route_keyboard():
    keyboard = InlineKeyboardMarkup()
    keyboard.row(InlineKeyboardButton('♻️ Поменять маршрут', callback_data=f'route_yes'))
    keyboard.row(InlineKeyboardButton('✅ Завершить маршрут', callback_data=f'route_no'))
    keyboard.row(InlineKeyboardButton('📍 Выбрать начальную точку', callback_data=f'route_geo'))
    return keyboard


@dp.message_handler(state=GeoForm.name, content_types=['location'])
async def handle_location(message: types.Message, state: FSMContext):
    msg = await message.reply(translation(message.from_user.id, 'creating_route_geo'), disable_web_page_preview=True)
    latitude, longitude = str(message.location.latitude), str(message.location.longitude)
    if bot.route_data.get(message.from_user.id) is None:
        bot.route_data[message.from_user.id] = {'geo': [latitude, longitude], 'request': None, 'json': None}
    else:
        bot.route_data[message.from_user.id]['geo'] = [latitude, longitude]
    await msg.edit_text(await get_route(message.from_user.id, bot.route_data[message.from_user.id]['request'], bot.route_data[message.from_user.id]['geo'][0], bot.route_data[message.from_user.id]['geo'][1]), parse_mode='MarkdownV2')
    await msg.edit_reply_markup(get_route_keyboard())
    await state.finish()


def get_support_keyboard(user_id: int):
    keyboard = InlineKeyboardMarkup()
    for i, j in bot.user_settings[str(user_id)]['tickets'].items():
        keyboard.add(InlineKeyboardButton(f'#t{i}: {crop(j[0][0])}', callback_data=f'exist_ticket_{i}'))
    if len(bot.user_settings[str(user_id)]['tickets'].keys()) < 10:
        keyboard.add(InlineKeyboardButton(translation(user_id, 'new_ticket'), callback_data='new_ticket'))
    return keyboard


def get_ticket_keyboard(user_id, ticket_id: int):
    keyboard = InlineKeyboardMarkup()
    keyboard.add(InlineKeyboardButton(translation(user_id, 'message'), callback_data=f'message_ticket_{ticket_id}'))
    keyboard.add(InlineKeyboardButton(translation(user_id, 'close_ticket'), callback_data=f'close_ticket_{ticket_id}'))
    keyboard.add(InlineKeyboardButton(translation(user_id, 'back'), callback_data=f'back_ticket'))
    return keyboard


def get_ticket_message_keyboard(user_id, ticket_id: int):
    keyboard = InlineKeyboardMarkup()
    keyboard.add(InlineKeyboardButton(translation(user_id, 'back'), callback_data=f'exist_backticket_{ticket_id}'))
    return keyboard


def get_new_ticket_cancel_keyboard(user_id):
    keyboard = InlineKeyboardMarkup()
    keyboard.add(InlineKeyboardButton(translation(user_id, 'back'), callback_data=f'cancel_ticket'))
    return keyboard


@dp.callback_query_handler(lambda call: call.data == 'new_ticket')
async def new_ticket(call: types.CallbackQuery):
    await SupportForm.name.set()
    await call.message.edit_text(translation(call.from_user.id, 'write_support_new'))
    await call.message.edit_reply_markup(get_new_ticket_cancel_keyboard(call.from_user.id))
    bot.states[call.from_user.id] = 'new'


@dp.callback_query_handler(lambda call: call.data == 'back_ticket')
async def back_ticket(call: types.CallbackQuery):
    tickets_len = len(bot.user_settings[str(call.from_user.id)]['tickets'])
    await call.message.edit_text(get_tickets_text(call.from_user.id, tickets_len))
    await call.message.edit_reply_markup(get_support_keyboard(call.from_user.id))


@dp.callback_query_handler(lambda call: call.data == 'cancel_ticket', state=SupportForm.name)
async def cancel_ticket(call: types.CallbackQuery, state: FSMContext):
    await back_ticket(call)
    await state.finish()


def quote_text(text):
    return f'<blockquote>{text}</blockquote>'


@dp.callback_query_handler(lambda call: 'exist_ticket_' in call.data or 'ticket_answer_' in call.data)
async def exist_ticket(call: types.CallbackQuery):
    ticket_id = int(call.data.split('_')[2])
    message_history = '\n'.join(f'{i[1]}: {quote_text(i[0])}' for i in bot.user_settings[str(call.from_user.id)]['tickets'][str(ticket_id)])
    message_history = message_history if len(message_history) < 4090 else f'…\n{message_history[len(message_history) - 4080:]}'
    if message_history.count('<blockquote>') < message_history.count('</blockquote>'):
        message_history = f'<blockquote>{message_history}'
    await call.message.edit_text(f'{translation(call.from_user.id, "ticket")} #t{ticket_id}\n\n{translation(call.from_user.id, "history")}:\n{message_history}', parse_mode='HTML')
    await call.message.edit_reply_markup(get_ticket_keyboard(call.from_user.id, ticket_id))


@dp.callback_query_handler(lambda call: 'exist_backticket_' in call.data, state=SupportForm.name)
async def exist_backticket(call: types.CallbackQuery, state: FSMContext):
    await exist_ticket(call)
    await state.finish()


def get_tickets_text(user_id, tickets_len):
    return translation(user_id, 'no_open_tickets') if tickets_len == 0 else f'{translation(user_id, "you_have")} {tickets_len} {translation(user_id, "open")}{translation(user_id, "j") if tickets_len % 10 == 1 and tickets_len % 100 != 11 else translation(user_id, "x")} {translation(user_id, "tickets")}{translation(user_id, "ov") if 5 <= tickets_len % 10 <= 9 or tickets_len % 10 == 0 or 11 <= tickets_len % 100 <= 19 else "" if tickets_len % 10 == 1 else translation(user_id, "a")}'


@dp.callback_query_handler(lambda call: 'close_ticket_' in call.data)
async def close_ticket(call: types.CallbackQuery):
    ticket_id = int(call.data.replace('close_ticket_', ''))
    bot.user_settings[str(call.from_user.id)]['tickets'].pop(str(ticket_id))
    write_dictionary(bot.user_settings)
    tickets_len = len(bot.user_settings[str(call.from_user.id)]['tickets'])
    await call.message.edit_text(get_tickets_text(call.from_user.id, tickets_len))
    await call.message.edit_reply_markup(get_support_keyboard(call.from_user.id))
    await bot.send_message(admin_chat, f'Тикет #t{ticket_id} закрыт пользователем')


@dp.callback_query_handler(lambda call: 'message_ticket_' in call.data)
async def message_ticket(call: types.CallbackQuery):
    ticket_id = int(call.data.replace('message_ticket_', ''))
    await SupportForm.name.set()
    await call.message.edit_text(translation(call.from_user.id, 'write_support'))
    await call.message.edit_reply_markup(get_ticket_message_keyboard(call.from_user.id, ticket_id))
    bot.states[call.from_user.id] = f'exist_{ticket_id}'


@dp.message_handler(state=SupportForm.name)
async def support_finish(message: types.Message, state: FSMContext):
    if bot.states[message.from_user.id] == 'new':
        ticket_number = bot.user_settings['ticket']
        bot.user_settings['ticket'] += 1
        bot.user_settings[str(message.from_user.id)]['tickets'][str(ticket_number)] = [[message.text, '👤']]
        await message.reply(f'{translation(message.from_user.id, "message_new_ticket")} #t{ticket_number}')
        text1 = 'Новый тикет'
    else:
        ticket_number = bot.states[message.from_user.id].split('exist_')[1]
        bot.user_settings[str(message.from_user.id)]['tickets'][str(ticket_number)].append([message.text, '👤'])
        await message.reply(f'{translation(message.from_user.id, "message_exist_ticket")} #t{ticket_number}')
        text1 = 'Новое сообщение по тикету'
    await bot.send_message(admin_chat, f'{text1} \\#t{ticket_number} от [{message.from_user.id}](tg://user?id={message.from_user.id}):\n>' + text_v2(message.text).replace('\n', '\n>'), parse_mode='MarkdownV2')
    write_dictionary(bot.user_settings)
    await state.finish()


@dp.edited_message_handler(lambda message: message.chat.type == 'private', commands=['support'])
@dp.message_handler(lambda message: message.chat.type == 'private', commands=['support'])
async def support(message: types.Message):
    tickets_len = len(bot.user_settings[str(message.from_user.id)]['tickets'])
    await message.reply(get_tickets_text(message.from_user.id, tickets_len), reply_markup=get_support_keyboard(message.from_user.id))


@dp.callback_query_handler(lambda call: call.data == 'toggle_language')
async def toggle_language(call: types.CallbackQuery):
    user_id = call.from_user.id
    new_language = 'ru' if bot.user_settings[str(user_id)]['language'] == 'en' else 'en'
    bot.user_settings[str(user_id)]['language'] = new_language
    keyboard = get_settings_keyboard(user_id)
    await call.message.edit_text(bot.texts[new_language]['settings'], reply_markup=keyboard)
    write_dictionary(bot.user_settings)


@dp.callback_query_handler(lambda call: call.data == 'toggle_menu')
async def toggle_menu(call: types.CallbackQuery):
    user_id = call.from_user.id
    new_menu = 'on' if bot.user_settings[str(user_id)]['menu'] == 'off' else 'off'
    bot.user_settings[str(user_id)]['menu'] = new_menu
    keyboard = get_settings_keyboard(user_id)
    await call.message.edit_text(translation(user_id, 'settings'), reply_markup=keyboard)
    msg = await bot.send_message(user_id, translation(user_id, 'menu_on') if new_menu == 'on' else 'ㅤ', reply_markup=get_reply_keyboard() if new_menu == 'on' else ReplyKeyboardRemove())
    if new_menu != 'on':
        await msg.delete()
    write_dictionary(bot.user_settings)


@dp.callback_query_handler(lambda call: 'route_' in call.data)
async def route_inline_handler(call: types.CallbackQuery):
    if call.data == 'route_yes':
        await call.message.reply(translation(call.from_user.id, 'route_suggestion'))
        await RouteForm.name.set()
    if call.data == 'route_geo':
        await call.message.reply(translation(call.from_user.id, 'send_geo'))
        await GeoForm.name.set()
        return
    else:
        await call.message.edit_reply_markup()


def get_route_text(user_id):
    return translation(user_id, 'creating_route_default') if bot.route_data.get(user_id) is None or bot.route_data[user_id]['geo'][0] is None else translation(user_id, 'creating_route_geo')


@dp.message_handler(state=RouteForm.name)
async def route_finish(message: types.Message, state: FSMContext):
    msg = await message.reply(get_route_text(message.from_user.id), disable_web_page_preview=True)
    if bot.route_data.get(message.from_user.id) is None:
        bot.route_data[message.from_user.id] = {'geo': [None, None], 'request': message.text, 'json': None}
    else:
        bot.route_data[message.from_user.id]['request'] = message.text
    await msg.edit_text(await get_route(message.from_user.id, bot.route_data[message.from_user.id]['request'], bot.route_data[message.from_user.id]['geo'][0], bot.route_data[message.from_user.id]['geo'][1]), parse_mode='MarkdownV2')
    await msg.edit_reply_markup(get_route_keyboard())
    await state.finish()


def text_v2(text):
    for i in ['_', '*', '[', ']', '(', ')', '~', '`', '>', '#', '+', '-', '=', '|', '{', '}', '.', '!']:
        text = text.replace(i, f'\\{i}')
    return text



@dp.edited_message_handler(lambda message: 'group' in message.chat.type and message.chat.id == admin_chat)
@dp.message_handler(lambda message: 'group' in message.chat.type and message.chat.id == admin_chat)
async def on_message_chat(message: types.Message):
    # if message.reply_to_message is None:
    #     await message.reply('Ответьте на тикет')
    if message.reply_to_message is not None and message.reply_to_message.from_user.id == bot.id and 'от ' in message.reply_to_message.text:
        ticket_id = message.reply_to_message.text.split('#t')[1].split(' ')[0]
        user_id = message.reply_to_message.text.split('от ')[1].split(':')[0]
        if any([ticket_id == i for i in bot.user_settings[user_id]['tickets'].keys()]):
            await bot.send_message(int(user_id), f'{translation(message.from_user.id, "message_from_support")} #t{ticket_id}', reply_markup=get_ticket_answer_keyboard(message.from_user.id, ticket_id))
            bot.user_settings[user_id]['tickets'][ticket_id].append([message.text, '🛠️'])
            await message.reply('Сообщение отправлено')
            write_dictionary(bot.user_settings)
        else:
            await message.reply('Этот тикет уже закрыт')


def shorten_text(text, length=1020):
    new_text = text
    if len(text) > length:
        new_text = ''
        for line in text.split('\n'):
            if len(new_text + f'\n{line}') > length:
                break
            else:
                new_text += f'\n{line}'
    return new_text


async def print_exception(e: Exception):
    exc_type, exc_obj, tb = sys.exc_info()
    f = tb.tb_frame
    lineno = tb.tb_lineno
    filename = f.f_code.co_filename
    linecache.checkcache(filename)
    line = linecache.getline(filename, lineno, f.f_globals)
    info, error = format_exception(exc_type, e, tb)[-2:]
    error = error.replace("\n", "")
    tts = f"Exception on line {lineno} ({error}) (строка {line})"
    print(tts)


@dp.edited_message_handler(lambda message: message.chat.type == 'private')
@dp.message_handler(lambda message: message.chat.type == 'private')
async def on_message(message: types.Message):
    msg = await message.reply(translation(message.from_user.id, 'loading'))
    try:
        answer, links = await get_answer(message.text, message.from_user.id)

        dialog_history = "\n".join([f"user: {q}" if q != '-' else "" for q in
                                    bot.user_settings[str(message.from_user.id)]['memory']['questions']])
        is_route = False
        try:
            from dialog_pipeline import classify_question_type
            question_type = classify_question_type(message.text, dialog_history)
            is_route = (question_type == "route")
        except Exception as e:
            print(f"Ошибка классификации: {e}")

        reply_markup = None
        if is_route:
            reply_markup = get_route_keyboard()
            answer = escape_text_except_links(answer)
        else:
            answer_split = answer.split('Оценка объекта')
            answer = answer_split[0].strip()
            if len(answer_split) > 1 and '/route' in answer_split[1]:
                answer += '\n' * 2 + [i for i in answer_split[1].split('\n') if '/route' in i][0]
    except Exception as e:
        await print_exception(e)
        await msg.edit_text(translation(message.from_user.id, 'unexpected_error'))
        return

    if len(links) == 0:
        try:
            if is_route:
                await msg.edit_text(shorten_text(answer, 4080), reply_markup=reply_markup)
            else:
                await msg.edit_text(shorten_text(answer, 4080), reply_markup=reply_markup)
        except Exception as e:
            print(f"Ошибка при отправке сообщения: {e}")
            await msg.edit_text(shorten_text(answer.replace('\\', ''), 4080))
        return

    try:
        answer_shorten = shorten_text(answer)
        if len(links) == 1:
            try:
                await message.reply_photo(photo=links[0], caption=answer_shorten, reply_markup=reply_markup)
            except Exception as e:
                print(f"Ошибка при отправке фото: {e}")
                await message.reply_photo(photo=links[0], caption=answer_shorten.replace('\\', ''))
        elif len(links) > 1:
            media_group = MediaGroup()
            for i, link in enumerate(links):
                if i == 0:
                    media_group.attach_photo(photo=link, caption=answer_shorten)
                else:
                    media_group.attach_photo(photo=link)
            await message.reply_media_group(media=media_group)
            if is_route:
                await message.reply("Выберите действие с маршрутом:", reply_markup=reply_markup)
        await msg.delete()
        return
    except Exception as e:
        print(f"Ошибка при отправке медиа: {e}")

    try:
        await msg.edit_text(shorten_text(answer.replace('\\', ''), 4080), reply_markup=reply_markup)
    except Exception as e:
        print(f"Критическая ошибка в обработке сообщения: {e}")
        await msg.edit_text(translation(message.from_user.id, 'unexpected_error'))


@dp.message_handler(content_types=types.ContentType.VOICE)
async def handle_voice_message(message: types.Message):
    msg = await message.reply(translation(message.from_user.id, 'loading'))
    file_info = await bot.get_file(message.voice.file_id)
    file_path = file_info.file_path
    local_file = f'{message.voice.file_id}.ogg'
    await bot.download_file(file_path, local_file)
    text = recognize(local_file)
    text = text if text is not None and text != '' and len(text) >= 2 else '-'
    remove(local_file)
    await msg.edit_text(f'Ваш вопрос: {quote_text(text)}\n\n{(await get_answer(text, message.from_user.id))[0].strip()}', parse_mode='HTML')


@dp.message_handler(content_types=[types.ContentType.ANY])
async def handle_any_message(message: types.Message):
    await message.reply(translation(message.from_user.id, 'unsupported_message'))


def recognize(audio):
    model = model_repository.recognition_model()
    model.model = 'general'
    model.language = 'ru-RU'
    model.audio_processing_type = AudioProcessingType.Full
    result = model.transcribe_file(audio)
    return result[0].normalized_text


async def main():
    bot.texts = load_dictionary('texts.json')
    bot.user_settings = load_dictionary('users.json')
    configure_credentials(
        yandex_credentials=creds.YandexCredentials(
            api_key=getenv('AUTH')
        )
    )
    bot.chroma_collection, client = init_chroma(remote=True)
    print(bot.chroma_collection.count())
    # create_or_update_chroma_collection(bot.chroma_collection)
    asyncio.create_task(news_task())
    await dp.start_polling()


if __name__ == '__main__':
    asyncio.run(main())