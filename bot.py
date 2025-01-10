import asyncio
import json
from pathlib import Path
from aiogram import types
from aiogram import Bot, Dispatcher
from aiogram.contrib.fsm_storage.memory import MemoryStorage
from aiogram.dispatcher import FSMContext
from aiogram.dispatcher.filters.state import StatesGroup, State
from aiogram.types import (InlineKeyboardButton, InlineKeyboardMarkup, KeyboardButton, ReplyKeyboardMarkup,
                           ReplyKeyboardRemove)
from dotenv import load_dotenv
from os import getenv, remove
from yandex_cloud_ml_sdk import AsyncYCloudML
from yandex_cloud_ml_sdk.search_indexes import StaticIndexChunkingStrategy, TextSearchIndexType
from speechkit import model_repository, configure_credentials, creds
from speechkit.stt import AudioProcessingType

load_dotenv()
bot = Bot(token=getenv('TOKEN'))
dp = Dispatcher(bot, storage=MemoryStorage())
bot.user_settings = {}
bot.texts = {}
bot.states = {}
bot.sdk = None
bot.files = None
CIS_COUNTRIES = ['ru', 'ua', 'by', 'kz', 'kg', 'am', 'uz', 'tj', 'az', 'md']
admin_chat = -1002411793280


async def files_delete():
    async for file in bot.sdk.files.list():
        await file.delete()


async def files_create():
    with open('data.json', 'r', encoding='utf-8') as file:
        data = json.load(file)
    docs = [f"{place['context']}\n\nimage_url для {place['title']}: {place['image_url_v2']}" for place in data['places']]
    files = []
    for i, doc in enumerate(docs):
        file_name = f"temp_doc_{i}.txt"
        with open(file_name, "w", encoding="utf-8") as f:
            f.write(doc)
        file = await bot.sdk.files.upload(file_name)
        files.append(file)
        remove(file_name)
    return files


async def get_answer(question: str, user_id: int) -> str:
    operation = await bot.sdk.search_indexes.create_deferred(
        bot.files,
        index_type=TextSearchIndexType(
            chunking_strategy=StaticIndexChunkingStrategy(
                max_chunk_size_tokens=700,
                chunk_overlap_tokens=300,
            )
        )
    )
    search_index = await operation
    tool = bot.sdk.tools.search_index(search_index)
    memory = bot.user_settings[str(user_id)]['memory']
    prompt = \
f'''
1. Контекст и цель:
    - Вы являетесь виртуальным гидом для посетителей музея-заповедника Петергоф.
    - Ваша цель — предоставлять исчерпывающие ответы на вопросы пользователей относительно объектов музея и маршрутов, основываясь на доступной базе данных. Поддерживайте интерес посетителя к посещению музея.

2. Коммуникация с пользователем:
    - Всегда стремитесь ответить до 1000 символов. Я запрещаю отвечать длиной больше 1000 символов.
    - Если вопрос не может быть решён на основе данных, вежливо признайте, что не имеете ответа, но предложите общую информацию о музее.

3. Подача информации:
    - При ответе на вопросы о конкретных объектах, предоставляйте название и завлекательное описание.
    - Если вопрос касается маршрутов, укажите несколько рекомендованных объектов последовательно, формируя маршрут.

4. Мотивация и вдохновение:
    - Используйте вдохновляющий и побуждающий язык, чтобы заинтересовать пользователя в посещении музея.
    - Подчеркните уникальные аспекты и ценность каждого объекта, сделав акцент на незабываемом опыте, который ждёт посетителя.
    - Иногда включайте предложения посетить сайт музея для более полной информации.

5. Ограничения:
    - Не используйте символы форматирования по типу "**" (звёздочки).
    - Длина сообщения до 1000 символов.
    - Отвечайте только на основе имеющейся информации. Если данных недостаточно, честно сообщите об этом, предлагая в качестве альтернативы общие советы по посещению музея.

6. Память:
    - Вот 3 последних запроса пользователя к вам и ваши ответы на них:
        1 (последний): вопрос: {memory["questions"][0]}; ваш ответ: {memory["answers"][0]};
        2 (предпоследний): вопрос: {memory["questions"][1]}; ваш ответ: {memory["answers"][1]};
        3 (предпредпоследний): вопрос: {memory["questions"][2]}; ваш ответ: {memory["answers"][2]};
    \"-\" означает отстутствие запроса. Пользователь может использовать местоимения или говорить в контексте этих сообщений, учитывайте это.

Обязательно соблюдайте эти рекомендации, чтобы предоставить пользователям интересные, информативные и мотивирующие ответы, вдохновляя их на посещение музея Петергоф.
'''
    assistant = await bot.sdk.assistants.create(
        name='rag-assistant',
        model='yandexgpt',
        tools=[tool],
        temperature=0.1,
        instruction=prompt,
        max_prompt_tokens=2000
    )
    thread = await bot.sdk.threads.create()
    try:
        await thread.write(question)
        run = await assistant.run(thread)
        result = await run
        bot.user_settings[str(user_id)]['memory']['questions'] = [question, memory['questions'][0], memory['questions'][1]]
        bot.user_settings[str(user_id)]['memory']['answers'] = [result.text, memory['answers'][0], memory['answers'][1]]
        write_dictionary(bot.user_settings)
        return result.text.replace('**', '')
    finally:
        await search_index.delete()
        await thread.delete()
        await assistant.delete()


async def get_answer_image(question: str) -> str:
    operation = await bot.sdk.search_indexes.create_deferred(
        bot.files,
        index_type=TextSearchIndexType(
            chunking_strategy=StaticIndexChunkingStrategy(
                max_chunk_size_tokens=700,
                chunk_overlap_tokens=300,
            )
        )
    )
    search_index = await operation
    tool = bot.sdk.tools.search_index(search_index)
    prompt = 'Предоставьте параметр image_url (ссылку на изображение), связанный с указанными, или одним из указанных в запросе объектов, если их несколько; или с указанным объектом, если в запросе он один. Отвечать на вопрос пользователя не требуется. В случае, если в запросе не фигурирует ни один объект, извинитесь.'
    assistant = await bot.sdk.assistants.create(
        name='rag-assistant',
        model='yandexgpt',
        tools=[tool],
        temperature=0.1,
        instruction=prompt,
        max_prompt_tokens=2000
    )
    thread = await bot.sdk.threads.create()
    try:
        await thread.write(question)
        run = await assistant.run(thread)
        result = await run
        return result.text
    finally:
        await search_index.delete()
        await thread.delete()
        await assistant.delete()


def load_dictionary(path='users.json'):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_dictionary(dictionary, path='users.json'):
    Path(path).write_text(json.dumps(dictionary, ensure_ascii=False, sort_keys=False, indent=4), encoding='utf-8')


@dp.edited_message_handler(lambda message: message.chat.type == 'private', commands=['help'])
@dp.message_handler(lambda message: message.chat.type == 'private', commands=['help'])
async def help_command(message: types.Message):
    await message.reply(bot.texts[bot.user_settings[str(message.from_user.id)]['language']]['help'])


@dp.edited_message_handler(lambda message: message.chat.type == 'private', commands=['start'])
@dp.message_handler(lambda message: message.chat.type == 'private', commands=['start'])
async def start(message: types.Message):
    if message.from_user.id not in bot.user_settings:
        user_country = message.from_user.language_code if message.from_user.language_code else 'en'
        bot.user_settings[str(message.from_user.id)] = {}
        bot.user_settings[str(message.from_user.id)]['language'] = 'ru' if user_country in CIS_COUNTRIES else 'en'
        bot.user_settings[str(message.from_user.id)]['menu'] = 'off'
        bot.user_settings[str(message.from_user.id)]['memory'] = {'questions': ['-', '-', '-'], 'answers': ['-', '-', '-']}
        write_dictionary(bot.user_settings)
    await message.reply(bot.texts[bot.user_settings[str(message.from_user.id)]['language']]['start'])


@dp.edited_message_handler(lambda message: message.chat.type == 'private', commands=['settings'])
@dp.message_handler(lambda message: message.chat.type == 'private', commands=['settings'])
async def settings(message: types.Message):
    keyboard = get_settings_keyboard(message.from_user.id)
    await message.reply(bot.texts[bot.user_settings[str(message.from_user.id)]['language']]['settings'], reply_markup=keyboard)


class SupportForm(StatesGroup):
    name = State()


def crop(text: str):
    return text if len(text) <= 10 else f'{text[:10]}...'


async def get_reply_keyboard():
    return ReplyKeyboardMarkup(keyboard=[[KeyboardButton('/help'), KeyboardButton('/settings'), KeyboardButton('/support')]])


def get_settings_keyboard(user_id: int):
    button1 = InlineKeyboardButton(bot.texts[bot.user_settings[str(user_id)]['language']]['language'], callback_data='toggle_language')
    button2 = InlineKeyboardButton(bot.texts[bot.user_settings[str(user_id)]['language']]['menu'] + ('✅' if bot.user_settings[str(user_id)]['menu'] == 'on' else '❌'), callback_data="toggle_menu")
    return InlineKeyboardMarkup().add(button1).add(button2)


def get_support_keyboard(user_id: int):
    keyboard = InlineKeyboardMarkup()
    for i in bot.user_settings[str(user_id)]['tickets']:
        keyboard.add(InlineKeyboardButton(crop(i['messages'][0]), callback_data=f'ticket_{i["id"]}'))
    return keyboard.add(InlineKeyboardButton('Новый тикет', callback_data='new_ticket'))


def get_ticket_keyboard(user_id: int, ticket_id: int):
    keyboard = InlineKeyboardMarkup()
    keyboard.add(InlineKeyboardButton('Закрыть тикет', callback_data=f'close_ticket_{ticket_id}'))
    keyboard.add(InlineKeyboardButton('Назад', callback_data=f'back_ticket_{user_id}'))
    return keyboard


@dp.callback_query_handler(lambda call: call.data == "new_ticket")
async def new_ticket(call: types.CallbackQuery):
    await SupportForm.name.set()
    await bot.send_message(call.from_user.id, 'new_ticket')
    bot.states[call.from_user.id] = 'new'


@dp.message_handler(state=SupportForm.name)
async def support_finish(message: types.Message, state: FSMContext):
    if message.text == '/cancel':
        await message.reply(bot.texts[bot.user_settings[str(message.from_user.id)]['language']]['support_cancel'])
        return
    if bot.states[message.from_user.id] == 'new':
        ticket_number = bot.user_settings['ticket']
        bot.user_settings['ticket'] += 1
        bot.user_settings[str(message.from_user.id)]['tickets'].append(ticket_number)
        write_dictionary(bot.user_settings)
        await message.reply(f'Сообщение отправлено, создан новый тикет #{ticket_number}')
        await bot.send_message(admin_chat, f'{message.from_user.id} ({message.message_id}):\n```{message.text}```', parse_mode='Markdown')
    else:
        pass
    await state.finish()


@dp.edited_message_handler(lambda message: message.chat.type == 'private', commands=['support'])
@dp.message_handler(lambda message: message.chat.type == 'private', commands=['support'])
async def support(message: types.Message):
    await message.reply('У вас нет открытых тикетов', reply_markup=get_support_keyboard(message.from_user.id))


@dp.callback_query_handler(lambda call: call.data == "toggle_language")
async def toggle_language(call: types.CallbackQuery):
    user_id = call.from_user.id
    new_language = 'ru' if bot.user_settings[str(user_id)]['language'] == 'en' else 'en'
    bot.user_settings[str(user_id)]['language'] = new_language
    keyboard = get_settings_keyboard(user_id)
    await call.message.edit_text(bot.texts[new_language]['settings'], reply_markup=keyboard)
    write_dictionary(bot.user_settings)


@dp.callback_query_handler(lambda call: call.data == "toggle_menu")
async def toggle_menu(call: types.CallbackQuery):
    user_id = call.from_user.id
    new_menu = 'on' if bot.user_settings[str(user_id)]['menu'] == 'off' else 'off'
    bot.user_settings[str(user_id)]['menu'] = new_menu
    keyboard = get_settings_keyboard(user_id)
    await call.message.edit_text(bot.texts[bot.user_settings[str(user_id)]['language']]['settings'], reply_markup=keyboard)
    msg = await bot.send_message(user_id, 'Меню включено⌨️' if new_menu == 'on' else 'ㅤ', reply_markup=await get_reply_keyboard() if new_menu == 'on' else ReplyKeyboardRemove())
    if new_menu != 'on':
        await msg.delete()
    write_dictionary(bot.user_settings)


@dp.edited_message_handler(lambda message: 'group' in message.chat.type and message.chat.id == admin_chat)
@dp.message_handler(lambda message: 'group' in message.chat.type and message.chat.id == admin_chat)
async def on_message_chat(message: types.Message):
    if message.reply_to_message is None:
        await message.reply('Ответьте на тикет')
    elif message.reply_to_message.from_user.id == bot.id and message.text != 'Ответьте на тикет':
        await bot.send_message(int(message.reply_to_message.text.split(' ')[0]), f'```{message.text}```', reply_to_message_id=int(message.reply_to_message.text.split('(')[1].split(')')[0]), parse_mode='Markdown')
        await message.reply('Сообщение отправлено')


@dp.edited_message_handler(lambda message: message.chat.type == 'private')
@dp.message_handler(lambda message: message.chat.type == 'private')
async def on_message(message: types.Message):
    msg = await message.reply(bot.texts[bot.user_settings[str(message.from_user.id)]['language']]['loading'])
    answer = await get_answer(message.text, message.from_user.id)
    print(answer)
    try:
        answer_img = await get_answer_image(message.text)
        print(answer_img)
        new_answer = answer
        if len(answer) > 1020:
            new_answer = ''
            for line in answer.split('\n'):
                if len(new_answer + f'\n{line}') > 1020:
                    break
                else:
                    new_answer += f'\n{line}'
        await message.reply_photo(photo=answer_img, caption=new_answer)
        await msg.delete()
    except Exception as e:
        print(e)
        await msg.edit_text(answer)


@dp.message_handler(content_types=types.ContentType.VOICE)
async def handle_voice_message(message: types.Message):
    msg = await message.reply(bot.texts[bot.user_settings[str(message.from_user.id)]['language']]['loading'])
    file_info = await bot.get_file(message.voice.file_id)
    file_path = file_info.file_path
    local_file = f"{message.voice.file_id}.ogg"
    await bot.download_file(file_path, local_file)
    text = recognize(local_file)
    remove(local_file)
    await msg.edit_text(f'Ваш вопрос: ```{text}```\n\n{await get_answer(text, message.from_user.id)}', parse_mode='Markdown')


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
    bot.sdk = AsyncYCloudML(folder_id=getenv('FOLDER'), auth=getenv('AUTH'))
    await files_delete()
    bot.files = await files_create()
    await dp.start_polling()


if __name__ == '__main__':
    asyncio.run(main())
