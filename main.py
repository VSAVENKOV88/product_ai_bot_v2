import os
import psycopg2 as sq2
import json
import asyncio
import logging
import numpy as np
from aiogram import Bot, Dispatcher, types, F
from aiogram.filters.command import Command
from aiogram.utils.keyboard import InlineKeyboardBuilder
from openai import OpenAI, OpenAIError
import tiktoken
import traceback
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from aiogram.exceptions import TelegramBadRequest
import datetime
import re

# Загрузка переменных окружения из .env
load_dotenv()

# Получение конфигурации
SECRET_INVITE_TOKEN = os.getenv("SECRET_INVITE_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
BOT_TOKEN = os.getenv("BOT_TOKEN")
DB_HOST = os.getenv("DB_HOST")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_PORT = os.getenv("DB_PORT")

# Инициализация моделей
client = OpenAI(api_key=OPENAI_API_KEY, base_url="https://api.deepseek.com")
bge_model = SentenceTransformer('BAAI/bge-m3')
max_token_count = 60000

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("bot_errors.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()

# Словари для хранения состояния пользователей
user_system_prompts = {}  # user_id: prompt_text
chat_contexts = {}        # (chat_id, user_id): message_history

def get_current_date():
    moscow_tz = datetime.timezone(datetime.timedelta(hours=3))
    return datetime.datetime.now(moscow_tz).strftime("%Y-%m-%d")

# Функция загрузки промта из БД по имени
def load_prompt_by_name(prompt_name):
    """Загружает системный промт из базы данных по имени"""
    try:
        conn = create_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT content FROM promt_data WHERE name = %s", (prompt_name,))
        result = cursor.fetchone()
        conn.close()
        
        if result:
            logger.info(f"Промт '{prompt_name}' успешно загружен из БД")
            return result[0]
        else:
            logger.error(f"Промт '{prompt_name}' не найден в базе данных")
            return None
    except Exception as e:
        logger.error(f"Ошибка загрузки промта '{prompt_name}': {str(e)}")
        return None

# Инициализация токенизатора
tokenizer = tiktoken.get_encoding("cl100k_base")

# Работа с базой
def create_db_connection():
    return sq2.connect(
        host=DB_HOST,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        port=DB_PORT
    )

def init_access_table():
    """Создает таблицу для контроля доступа, если она не существует"""
    try:
        conn = create_db_connection()
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS access_control (
                user_id BIGINT PRIMARY KEY,
                username VARCHAR(255),
                full_name VARCHAR(255),
                is_allowed BOOLEAN DEFAULT FALSE
            )
        ''')
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"Ошибка инициализации таблицы доступа: {str(e)}")

def get_user_access(user_id):
    """Проверяет наличие доступа у пользователя"""
    try:
        conn = create_db_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT is_allowed FROM access_control WHERE user_id = %s', (user_id,))
        result = cursor.fetchone()
        conn.close()
        return result[0] if result else False
    except Exception as e:
        logger.error(f"Ошибка получения доступа пользователя {user_id}: {str(e)}")
        return False

def set_user_access(user_id, username, full_name, is_allowed=True):
    """Устанавливает доступ для пользователя"""
    try:
        conn = create_db_connection()
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO access_control (user_id, username, full_name, is_allowed)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (user_id) DO UPDATE SET
                is_allowed = EXCLUDED.is_allowed
        ''', (user_id, username, full_name, is_allowed))
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"Ошибка установки доступа для пользователя {user_id}: {str(e)}")

async def vector_search(question_embedding, limit=24):
    """Выполняет векторный поиск в базе данных"""
    try:
        connection = create_db_connection()
        cursor = connection.cursor()
        
        sql = """
            SELECT TO_CHAR(date, 'YYYY-MM-DD') AS date_str, chunk_text
            FROM article_chunks
            ORDER BY embedding <-> %s::vector
            LIMIT %s
        """
        
        cursor.execute(sql, (question_embedding.tolist(), limit))
        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        result = []
        for row in rows:
            result.append(dict(zip(columns, row)))
        
        connection.close()
        return result
    except Exception as e:
        logger.error(f"Ошибка векторного поиска: {str(e)}")
        return []

# Подсчет токенов
def count_tokens(text):
    return len(tokenizer.encode(text))

# Обработка чанка данных с учетом контекста
async def process_data_chunk(chunk, chat_context, user_message, system_prompt):
    try:
        chunk_json = json.dumps(chunk, ensure_ascii=False)
        
        # Формируем сообщения с учетом контекста диалога
        messages = [{"role": "system", "content": system_prompt}]
        
        # Добавляем историю диалога
        for msg in chat_context:
            messages.append(msg)
        
        # Добавляем текущие данные и вопрос
        messages.append({
            "role": "user", 
            "content": f"Контекст диалога: {chat_context}\n\nВопрос: {user_message}\nДанные: {chunk_json}"
        })
        
        response = client.chat.completions.create(
            model="deepseek-chat",
            max_tokens=1000,
            messages=messages,
            stream=False,
        )
        return response.choices[0].message.content
        
    except OpenAIError as ex:
        logger.error(f"Ошибка OpenAI при обработке чанка: {str(ex)}")
        return "Нет информации"
    except Exception as ex:
        logger.error(f"Ошибка обработки чанка: {str(ex)}")
        return "Нет информации"

# Форматирование контекста для ИИ
def format_chat_context(context, max_tokens=4000):
    try:
        formatted = []
        current_tokens = 0
        
        # Идем от последнего сообщения к первому
        for msg in reversed(context):
            msg_tokens = count_tokens(f"{msg['role']}: {msg['content']}")
            if current_tokens + msg_tokens > max_tokens:
                break
            formatted.insert(0, msg)  # Добавляем в начало для сохранения порядка
            current_tokens += msg_tokens
        
        return formatted
    except Exception as e:
        logger.error(f"Ошибка форматирования контекста: {str(e)}")
        return []

# Генерация эмбеддинга для текста
async def get_text_embedding(text):
    """Асинхронно генерирует векторное представление текста"""
    loop = asyncio.get_running_loop()
    # Выполняем в отдельном потоке, чтобы не блокировать event loop
    embedding = await loop.run_in_executor(None, bge_model.encode, text)
    return embedding / np.linalg.norm(embedding)  # Нормализация для L2-расстояния

# Основная функция обработки запроса
async def handle_user_query(user_message, message, chat_context, user_id):
    try:
        # Получаем текущий промт пользователя
        current_prompt = user_system_prompts.get(user_id)
        if not current_prompt:
            # Если промт не установлен, используем BDM по умолчанию
            current_prompt = load_prompt_by_name('bdm_promt') or ""
            user_system_prompts[user_id] = current_prompt
        
        # Отправляем промежуточные сообщения с reply
        status_msg = await bot.send_message(
            message.chat.id, 
            '🔍 Анализирую вопрос...', 
            reply_to_message_id=message.message_id
        )
        question_embedding = await get_text_embedding(user_message)
        
        await bot.edit_message_text(
            '🔮 Ищу в базе знаний...', 
            chat_id=status_msg.chat.id,
            message_id=status_msg.message_id
        )
        search_results = await vector_search(question_embedding)
        
        if not search_results:
            await bot.edit_message_text(
                "По вашему запросу ничего не найдено.",
                chat_id=status_msg.chat.id,
                message_id=status_msg.message_id
            )
            return "По вашему запросу ничего не найдено."
        
        # 3. Форматирование результатов для ИИ
        formatted_results = []
        for i, result in enumerate(search_results, 1):
            formatted_results.append(
                f"Результат {i}:\n"
                f"Дата: {result['date_str']}\n"
                f"Описание: {result['chunk_text']}\n"
            )
        
        context_str = "\n\n".join(formatted_results)
        
        # 4. Формирование запроса к ИИ
        current_date = get_current_date()
        
        messages = [
            {"role": "system", "content": f"Текущая системная дата: {current_date}\n{current_prompt}"},
            *format_chat_context(chat_context),
            {"role": "user", "content": f"""
                Контекст диалога: {format_chat_context(chat_context)}
                Релевантные данные:
                {context_str}
                
                Вопрос: {user_message}
            """}
        ]
        
        # 5. Запрос к ИИ для формирования ответа
        await bot.edit_message_text(
            '💽 Формирую ответ...', 
            chat_id=status_msg.chat.id,
            message_id=status_msg.message_id
        )
        response = client.chat.completions.create(
            model="deepseek-chat",
            max_tokens=1000,
            messages=messages,
            stream=False,
        )
        
        result = response.choices[0].message.content
        
        # Удаляем статусное сообщение
        try:
            await bot.delete_message(status_msg.chat.id, status_msg.message_id)
        except Exception:
            pass
            
        return result
    
    except Exception as e:
        logger.error(f"Ошибка обработки запроса: {str(e)}\n{traceback.format_exc()}")
        return "Произошла ошибка при обработке запроса. Пожалуйста, попробуйте еще раз."

# Хэндлер на команду /start
@dp.message(Command("start"), F.chat.type == "private")
async def cmd_start(message: types.Message):
    try:
        user_id = message.from_user.id
        args = message.text.split()
        
        # Проверка инвайт-токена
        if len(args) > 1 and args[1] == SECRET_INVITE_TOKEN:
            set_user_access(
                user_id=user_id,
                username=message.from_user.username,
                full_name=message.from_user.full_name,
                is_allowed=True
            )
            # Устанавливаем промт по умолчанию (BDM)
            default_prompt = load_prompt_by_name('bdm_promt') or ""
            user_system_prompts[user_id] = default_prompt
            
            await message.answer('''🤖 Привет! 

🤖 Я твой личный AI-помощник
───────────────────────
Имею доступ к:
📈 Аналитике Альфа Инвестиций (прочитал более 1000 статей)
💼 Каталогу продуктов
⚡ Готовым инструментам

Просто запроси:
• "Справка по SBER"
• "Скрипт: [аналитика] + [продукт]"
• "Стратегия: [профиль риска]"

Данные только из хранилища Альфа ✅
───────────────────────
✨ Экономь время. Укрепляй экспертизу. Бусти продажи!
''')
            # Предлагаем выбрать роль
            await show_role_menu(message.chat.id)
        else:
            # Проверяем есть ли доступ у пользователя
            if get_user_access(user_id):
                # Устанавливаем промт по умолчанию если еще не установлен
                if user_id not in user_system_prompts:
                    default_prompt = load_prompt_by_name('bdm_promt') or ""
                    user_system_prompts[user_id] = default_prompt
                    
                await message.answer('''🤖 С возвращением! 
Вы уже имеете доступ к боту. Можете задавать вопросы.''')
                # Предлагаем выбрать роль
                await show_role_menu(message.chat.id)
            else:
                await message.answer("⛔ Доступ запрещен. Используйте правильную пригласительную ссылку.")
    except Exception as e:
        logger.error(f"Ошибка в обработке /start: {str(e)}\n{traceback.format_exc()}")

# Показ меню выбора роли
async def show_role_menu(chat_id):
    builder = InlineKeyboardBuilder()
    builder.add(types.InlineKeyboardButton(
        text="Аналитик", 
        callback_data="role_expert")
    )
    builder.add(types.InlineKeyboardButton(
        text="ИК", 
        callback_data="role_bdm")
    )
    builder.add(types.InlineKeyboardButton(
        text="Продакт", 
        callback_data="role_product")
    )
    builder.adjust(1)  # По одной кнопке в строке
    await bot.send_message(
        chat_id,
        "👥 Выбери, кто тебе нужен?",
        reply_markup=builder.as_markup()
    )

# Обработчик выбора роли
@dp.callback_query(F.data.startswith("role_"))
async def select_role(callback: types.CallbackQuery):
    user_id = callback.from_user.id
    role_type = callback.data.split("_")[1]
    
    # Соответствие callback-данных именам промтов в БД
    role_mapping = {
        "expert": "expert_promt",
        "bdm": "bdm_promt",
        "product": "product_expert"
    }
    
    if role_type in role_mapping:
        prompt_name = role_mapping[role_type]
        new_prompt = load_prompt_by_name(prompt_name)
        
        if new_prompt:
            user_system_prompts[user_id] = new_prompt
            role_names = {
                "expert": "Аналитик",
                "bdm": "ИК",
                "product": "Продакт"
            }
            await callback.answer(f"Роль изменена на: {role_names[role_type]}")
            await callback.message.edit_text(
                f"✅ Теперь ты общаешься с {role_names[role_type]}!\n"
                "Задавай свой вопрос или используй /role чтобы изменить роль"
            )
        else:
            await callback.answer("Ошибка загрузки роли. Попробуйте позже.")
    else:
        await callback.answer("Неизвестная роль")

# Хэндлер на команду /role
@dp.message(Command("role"), F.chat.type == "private")
async def cmd_role(message: types.Message):
    user_id = message.from_user.id
    if get_user_access(user_id):
        await show_role_menu(message.chat.id)
    else:
        await message.answer("⛔ У вас нет доступа к этой команде")

# Основной обработчик сообщений
@dp.message()
async def send(message: types.Message):
    try:
        # Пропускаем сообщения от ботов
        if message.from_user.is_bot:
            return
            
        # Пропускаем команды
        if message.text and message.text.startswith('/'):
            return
            
        user_id = message.from_user.id
        
        # Логируем полученное сообщение
        logger.info(f"Получено сообщение: chat_id={message.chat.id}, user_id={user_id}, text={message.text}")
        logger.info(f"Тип чата: {message.chat.type}")
        
        # Проверка доступа
        if not get_user_access(user_id):
            logger.info(f"Пользователь {user_id} не имеет доступа")
            # В группе не отвечаем, в личке предупреждаем
            if message.chat.type == "private":
                await message.answer("🔒 Доступ только по приглашению. Используйте пригласительную ссылку для активации.")
            return
            
        # Обработка групповых чатов
        text = message.text or ""
        is_private = message.chat.type == "private"
        bot_username = (await bot.get_me()).username
        mentioned = False
        is_reply = False
        
        if not is_private:
            # Проверяем упоминание бота
            if bot_username:
                # Ищем упоминание в любом месте сообщения
                pattern = re.compile(fr'@?{re.escape(bot_username)}', re.IGNORECASE)
                if pattern.search(text):
                    mentioned = True
                    # Удаляем упоминание из текста
                    text = pattern.sub('', text, count=1).strip()
            
            # Проверяем, является ли сообщение ответом на сообщение бота
            if message.reply_to_message and message.reply_to_message.from_user.id == bot.id:
                is_reply = True
            
            logger.info(f"Упоминание: {mentioned}, Ответ: {is_reply}")
            
            # Если нет ни упоминания, ни ответа, то игнорируем
            if not (mentioned or is_reply):
                logger.info("Сообщение не содержит упоминания или ответа на бота - игнорируем")
                return
        else:
            # В личных сообщениях обрабатываем все
            mentioned = True
        
        # Если после удаления упоминания текст пуст, игнорируем
        if not text:
            logger.info("Текст пуст после удаления упоминания - игнорируем")
            return
            
        logger.info(f"Обрабатываем сообщение: {text}")
        await asyncio.sleep(1)
        
        # Ключ контекста: (chat_id, user_id)
        context_key = (message.chat.id, user_id)
        
        # Инициализация контекста для нового пользователя/чата
        if context_key not in chat_contexts:
            chat_contexts[context_key] = []
        
        # Добавляем текущее сообщение пользователя в контекст
        user_message = {
            "role": "user",
            "content": text
        }
        chat_contexts[context_key].append(user_message)
        
        try:
            # Обработка запроса
            ai_answer = await handle_user_query(
                user_message=text,
                message=message,
                chat_context=chat_contexts[context_key],
                user_id=user_id
            )
            
            # Добавляем ответ ИИ в контекст
            ai_message = {
                "role": "assistant",
                "content": ai_answer
            }
            chat_contexts[context_key].append(ai_message)
            
            # Отправка ответа с reply
            try:
                await message.reply(ai_answer, parse_mode="Markdown")
            except TelegramBadRequest:
                await message.reply(ai_answer)  # Отправка без Markdown
            
        except Exception as ex:
            logger.error(f"Ошибка обработки запроса: {str(ex)}\n{traceback.format_exc()}")
            await message.reply("Ошибка в обработке данных. Попробуйте еще раз")
    
    except Exception as e:
        logger.error(f"Критическая ошибка в обработчике send: {str(e)}\n{traceback.format_exc()}")

# Запуск процесса поллинга
async def main():
    try:
        # Инициализация системы доступа
        init_access_table()
        logger.info("Бот запущен")
        await dp.start_polling(bot)
    except Exception as e:
        logger.error(f"Критическая ошибка при запуске бота: {str(e)}\n{traceback.format_exc()}")

if __name__ == "__main__":
    asyncio.run(main())