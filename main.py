import os
import psycopg2 as sq2
import json
import asyncio
import logging
import numpy as np
from aiogram import Bot, Dispatcher, types, F
from aiogram.filters.command import Command
from aiogram.filters.state import State, StatesGroup
from aiogram.fsm.context import FSMContext
from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton
from openai import OpenAI, OpenAIError
import tiktoken
import traceback
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# Загрузка переменных окружения из .env
load_dotenv()

# Получение конфигурации
SECRET_INVITE_TOKEN = os.getenv("SECRET_INVITE_TOKEN")
ADMIN_IDS = [int(id) for id in os.getenv("ADMIN_IDS").split(",")] if os.getenv("ADMIN_IDS") else []
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

# Состояния для администратора
class AdminState(StatesGroup):
    ADD_ACCESS = State()
    REMOVE_ACCESS = State()
    ADD_ADMIN = State()
    REMOVE_ADMIN = State()

# Системные промты
system_promt_data = """
    Ты помощник аналитика. Ответь на вопрос пользователя, используя ТОЛЬКО предоставленные данные. 
    Учитывай контекст диалога при формировании ответа.
    Если в данных нет информации для ответа, скажи 'Нет информации'.
"""

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

async def vector_search(question_embedding, limit=15):
    """Выполняет векторный поиск в базе данных"""
    try:
        connection = create_db_connection()
        cursor = connection.cursor()
        
        # Фиксированный запрос векторного поиска
        sql = """
            SELECT TO_CHAR(date, 'YYYY-MM-DD') AS date_str, company, summary
            FROM ai_product_comp
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

# Функции управления доступом
def init_access_table():
    try:
        conn = create_db_connection()
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS access_control (
                user_id BIGINT PRIMARY KEY,
                username VARCHAR(255),
                full_name VARCHAR(255),
                is_allowed BOOLEAN DEFAULT FALSE,
                is_admin BOOLEAN DEFAULT FALSE
            )
        ''')
        # Добавляем администраторов
        for admin_id in ADMIN_IDS:
            cursor.execute('''
                INSERT INTO access_control (user_id, is_allowed, is_admin)
                VALUES (%s, TRUE, TRUE)
                ON CONFLICT (user_id) DO UPDATE SET is_admin = EXCLUDED.is_admin
            ''', (admin_id,))
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"Ошибка инициализации таблицы доступа: {str(e)}")

def get_user_access(user_id):
    try:
        conn = create_db_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT is_allowed, is_admin FROM access_control WHERE user_id = %s', (user_id,))
        result = cursor.fetchone()
        conn.close()
        return result or (False, False)
    except Exception as e:
        logger.error(f"Ошибка получения доступа пользователя {user_id}: {str(e)}")
        return False, False

def set_user_access(user_id, username, full_name, is_allowed=True, is_admin=False):
    try:
        conn = create_db_connection()
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO access_control (user_id, username, full_name, is_allowed, is_admin)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (user_id) DO UPDATE SET
                is_allowed = EXCLUDED.is_allowed,
                is_admin = EXCLUDED.is_admin
        ''', (user_id, username, full_name, is_allowed, is_admin))
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"Ошибка установки доступа для пользователя {user_id}: {str(e)}")

def get_all_users():
    try:
        conn = create_db_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT user_id, username, full_name, is_allowed, is_admin FROM access_control')
        users = cursor.fetchall()
        conn.close()
        return users
    except Exception as e:
        logger.error(f"Ошибка получения списка пользователей: {str(e)}")
        return []

# Подсчет токенов
def count_tokens(text):
    return len(tokenizer.encode(text))

# Разделение данных на чанки
def split_data(data, max_tokens=45000):
    try:
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        for item in data:
            item_tokens = count_tokens(json.dumps(item, ensure_ascii=False))
            
            if current_tokens + item_tokens > max_tokens:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = [item]
                current_tokens = item_tokens
            else:
                current_chunk.append(item)
                current_tokens += item_tokens
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    except Exception as e:
        logger.error(f"Ошибка разделения данных: {str(e)}")
        return [data]

# Обработка чанка данных с учетом контекста
async def process_data_chunk(chunk, chat_context, user_message):
    try:
        chunk_json = json.dumps(chunk, ensure_ascii=False)
        
        # Формируем сообщения с учетом контекста диалога
        messages = [{"role": "system", "content": system_promt_data}]
        
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
async def handle_user_query(user_message, chat_id, chat_context):
    try:
        # 1. Генерация эмбеддинга для вопроса
        await bot.send_message(chat_id, '🔍 Анализирую вопрос...')
        question_embedding = await get_text_embedding(user_message)
        
        # 2. Выполнение векторного поиска
        await bot.send_message(chat_id, '🔎 Ищу в базе знаний...')
        search_results = await vector_search(question_embedding)
        
        if not search_results:
            return "По вашему запросу ничего не найдено."
        
        # 3. Форматирование результатов для ИИ
        formatted_results = []
        for i, result in enumerate(search_results, 1):
            formatted_results.append(
                f"Результат {i}:\n"
                f"Дата: {result['date_str']}\n"
                f"Компания: {result['company']}\n"
                f"Описание: {result['summary']}\n"
            )
        
        context_str = "\n\n".join(formatted_results)
        
        # 4. Формирование запроса к ИИ
        messages = [
            {"role": "system", "content": system_promt_data},
            *format_chat_context(chat_context),
            {"role": "user", "content": f"""
                Контекст диалога: {format_chat_context(chat_context)}
                Релевантные данные:
                {context_str}
                
                Вопрос: {user_message}
            """}
        ]
        
        # 5. Запрос к ИИ для формирования ответа
        await bot.send_message(chat_id, '🧠 Формирую ответ...')
        response = client.chat.completions.create(
            model="deepseek-reasoner",
            max_tokens=1500,
            messages=messages,
            stream=False,
        )
        
        return response.choices[0].message.content
    
    except Exception as e:
        logger.error(f"Ошибка обработки запроса: {str(e)}\n{traceback.format_exc()}")
        return "Произошла ошибка при обработке запроса. Пожалуйста, попробуйте еще раз."

# Хэндлер на команду /start
@dp.message(Command("start"))
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
            await message.answer('''🤖 Привет! 
Я твой личный AI-эксперт по анализу продуктов и сервисов конкурентов.

💡 Задавай вопросы как человеку, например:
• "Какие продукты выпустил Т-Банк за последний месяц"
• "Что нового сделал Сбер"

''')
        else:
            # Проверяем есть ли доступ у пользователя
            is_allowed, _ = get_user_access(user_id)
            if is_allowed:
                await message.answer('''🤖 С возвращением! 
Вы уже имеете доступ к боту. Можете задавать вопросы.''')
            else:
                await message.answer("⛔ Доступ запрещен. Используйте правильную пригласительную ссылку.")
    except Exception as e:
        logger.error(f"Ошибка в обработке /start: {str(e)}\n{traceback.format_exc()}")

# Административные команды
@dp.message(Command("admin"))
async def admin_panel(message: types.Message, state: FSMContext):
    try:
        user_id = message.from_user.id
        _, is_admin = get_user_access(user_id)
        
        if not is_admin:
            await message.answer("⛔ У вас нет прав администратора")
            return
        
        keyboard = InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton(text="📊 Список пользователей", callback_data="user_list")],
            [InlineKeyboardButton(text="➕ Добавить доступ", callback_data="add_access"),
             InlineKeyboardButton(text="➖ Удалить доступ", callback_data="remove_access")],
            [InlineKeyboardButton(text="👑 Добавить админа", callback_data="add_admin"),
             InlineKeyboardButton(text="👑 Удалить админа", callback_data="remove_admin")]
        ])
        
        await message.answer("⚙️ Панель администратора:", reply_markup=keyboard)
        await state.clear()
    except Exception as e:
        logger.error(f"Ошибка в админ-панели: {str(e)}\n{traceback.format_exc()}")

@dp.callback_query(F.data == "user_list")
async def list_users(callback: types.CallbackQuery, state: FSMContext):
    try:
        users = get_all_users()
        if not users:
            await callback.message.answer("ℹ️ Нет пользователей в базе")
            return
        
        response = "👥 Список пользователей:\n\n"
        for user in users:
            user_id, username, full_name, is_allowed, is_admin = user
            status = "✅ Доступен" if is_allowed else "⛔ Заблокирован"
            role = "👑 Админ" if is_admin else "👤 Пользователь"
            response += f"{full_name} (@{username})\nID: {user_id}\n{status} | {role}\n\n"
        
        await callback.message.edit_text(response)
    except Exception as e:
        logger.error(f"Ошибка в списке пользователей: {str(e)}\n{traceback.format_exc()}")
        await callback.answer("Ошибка при получении списка пользователей", show_alert=True)

@dp.callback_query(F.data == "add_access")
async def add_access(callback: types.CallbackQuery, state: FSMContext):
    try:
        await callback.message.edit_text(
            text="👤 Введите ID пользователя для предоставления доступа:",
            reply_markup=InlineKeyboardMarkup(inline_keyboard=[
                [InlineKeyboardButton(text="🔙 Назад", callback_data="admin_back")]
            ])
        )
        await state.set_state(AdminState.ADD_ACCESS)
    except Exception as e:
        logger.error(f"Ошибка в add_access: {str(e)}\n{traceback.format_exc()}")
        await callback.answer("Ошибка при обработке", show_alert=True)

@dp.callback_query(F.data == "remove_access")
async def remove_access(callback: types.CallbackQuery, state: FSMContext):
    try:
        await callback.message.edit_text(
            text="👤 Введите ID пользователя для отзыва доступа:",
            reply_markup=InlineKeyboardMarkup(inline_keyboard=[
                [InlineKeyboardButton(text="🔙 Назад", callback_data="admin_back")]
            ])
        )
        await state.set_state(AdminState.REMOVE_ACCESS)
    except Exception as e:
        logger.error(f"Ошибка в remove_access: {str(e)}\n{traceback.format_exc()}")
        await callback.answer("Ошибка при обработке", show_alert=True)

@dp.callback_query(F.data == "add_admin")
async def add_admin(callback: types.CallbackQuery, state: FSMContext):
    try:
        await callback.message.edit_text(
            text="👑 Введите ID пользователя для назначения администратором:",
            reply_markup=InlineKeyboardMarkup(inline_keyboard=[
                [InlineKeyboardButton(text="🔙 Назад", callback_data="admin_back")]
            ])
        )
        await state.set_state(AdminState.ADD_ADMIN)
    except Exception as e:
        logger.error(f"Ошибка в add_admin: {str(e)}\n{traceback.format_exc()}")
        await callback.answer("Ошибка при обработке", show_alert=True)

@dp.callback_query(F.data == "remove_admin")
async def remove_admin(callback: types.CallbackQuery, state: FSMContext):
    try:
        await callback.message.edit_text(
            text="👑 Введите ID администратора для разжалования:",
            reply_markup=InlineKeyboardMarkup(inline_keyboard=[
                [InlineKeyboardButton(text="🔙 Назад", callback_data="admin_back")]
            ])
        )
        await state.set_state(AdminState.REMOVE_ADMIN)
    except Exception as e:
        logger.error(f"Ошибка в remove_admin: {str(e)}\n{traceback.format_exc()}")
        await callback.answer("Ошибка при обработке", show_alert=True)

# Обработка ввода ID для административных действий
@dp.message(AdminState.ADD_ACCESS)
async def handle_add_access(message: types.Message, state: FSMContext):
    try:
        user_id = int(message.text)
        set_user_access(
            user_id=user_id,
            username="manual_add",
            full_name="Manual Add",
            is_allowed=True
        )
        await message.answer(f"✅ Пользователю {user_id} предоставлен доступ")
        await state.clear()
    except ValueError:
        await message.answer("❌ Неверный формат ID. Введите числовой ID пользователя.")
    except Exception as e:
        logger.error(f"Ошибка в handle_add_access: {str(e)}\n{traceback.format_exc()}")
        await message.answer("❌ Ошибка при добавлении доступа")

@dp.message(AdminState.REMOVE_ACCESS)
async def handle_remove_access(message: types.Message, state: FSMContext):
    try:
        user_id = int(message.text)
        set_user_access(
            user_id=user_id,
            username="manual_add",
            full_name="Manual Add",
            is_allowed=False
        )
        await message.answer(f"⛔ Пользователю {user_id} отозван доступ")
        await state.clear()
    except ValueError:
        await message.answer("❌ Неверный формат ID. Введите числовой ID пользователя.")
    except Exception as e:
        logger.error(f"Ошибка в handle_remove_access: {str(e)}\n{traceback.format_exc()}")
        await message.answer("❌ Ошибка при отзыве доступа")

@dp.message(AdminState.ADD_ADMIN)
async def handle_add_admin(message: types.Message, state: FSMContext):
    try:
        user_id = int(message.text)
        set_user_access(
            user_id=user_id,
            username="admin_add",
            full_name="Admin Add",
            is_allowed=True,
            is_admin=True
        )
        await message.answer(f"👑 Пользователь {user_id} назначен администратором")
        await state.clear()
    except ValueError:
        await message.answer("❌ Неверный формат ID. Введите числовой ID пользователя.")
    except Exception as e:
        logger.error(f"Ошибка в handle_add_admin: {str(e)}\n{traceback.format_exc()}")
        await message.answer("❌ Ошибка при назначении администратора")

@dp.message(AdminState.REMOVE_ADMIN)
async def handle_remove_admin(message: types.Message, state: FSMContext):
    try:
        user_id = int(message.text)
        set_user_access(
            user_id=user_id,
            username="admin_remove",
            full_name="Admin Remove",
            is_admin=False
        )
        await message.answer(f"👤 Пользователь {user_id} больше не администратор")
        await state.clear()
    except ValueError:
        await message.answer("❌ Неверный формат ID. Введите числовой ID пользователя.")
    except Exception as e:
        logger.error(f"Ошибка в handle_remove_admin: {str(e)}\n{traceback.format_exc()}")
        await message.answer("❌ Ошибка при снятии прав администратора")

@dp.callback_query(F.data == "admin_back")
async def admin_back(callback: types.CallbackQuery, state: FSMContext):
    try:
        await state.clear()
        await admin_panel(callback.message, state)
    except Exception as e:
        logger.error(f"Ошибка в admin_back: {str(e)}\n{traceback.format_exc()}")
        await callback.answer("Ошибка при возврате в меню", show_alert=True)

# Хранилище контекста по чатам
chat_contexts = {}

# Основной обработчик сообщений
@dp.message()
async def send(message: types.Message):
    try:
        # Пропускаем команды
        if message.text and message.text.startswith('/'):
            return
            
        user_id = message.from_user.id
        
        # Проверка доступа
        is_allowed, _ = get_user_access(user_id)
        if not is_allowed:
            await message.answer("🔒 Доступ только по приглашению. Используйте пригласительную ссылку для активации.")
            return
            
        await asyncio.sleep(1)
        await message.answer('готовлю ответ ⏳')
        
        chat_id = message.chat.id
        
        # Инициализация контекста для нового чата
        if chat_id not in chat_contexts:
            chat_contexts[chat_id] = []
        
        # Добавляем текущее сообщение пользователя в контекст
        user_message = {
            "role": "user",
            "content": message.text
        }
        chat_contexts[chat_id].append(user_message)
        
        try:
            # Обработка запроса
            ai_answer = await handle_user_query(
                user_message=message.text,
                chat_id=chat_id,
                chat_context=chat_contexts[chat_id]
            )
            
            # Добавляем ответ ИИ в контекст
            ai_message = {
                "role": "assistant",
                "content": ai_answer
            }
            chat_contexts[chat_id].append(ai_message)
            
            await message.answer(ai_answer, parse_mode="markdown")
        except Exception as ex:
            logger.error(f"Ошибка обработки запроса: {str(ex)}\n{traceback.format_exc()}")
            await message.answer("Ошибка в обработке данных. Попробуйте еще раз")
    
    except Exception as e:
        logger.error(f"Критическая ошибка в обработчике send: {str(e)}\n{traceback.format_exc()}")

# Запуск процесса поллинга
async def main():
    try:
        # Инициализация системы доступа
        init_access_table()
        logger.info("Бот запущен и готов к работе")
        await dp.start_polling(bot)
    except Exception as e:
        logger.error(f"Критическая ошибка при запуске бота: {str(e)}\n{traceback.format_exc()}")

if __name__ == "__main__":
    asyncio.run(main())