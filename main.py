import os
import psycopg2 as sq2
import json
import asyncio
import logging
from aiogram import Bot, Dispatcher, types, F
from aiogram.filters.command import Command
from aiogram.filters.state import State, StatesGroup
from aiogram.fsm.context import FSMContext
from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton
from openai import OpenAI, OpenAIError
import tiktoken
import traceback
from dotenv import load_dotenv

# Загрузка переменных окружения из .env
load_dotenv()

# Получение конфигурации из переменных окружения
SECRET_INVITE_TOKEN = os.getenv("SECRET_INVITE_TOKEN")
ADMIN_IDS = os.getenv("ADMIN_IDS")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
BOT_TOKEN = os.getenv("BOT_TOKEN")
DB_HOST = os.getenv("DB_HOST")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_PORT = os.getenv("DB_PORT")

client = OpenAI(api_key=OPENAI_API_KEY, base_url="https://api.deepseek.com")
max_token_count = 60000

# Инициализация бота
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
system_promt_chat = """
    Ты продуктовый аналитик. Используй информацию ТОЛЬКО от подсистем. При ответе вместо "предоставленных ответов подсистем" говори "моих данных". Если подсистемы вернули "Нет данных для ответа", отвечай: "У меня нет информации по этому вопросу".
    Всегда отвечай только на последний вопрос, не подводи итог по предыдущим:
"""

system_promt_sql = """
    Ты эксперт по SQL. Тебе будут даны ключевые слова и контекст диалога. Сгенерируй posgress SQL-запрос к таблице ai_product_comp.
    Структура таблицы: date, company, summary, key_words.
    Правила:
    - нужно преобразовать в рузультате sql-запроса date в str, например: SELECT TO_CHAR(date, 'YYYY-MM-DD') AS date_str, summary
    - Если в вопросе нет отсылки к ключевым словам, ОБЯЗАТЕЛЬНО генерируй следующий запрос к базе: SELECT * FROM ai_product_comp WHERE false;
    - Если в вопросе просят сравнивать, просто фильтруй по объектам сравнения оператором LIKE. Например: company LIKE '%Альфа%' or company LIKE '%Т-%'
    - При запросе про Тинькофф используй '%Т-%'
    - Не используй: регулярные выражения, операторы '&&', '@>', ARRAY
    - Фильтруй ТОЛЬКО через LIKE по key_words
    - Не используй оператор LIMIT
    - В ответе нужен только sql код
    - Все условия для key_words должны использовать OR и быть в формате: key_words LIKE '%keyword%'
    - СТРОГО ЗАПРЕЩЕНО придумывать новые ключевые слова
    - Используй только список ключевых слов, который тебе дают
"""

system_promt_data = """
    Ты помощник аналитика. Ответь на вопрос пользователя, используя ТОЛЬКО предоставленные данные. 
    Учитывай контекст диалога при формировании ответа.
    Если в данных нет информации для ответа, скажи 'Нет информации'.
"""

system_promt_summary = """
    Ты продуктовый аналитик. Ты получил ответы от нескольких подсистем, которые обрабатывали данные частями. 
    Объедини эти ответы и сформируй итоговый ответ на вопрос пользователя, используя ТОЛЬКО ответы подсистем.
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

async def get_sql_as_dicts(sql_string):
    try:
        connection = create_db_connection()
        cursor = connection.cursor()
        cursor.execute(sql_string)
        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        result = []
        for row in rows:
            result.append(dict(zip(columns, row)))
        connection.close()
        return result
    except Exception as e:
        logger.error(f"Ошибка при выполнении SQL: {str(e)}\nSQL: {sql_string}")
        raise

# Запрос ключевых слов
sq_key = '''
    SELECT DISTINCT TRIM(BOTH FROM keyword) AS unique_keyword
    FROM (
        SELECT unnest(string_to_array(key_words, ', ')) AS keyword
        FROM ai_product_comp
        ) AS keywords_subquery
    WHERE TRIM(BOTH FROM keyword) != ''
    ORDER BY unique_keyword;
'''

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
def format_chat_context(context, max_tokens=2000):
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

# AI-поиск по ключевым словам
async def conn_sql_key(user_message, chat_id, chat_context):
    global key_word
    
    try:
        # 1. Подготавливаем контекст для SQL-генератора
        sql_context = format_chat_context(chat_context)
        
        # 2. Генерация SQL-запроса с учетом контекста
        sql_messages = [
            {"role": "system", "content": system_promt_sql},
            *sql_context,
            {"role": "user", "content": f"""
                Ключевые слова: {key_word}
                Напиши SQL запрос для ответа на текущий вопрос, используя ключевые слова с учетом контекста диалога.
                Текущий вопрос: {user_message}
            """}
        ]
        
        try:
            response = client.chat.completions.create(
                model="deepseek-chat",
                max_tokens=500,
                messages=sql_messages,
                stream=False,
            )
            ai_sql = response.choices[0].message.content
        except OpenAIError as ex:
            logger.error(f"Ошибка генерации SQL: {str(ex)}")
            raise Exception("Ошибка в работе ИИ")
        
        cleaned_sql = ai_sql.replace("```sql", "").replace("```", "").strip()
        logger.info(f"Original SQL: {cleaned_sql}")
        
        # 3. Выполнение SQL и получение данных
        try:
            ai_sql_data = await get_sql_as_dicts(cleaned_sql)
        except Exception as e:
            logger.error(f"Ошибка выполнения SQL: {str(e)}")
            raise Exception("Ошибка обработки данных")
        
        # 4. Разбивка данных на чанки и обработка с учетом контекста
        await bot.send_message(chat_id, 'обрабатываю данные 📊')
        chunks = split_data(ai_sql_data)
        ai_response_list = []
        
        for i, chunk in enumerate(chunks, 1):
            await bot.send_message(chat_id, f'🔍 Анализирую часть {i}/{len(chunks)}...')
            response = await process_data_chunk(chunk, sql_context, user_message)
            ai_response_list.append(response)
        
        # 5. Сборка финального ответа с учетом контекста
        if not ai_response_list:
            responses_str = "Нет данных для ответа."
        else: responses_str = "\n\n".join([f"Часть {i}:\n{r}" for i, r in enumerate(ai_response_list, 1)])
        
        summary_messages = [
            {"role": "system", "content": system_promt_summary},
            *sql_context,
            {"role": "user", "content": f"""
                Контекст диалога: {sql_context}
                Вопрос пользователя: {user_message}
                Ответы от подсистем:
                {responses_str}
                
                Сформируй итоговый ответ, объединив информацию из всех частей.
            """}
        ]
        
        try:
            response = client.chat.completions.create(
                model="deepseek-reasoner",
                max_tokens=1000,
                messages=summary_messages,
                stream=False,
            )
            final_response = response.choices[0].message.content
        except OpenAIError as ex:
            logger.error(f"Ошибка формирования ответа: {str(ex)}")
            raise Exception("Ошибка в работе ИИ")
        
        return final_response
    
    except Exception as e:
        logger.error(f"Ошибка в conn_sql_key: {str(e)}\n{traceback.format_exc()}")
        raise

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

# Основной обработчик сообщений (должен быть последним!)
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
            ai_answer = await conn_sql_key(
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
            await message.answer("Ошибка в обработке данных. Попробуй еще раз")
    
    except Exception as e:
        logger.error(f"Критическая ошибка в обработчике send: {str(e)}\n{traceback.format_exc()}")

# Запуск процесса поллинга
async def main():
    global key_word
    
    try:
        # Инициализация системы доступа
        init_access_table()
        
        # Загрузка ключевых слов
        connection = create_db_connection()
        cursor = connection.cursor()
        cursor.execute(sq_key)
        key_word = ", ".join([row[0] for row in cursor.fetchall()])
        connection.close()
        
        await dp.start_polling(bot)
    except Exception as e:
        logger.error(f"Критическая ошибка при запуске бота: {str(e)}\n{traceback.format_exc()}")

if __name__ == "__main__":
    asyncio.run(main())