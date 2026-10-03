"""RAG-агент: перед ответом ищет информацию в векторной базе Pinecone."""

import hashlib
import logging
import os
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone

# Названия моделей — это не секреты, поэтому их можно держать в коде
CHAT_MODEL = "openai:gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-small"  # размерность вектора 1536

# Настройки нарезки текста на чанки (в символах)
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Настройки загрузки страниц
REQUEST_TIMEOUT = 15  # секунд, чтобы программа не зависла навсегда
# Википедия отвечает 403 на запросы с «поддельным браузером» или без контактов.
# Её правила требуют формат: ИмяБота/версия (контакт) библиотека/версия
USER_AGENT = (
    "RAGCatsBot/1.0 (educational project; https://github.com/arinalogic) "
    f"python-requests/{requests.__version__}"
)

# Открытое API со случайными фактами о кошках: ключ не нужен
CAT_FACT_API_URL = "https://catfact.ninja/fact"

# Теги, в которых нет основного текста: скрипты, стили, меню, подвал и т.п.
NOISE_TAGS = ["script", "style", "noscript", "nav", "header", "footer",
              "aside", "form", "iframe", "svg", "sup", "table"]

# Классы блоков Википедии, где нет полезного текста статьи:
# кнопки «править», навигационные плашки, сноски, оглавление, карточки
NOISE_CLASSES = ["mw-editsection", "navbox", "reference", "reflist", "hatnote",
                 "toc", "mw-references-wrap", "infobox"]

# Разделы в конце статьи: всё, что после них, — не основной текст
STOP_SECTIONS = ["примечания", "литература", "ссылки", "см. также"]

# Теги, из которых собираем текст: заголовки, абзацы и пункты списков
TEXT_TAGS = ["h2", "h3", "p", "li"]

# Чанки короче этого числа символов почти не несут смысла — не сохраняем
MIN_CHUNK_LENGTH = 100

SYSTEM_PROMPT = """Ты полезный ассистент.

## Инструменты

- `search_knowledge_base`: ищет информацию в базе знаний. Всегда вызывай его
  перед ответом на вопрос пользователя.
- `add_url_to_knowledge_base`: читает веб-страницу по ссылке и сохраняет её
  текст в базу знаний. Вызывай его, когда пользователь просит прочитать,
  изучить или запомнить страницу по ссылке.
- `call_api`: получает из интернета (из внешнего API) случайный интересный
  факт о кошках. Вызывай его, когда пользователь просит интересный факт
  о кошках из интернета или из API. Для таких просьб базу знаний искать
  не нужно. Факт приходит на английском — переведи его на русский.

Отвечай на основе найденных фрагментов. Если в базе ничего не нашлось,
честно скажи об этом."""

# Логгер этого модуля: через него пишем сообщения об этапах работы
logger = logging.getLogger(__name__)

# Настройки вывода в терминал
LOG_FILE = "rag_agent.log"
LINE_WIDTH = 50
PREVIEW_LENGTH = 200


def print_tool_call(name: str) -> None:
    """Печатает в консоль, какой инструмент вызвал агент."""
    print(f"🔧 Агент вызвал инструмент: {name}")


def get_env(name: str) -> str:
    """Берёт переменную из окружения. Если её нет — сразу понятная ошибка."""
    value = os.getenv(name)
    if not value:
        raise ValueError(f"Переменная {name} не задана в файле .env")
    return value


def is_valid_url(url: str) -> bool:
    """Проверяет, что ссылка похожа на настоящий адрес сайта: http(s)://домен."""
    parts = urlparse(url)
    return parts.scheme in ("http", "https") and bool(parts.netloc)


def download_page(url: str) -> bytes:
    """Скачивает страницу и возвращает её HTML в виде байтов."""
    headers = {"User-Agent": USER_AGENT}
    try:
        response = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
        # Если сервер ответил ошибкой (404, 500 и т.д.) — тут будет исключение
        response.raise_for_status()
    except requests.exceptions.Timeout as error:
        raise RuntimeError(
            f"Страница не ответила за {REQUEST_TIMEOUT} секунд: {url}"
        ) from error
    except requests.exceptions.ConnectionError as error:
        raise RuntimeError(
            f"Не удалось подключиться к сайту. Проверьте интернет и ссылку: {url}"
        ) from error
    except requests.exceptions.HTTPError as error:
        raise RuntimeError(
            f"Сайт вернул ошибку {response.status_code}: {url}"
        ) from error
    except requests.exceptions.RequestException as error:
        raise RuntimeError(f"Не удалось загрузить страницу: {error}") from error

    # Отдаём байты, а не текст: BeautifulSoup сам определит кодировку
    return response.content


def clean_spaces(text: str) -> str:
    """Заменяет любые пробелы, табы и переносы подряд на один пробел."""
    return " ".join(text.split())


def remove_noise_elements(soup: BeautifulSoup) -> None:
    """Удаляет из HTML теги и блоки, где нет основного текста статьи."""
    # Теги: скрипты, меню, номера сносок (sup), таблицы и т.п.
    for tag in soup(NOISE_TAGS):
        tag.decompose()
    # Меню бывают и в обычных div, но с пометкой role="navigation"
    for tag in soup.select('[role="navigation"]'):
        tag.decompose()
    # Блоки Википедии по классам: ".navbox, .reflist, ..."
    selector = ", ".join(f".{name}" for name in NOISE_CLASSES)
    for tag in soup.select(selector):
        tag.decompose()


def find_content_area(soup: BeautifulSoup):
    """Находит часть страницы, где лежит сам текст статьи."""
    # В Википедии текст статьи лежит в div.mw-parser-output.
    # Для других сайтов пробуем <main>, потом <article>, потом <body>
    return (
        soup.select_one("#mw-content-text .mw-parser-output")
        or soup.find("main")
        or soup.find("article")
        or soup.body
        or soup
    )


def is_stop_heading(text: str) -> bool:
    """Проверяет, что заголовок — начало служебного раздела в конце статьи."""
    return text.lower() in STOP_SECTIONS


def extract_main_text(html: bytes) -> str:
    """Убирает из HTML служебные блоки и возвращает чистый текст статьи.

    Абзацы в результате разделены пустой строкой.
    """
    soup = BeautifulSoup(html, "html.parser")

    # 1. Удаляем мусор ДО извлечения текста, иначе он попадёт в абзацы
    remove_noise_elements(soup)

    # 2. Ищем основную часть страницы
    content = find_content_area(soup)

    # 3. Идём по заголовкам и абзацам в том порядке, как они стоят на странице
    paragraphs: list[str] = []
    for element in content.find_all(TEXT_TAGS):
        # Абзац внутри пункта списка (или вложенный пункт) уже вошёл
        # в текст «родителя» — иначе текст задвоится
        if element.find_parent(["p", "li"]):
            continue

        text = clean_spaces(element.get_text(separator=" "))
        if not text:
            continue

        # Дошли до «Примечаний», «Литературы» и т.п. — дальше всё отбрасываем
        if element.name in ("h2", "h3") and is_stop_heading(text):
            break

        paragraphs.append(text)

    # 4. Разделяем абзацы пустой строкой: сплиттер режет текст
    #    в первую очередь именно по "\n\n", и абзацы не рвутся посередине
    return "\n\n".join(paragraphs)


@tool
def call_api() -> str:
    """Получает из интернета (из внешнего API) случайный интересный факт о кошках.

    Используй, когда пользователь просит интересный факт о кошках
    из интернета или из API. Факт возвращается на английском языке.
    """
    print_tool_call("call_api")
    logger.info("API-инструмент вызван")
    headers = {"User-Agent": USER_AGENT}

    # Агенту нельзя «падать»: любую ошибку возвращаем ему понятным текстом
    try:
        response = requests.get(
            CAT_FACT_API_URL, headers=headers, timeout=REQUEST_TIMEOUT
        )
        # Если сервер ответил не 200 (404, 500 и т.д.) — тут будет исключение
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.Timeout:
        message = f"API не ответило за {REQUEST_TIMEOUT} секунд."
        logger.error(message)
        return message
    except requests.exceptions.ConnectionError:
        message = "Не удалось подключиться к API. Проверьте интернет."
        logger.error(message)
        return message
    except requests.exceptions.HTTPError:
        message = f"API вернуло ошибку {response.status_code}."
        logger.error(message)
        return message
    except requests.exceptions.RequestException as error:
        message = f"Не удалось выполнить запрос к API: {error}"
        logger.error(message)
        return message
    except ValueError:
        # Сюда попадём, если сервер прислал не JSON
        message = "API вернуло ответ не в формате JSON."
        logger.error(message)
        return message

    # В ответе может не оказаться поля "fact" — проверяем, а не надеемся
    fact = data.get("fact") if isinstance(data, dict) else None
    if not fact:
        message = "В ответе API нет поля 'fact'."
        logger.error(message)
        return message

    logger.info("Ответ получен: %s", fact)
    return fact


class RAGAgent:
    """Агент, который ищет ответы в векторной базе Pinecone."""

    def __init__(self) -> None:
        # 1. Читаем все ключи и настройки из файла .env
        load_dotenv()
        pinecone_api_key = get_env("PINECONE_API_KEY")
        index_name = get_env("PINECONE_INDEX_NAME")
        openai_api_key = get_env("OPENAI_API_KEY")
        openai_base_url = get_env("OPENAI_BASE_URL")

        # 2. Подключаемся к Pinecone и проверяем, что индекс существует
        pinecone_client = Pinecone(api_key=pinecone_api_key)
        if not pinecone_client.has_index(index_name):
            raise ValueError(f"Индекс '{index_name}' не найден в Pinecone")
        self.index = pinecone_client.Index(index_name)

        # 3. Эмбеддинги превращают текст в вектор из чисел.
        #    Запросы идут через ProxyAPI, поэтому указываем base_url
        self.embeddings = OpenAIEmbeddings(
            model=EMBEDDING_MODEL,
            api_key=openai_api_key,
            base_url=openai_base_url,
        )

        # 4. Векторное хранилище связывает индекс Pinecone и эмбеддинги
        self.vector_store = PineconeVectorStore(
            index=self.index,
            embedding=self.embeddings,
        )

        # 5. Модель чата, тоже через ProxyAPI
        self.model = init_chat_model(
            CHAT_MODEL,
            api_key=openai_api_key,
            base_url=openai_base_url,
            temperature=0.5,
            timeout=60,
        )

        # 6. Сплиттер режет длинный текст на куски с перекрытием,
        #    чтобы мысль на границе двух чанков не потерялась
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )

        # 7. Создаём агента и даём ему инструменты
        self.agent = create_agent(
            model=self.model,
            tools=[
                self._make_search_tool(),
                self._make_add_url_tool(),
                call_api,
            ],
            system_prompt=SYSTEM_PROMPT,
        )

    def search(self, question: str, k: int = 3) -> list[Document]:
        """Ищет в Pinecone k фрагментов, самых близких по смыслу к вопросу."""
        try:
            return self.vector_store.similarity_search(question, k=k)
        except Exception as error:
            raise RuntimeError(f"Ошибка поиска в Pinecone: {error}") from error

    def _make_search_tool(self):
        """Оборачивает метод search в инструмент, который понятен агенту."""

        @tool
        def search_knowledge_base(question: str) -> str:
            """Ищет информацию в базе знаний по вопросу пользователя."""
            print_tool_call("search_knowledge_base")
            documents = self.search(question)
            if not documents:
                return "В базе знаний ничего не найдено."
            # Склеиваем найденные фрагменты в один текст для модели
            return "\n\n".join(doc.page_content for doc in documents)

        return search_knowledge_base

    def delete_old_chunks(self, url_hash: str) -> int:
        """Удаляет из Pinecone все чанки одной ссылки и возвращает их число.

        У чанков ссылки id вида "<хеш>-0", "<хеш>-1", ...
        Serverless-индекс не умеет удалять по фильтру метаданных,
        зато умеет искать id по началу строки (prefix). Им и пользуемся.
        """
        # Дефис в конце нужен, чтобы префикс точно совпал только с этим хешем
        prefix = f"{url_hash}-"
        deleted_count = 0
        try:
            # index.list отдаёт id порциями (списками), а не все сразу
            for id_batch in self.index.list(prefix=prefix):
                if id_batch:
                    self.index.delete(ids=id_batch)
                    deleted_count += len(id_batch)
        except Exception as error:
            raise RuntimeError(
                f"Не удалось удалить старые чанки из Pinecone: {error}"
            ) from error
        return deleted_count

    def add_url(self, url: str) -> int:
        """Читает страницу по ссылке, режет на чанки и сохраняет в Pinecone.

        Возвращает число сохранённых чанков.
        При проблемах выбрасывает ValueError или RuntimeError с текстом на русском.
        """
        url = url.strip()

        # 1. Проверяем ссылку до того, как идти в интернет
        if not is_valid_url(url):
            raise ValueError(
                f"Неверная ссылка: '{url}'. Ссылка должна начинаться с http:// или https://"
            )

        # 2. Скачиваем страницу
        html = download_page(url)
        logger.info("Страница загружена: %s", url)

        # 3. Оставляем только основной текст
        text = extract_main_text(html)
        logger.info("Получено %d символов текста", len(text))
        if not text:
            raise ValueError(f"На странице не найден текст: {url}")

        # 4. Режем текст на чанки. В метаданные каждого чанка кладём ссылку,
        #    чтобы потом было видно, откуда взят фрагмент
        documents = self.text_splitter.create_documents(
            [text], metadatas=[{"source": url}]
        )
        logger.info("Получено %d чанков", len(documents))

        # 5. Короткие обрывки почти не несут смысла и только мешают поиску
        documents = [
            doc for doc in documents
            if len(doc.page_content) >= MIN_CHUNK_LENGTH
        ]
        logger.info(
            "Чанков после отбора (от %d символов): %d",
            MIN_CHUNK_LENGTH, len(documents),
        )
        if not documents:
            raise ValueError(f"На странице нет фрагментов достаточной длины: {url}")

        # 6. Делаем постоянные id: одна и та же ссылка всегда даёт те же id
        url_hash = hashlib.md5(url.encode("utf-8")).hexdigest()
        ids = [f"{url_hash}-{number}" for number in range(len(documents))]

        # 7. Удаляем старые чанки этой ссылки. Перезаписи по id мало:
        #    если раньше чанков было 50, а теперь 30, то 30–49 остались бы
        deleted_count = self.delete_old_chunks(url_hash)
        logger.info("Удалено старых чанков: %d", deleted_count)

        # 8. Сохраняем: vector_store сам превратит каждый чанк в эмбеддинг
        try:
            self.vector_store.add_documents(documents, ids=ids)
        except Exception as error:
            raise RuntimeError(f"Не удалось сохранить в Pinecone: {error}") from error
        logger.info("Сохранено в Pinecone: %d чанков", len(documents))

        return len(documents)

    def _make_add_url_tool(self):
        """Оборачивает метод add_url в инструмент для агента."""

        @tool
        def add_url_to_knowledge_base(url: str) -> str:
            """Читает веб-страницу по ссылке и сохраняет её текст в базу знаний."""
            print_tool_call("add_url_to_knowledge_base")
            # Агенту нельзя «падать»: любую ошибку возвращаем ему текстом
            try:
                chunk_count = self.add_url(url)
            except (ValueError, RuntimeError) as error:
                return f"Не удалось добавить страницу: {error}"
            return f"Страница сохранена в базу знаний. Чанков: {chunk_count}"

        return add_url_to_knowledge_base

    def ask(self, question: str) -> str:
        """Задаёт вопрос агенту и возвращает текст его ответа."""
        result = self.agent.invoke(
            {"messages": [{"role": "user", "content": question}]}
        )
        return result["messages"][-1].content

    def check_connection(self) -> int:
        """Спрашивает у Pinecone статистику индекса: сколько в нём векторов."""
        stats = self.index.describe_index_stats()
        return stats.total_vector_count


def setup_logging() -> None:
    """Пишет логи уровня INFO только в файл, чтобы не засорять консоль."""
    logging.basicConfig(
        filename=LOG_FILE,
        encoding="utf-8",
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    # Библиотеки httpx и httpx2 (её использует openai) пишут лог
    # о каждом HTTP-запросе — скрываем этот шум
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpx2").setLevel(logging.WARNING)


def print_header(title: str) -> None:
    """Печатает главный заголовок с двойной линией под ним."""
    print(title)
    print("=" * LINE_WIDTH)


def print_section(title: str) -> None:
    """Печатает заголовок раздела с пустой строкой перед ним."""
    print(f"\n{title}")


def print_separator() -> None:
    """Печатает линию между разделами."""
    print("\n" + "-" * LINE_WIDTH)


def print_error(message: str) -> None:
    """Печатает ошибку в консоль, а подробности пишет в лог."""
    print(f"❌ {message}")
    logger.exception(message)


def make_preview(text: str) -> str:
    """Делает из фрагмента одну строку не длиннее PREVIEW_LENGTH символов."""
    return clean_spaces(text)[:PREVIEW_LENGTH]


def demo_connection() -> RAGAgent | None:
    """Раздел 1: создаёт агента и проверяет подключение к Pinecone."""
    print_section("1️⃣ Проверка подключения к Pinecone")
    try:
        rag_agent = RAGAgent()
        vector_count = rag_agent.check_connection()
    except Exception as error:
        print_error(f"Не удалось подключиться к Pinecone: {error}")
        return None
    print(f"✅ Подключение успешно. Векторов в индексе: {vector_count}")
    return rag_agent


def demo_add_url(rag_agent: RAGAgent | None, url: str) -> None:
    """Раздел 2: загружает страницу в базу знаний."""
    print_section("2️⃣ Загрузка страницы в базу знаний")
    print(f"🌐 {url}")
    if rag_agent is None:
        print("❌ Агент не создан, поэтому страницу загрузить нельзя")
        return
    try:
        saved_chunks = rag_agent.add_url(url)
    except Exception as error:
        print_error(f"Не удалось загрузить страницу: {error}")
        return
    print(f"✅ Сохранено чанков: {saved_chunks}")


def demo_search(rag_agent: RAGAgent | None, question: str) -> None:
    """Раздел 3: ищет фрагменты по вопросу и печатает их."""
    print_section("3️⃣ Поиск по базе знаний")
    print(f"❓ Вопрос: {question}")
    if rag_agent is None:
        print("❌ Агент не создан, поэтому поиск выполнить нельзя")
        return
    try:
        found = rag_agent.search(question)
    except Exception as error:
        print_error(f"Не удалось выполнить поиск: {error}")
        return
    if not found:
        print("ℹ️ В базе знаний ничего не найдено")
        return
    for number, doc in enumerate(found, start=1):
        source = doc.metadata.get("source", "источник неизвестен")
        print(f"\n📄 Фрагмент {number}")
        print(make_preview(doc.page_content))
        print(f"🔗 Источник: {source}")


def demo_api(rag_agent: RAGAgent | None, question: str) -> None:
    """Раздел 4: вызывает API напрямую, а потом через агента."""
    print_section("4️⃣ Запрос к внешнему API")

    # Прямой вызов инструмента: агент и Pinecone для этого не нужны.
    # У инструмента нет аргументов, поэтому передаём пустой словарь
    fact = call_api.invoke({})
    print(f"🐱 Факт из API: {fact}")

    print(f"\n❓ Вопрос: {question}")
    if rag_agent is None:
        print("❌ Агент не создан, поэтому задать вопрос нельзя")
        return
    try:
        answer = rag_agent.ask(question)
    except Exception as error:
        print_error(f"Агент не смог ответить: {error}")
        return
    print(f"💬 Ответ агента: {answer}")


if __name__ == "__main__":
    setup_logging()

    test_url = "https://ru.wikipedia.org/wiki/Домашняя_кошка"
    test_question = "Сколько живёт домашняя кошка?"

    print_header("🤖 Демонстрация RAG-агента")

    # Каждый раздел сам ловит свои ошибки, поэтому следующие всё равно выполнятся
    agent = demo_connection()
    print_separator()
    demo_add_url(agent, test_url)
    print_separator()
    demo_search(agent, test_question)
    print_separator()
    demo_api(agent, "Получи из API интересный факт про кошку")

    print("\n" + "=" * LINE_WIDTH)
    print("✅ Демонстрация завершена")
