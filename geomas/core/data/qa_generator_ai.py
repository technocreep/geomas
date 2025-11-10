"""
QA Generator AI - генерация вопросов-ответов с помощью Google Gemini API.

Этот модуль использует Google Gemini для создания высококачественных QA пар
из геологических текстов PDF документов.
"""

import json
import os
import time
import math
from pathlib import Path
from typing import List, Dict, Tuple, Optional

try:
    import PyPDF2
    import google.generativeai as genai
except ImportError:
    raise ImportError(
        "Требуется установка зависимостей: pip install PyPDF2 google-generativeai"
    )

from geomas.core.logging.logger import get_logger

_log = get_logger("QA_GENERATOR_AI")


class QAGeneratorAI:
    """
    Генератор QA пар с использованием Google Gemini API.
    
    Включает функциональность:
    1. Разбиение больших PDF на части
    2. Генерация вопросов-ответов через Gemini API
    3. Очистка и постобработка результатов
    """
    
    PROMPT_TEMPLATE = """Ты — эксперт-геолог, специализирующийся на создании обучающих данных для языковых моделей.

ЗАДАЧА: Из предоставленного геологического текста создай набор вопросов-ответов для обучения геологического ассистента.

ИНСТРУКЦИИ:

1. ТИПЫ ВОПРОСОВ (создай по 2-3 вопроса каждого типа):

   a) Общая информация:
      - Где находится месторождение?
      - Какие компании разрабатывают месторождение?
      - К какому административному району относится?

   b) Содержания и запасы:
      - Каковы содержания золота и серебра?
      - Какие запасы утверждены для месторождения?
      - Каков ресурсный потенциал?

   c) Геологическая характеристика:
      - К какому типу оруденения относится месторождение?
      - Какова рудная формация?
      - Какие минералы присутствуют в рудах?

   d) Рудные тела:
      - Какова морфология рудных тел?
      - Каковы размеры и условия залегания рудных зон?
      - Какова мощность рудных тел?

   e) Технологические характеристики:
      - Какая технология обогащения применяется?
      - Каково извлечение полезных компонентов?
      - Какой способ отработки месторождения?

   f) История изучения:
      - Когда было открыто месторождение?
      - Какие геологоразведочные работы проводились?
      - Когда были утверждены запасы?

2. ТРЕБОВАНИЯ К ВОПРОСАМ:
   - Вопрос должен быть конкретным и четким
   - Используй естественный язык (как спросил бы реальный геолог)
   - Избегай слишком длинных вопросов (макс. 15-20 слов)
   - Разнообразь формулировки ("Каков...", "Опиши...", "Какие...", "Расскажи о...")

3. ТРЕБОВАНИЯ К ОТВЕТАМ:
   - Ответ должен быть точным и содержать конкретные данные из текста
   - Используй полные предложения, но будь лаконичен
   - Включай числовые данные, даты, названия
   - Ответ должен прямо отвечать на вопрос
   - Если предполагается в вопросе упоминать месторождение, то это необходимо делать обязательно с названием
   - Не нужно писать ответ, если в тексте нет информации по какой-то категории

4. ФОРМАТ ВЫХОДА (JSON):

```json
[
  {{
    "question": "Где находится месторождение Агинское?",
    "answer": "Месторождение Агинское находится в центральной части Камчатского полуострова, западнее Срединного хребта, на территории Быстринского административного района.",
    "category": "GENERAL_INFO"
  }},
  {{
    "question": "Каковы средние содержания золота и серебра на месторождении Агинское?",
    "answer": "Средние по месторождению содержания составляют: золота 43,7 г/т, серебра 19,1 г/т. В наиболее богатых рудных столбах содержания золота достигают 1000-6000 г/т, максимальное — 6120 г/т.",
    "category": "ORE_COMPONENT"
  }}
]
```

5. КАТЕГОРИИ:
   - "GENERAL_INFO" - Общие сведения (лицензия, положение, инфраструктура, физико-географические условия)
   - "STUDY_INFO" - Изученность – общая информация (объемы и виды работ)
   - "ORE_FORMATION" - Рудная формация/ Геолого-промышленный тип оруденения
   - "ORE_COMPONENT" - Полезный компонент руд
   - "RESOURCE_POTENTIAL" - Ресурсный потенциал
   - "METALLOGENIC_CHAR" - Металлогенические характеристики
   - "STRUCTURAL_CHAR" - Структурно-тектонические характеристики
   - "ORE_BODIES" - Рудные зоны / тела (морфология, размеры и условия залегания)
   - "GEO_CHEMICAL" - Геохимические признаки
   - "ORE_COMPOSITION" - Состав руд
   - "MINERALOGICAL" - Минералогические признаки
   - "METASOMATIC" - Метасоматические изменения
   - "FORMATION_CONDITIONS" - Условия формирования
   - "TECHNOLOGICAL" - Технологические признаки / обогащение / горное дело
   - "SOURCES" - Источники информации
   - "STRATIGRAPHY" - Стратиграфия и типы пород
   - "GEODYNAMIC" - Геодинамические характеристики

6. ВАЖНО:
   - НЕ выдумывай информацию — используй только данные из текста
   - Если в тексте нет информации по какой-то категории — пропусти её
   - Создай минимум 10-15 вопросов, максимум 30
   - Избегай дублирования — каждый вопрос должен быть уникальным

ТЕКСТ ДЛЯ АНАЛИЗА:

{text}

ВЕРНИ ТОЛЬКО ВАЛИДНЫЙ JSON С МАССИВОМ ВОПРОСОВ-ОТВЕТОВ БЕЗ ДОПОЛНИТЕЛЬНОГО ТЕКСТА."""
    
    def __init__(
        self,
        api_key: str,
        model_name: str = "gemini-2.5-flash-lite",
        temperature: float = 0.2,
        max_output_tokens: int = 8192
    ):
        """
        Инициализация генератора AI.
        
        Args:
            api_key: API ключ Google Gemini
            model_name: Название модели Gemini
            temperature: Температура генерации (0.0-1.0)
            max_output_tokens: Максимальное количество токенов в ответе
        """
        self.api_key = api_key
        self.model_name = model_name
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        
        # Настройка Gemini API
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        
        _log.info(f"QAGeneratorAI инициализирован (модель={model_name})")
    
    # ========== ФУНКЦИИ ДЛЯ РАЗБИЕНИЯ PDF (из клетки 3) ==========
    
    @staticmethod
    def extract_text_from_pdf(pdf_path: str) -> str:
        """
        Извлекает текст из PDF файла.
        
        Args:
            pdf_path: Путь к PDF файлу
            
        Returns:
            Извлеченный текст
        """
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    text += page.extract_text() + "\n"
                return text.strip()
        except Exception as e:
            _log.error(f"Ошибка при чтении PDF {pdf_path}: {str(e)}")
            return ""
    
    @staticmethod
    def get_pdf_info(pdf_path: str) -> Tuple[int, str]:
        """
        Получает информацию о PDF файле.
        
        Args:
            pdf_path: Путь к PDF файлу
            
        Returns:
            Кортеж (количество страниц, размер файла)
        """
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                num_pages = len(pdf_reader.pages)
            
            file_size = os.path.getsize(pdf_path)
            size_mb = file_size / (1024 * 1024)
            
            return num_pages, f"{size_mb:.2f} МБ"
        except Exception as e:
            _log.error(f"Ошибка при получении информации о PDF: {str(e)}")
            return 0, "0 МБ"
    
    @staticmethod
    def validate_page_range(
        total_pages: int,
        start: Optional[int],
        end: Optional[int]
    ) -> Tuple[int, int]:
        """
        Проверяет и корректирует диапазон страниц.
        
        Args:
            total_pages: Общее количество страниц
            start: Начальная страница (с 1)
            end: Конечная страница (с 1)
            
        Returns:
            Кортеж (начальный_индекс, конечный_индекс) с 0
        """
        if start is None:
            start = 1
        if end is None:
            end = total_pages
        
        if start < 1:
            _log.warning(f"Начальная страница {start} < 1, установлено значение 1")
            start = 1
        
        if start > total_pages:
            _log.warning(f"Начальная страница {start} > {total_pages}")
            start = total_pages
        
        if end < start:
            _log.warning(f"Конечная страница {end} < начальной {start}")
            end = start
        
        if end > total_pages:
            _log.warning(f"Конечная страница {end} > {total_pages}")
            end = total_pages
        
        # Преобразуем в индексы (с 0)
        start_idx = start - 1
        end_idx = end
        
        return start_idx, end_idx
    
    def split_pdf_by_pages(
        self,
        input_pdf: str,
        output_dir: str,
        pages_per_file: int = 10,
        start_page: Optional[int] = None,
        end_page: Optional[int] = None,
        prefix: str = "part"
    ) -> List[str]:
        """
        Разбивает PDF по количеству страниц в каждом файле.
        
        Args:
            input_pdf: Путь к исходному PDF
            output_dir: Директория для сохранения частей
            pages_per_file: Количество страниц в каждом файле
            start_page: Начальная страница (с 1, None = с начала)
            end_page: Конечная страница (с 1, None = до конца)
            prefix: Префикс имен выходных файлов
            
        Returns:
            Список путей к созданным файлам
        """
        try:
            with open(input_pdf, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                total_pages = len(pdf_reader.pages)
                
                start_idx, end_idx = self.validate_page_range(
                    total_pages, start_page, end_page
                )
                working_pages = end_idx - start_idx
                
                _log.info(f"Всего страниц: {total_pages}")
                _log.info(f"Диапазон: {start_idx + 1}-{end_idx} ({working_pages} стр.)")
                
                num_files = math.ceil(working_pages / pages_per_file)
                _log.info(f"Будет создано файлов: {num_files}")
                
                os.makedirs(output_dir, exist_ok=True)
                
                created_files = []
                
                for i in range(num_files):
                    file_start = start_idx + (i * pages_per_file)
                    file_end = min(start_idx + ((i + 1) * pages_per_file), end_idx)
                    
                    pdf_writer = PyPDF2.PdfWriter()
                    
                    for page_num in range(file_start, file_end):
                        pdf_writer.add_page(pdf_reader.pages[page_num])
                    
                    output_filename = f"{prefix}_{i+1:03d}_pages_{file_start+1}-{file_end}.pdf"
                    output_path = os.path.join(output_dir, output_filename)
                    
                    with open(output_path, 'wb') as output_file:
                        pdf_writer.write(output_file)
                    
                    file_size = os.path.getsize(output_path) / 1024
                    _log.info(
                        f"Создан: {output_filename} "
                        f"({file_end - file_start} стр., {file_size:.1f} КБ)"
                    )
                    
                    created_files.append(output_path)
                
                return created_files
                
        except Exception as e:
            _log.error(f"Ошибка при разбиении PDF: {str(e)}")
            return []
    
    # ========== ФУНКЦИИ ДЛЯ ГЕНЕРАЦИИ QA (из клетки 2) ==========
    
    @staticmethod
    def clean_json_response(response_text: str) -> str:
        """
        Очищает ответ от markdown форматирования и извлекает JSON.
        
        Args:
            response_text: Ответ модели
            
        Returns:
            Очищенный JSON текст
        """
        response_text = response_text.strip()
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        elif response_text.startswith("```"):
            response_text = response_text[3:]
        
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        
        return response_text.strip()
    
    def process_pdf_with_gemini(self, pdf_path: str) -> List[Dict]:
        """
        Обрабатывает один PDF файл через Gemini API.
        
        Args:
            pdf_path: Путь к PDF файлу
            
        Returns:
            Список QA пар
        """
        _log.info(f"Обработка файла: {os.path.basename(pdf_path)}")
        
        # Извлекаем текст из PDF
        text = self.extract_text_from_pdf(pdf_path)
        
        if not text:
            _log.error("Не удалось извлечь текст из PDF")
            return []
        
        _log.info(f"Извлечено {len(text)} символов текста")
        
        # Формируем промпт
        prompt = self.PROMPT_TEMPLATE.format(text=text)
        
        # Отправляем запрос к Gemini
        try:
            _log.info("Отправка запроса к Gemini API...")
            response = self.model.generate_content(
                prompt,
                generation_config={
                    "temperature": self.temperature,
                    "top_p": 0.95,
                    "top_k": 40,
                    "max_output_tokens": self.max_output_tokens,
                }
            )
            
            _log.info("Получен ответ от Gemini")
            
            # Парсим JSON ответ
            response_text = response.text
            cleaned_response = self.clean_json_response(response_text)
            
            qa_pairs = json.loads(cleaned_response)
            
            _log.info(f"Сгенерировано {len(qa_pairs)} вопросов-ответов")
            
            # Добавляем метаданные
            for qa in qa_pairs:
                qa['source_file'] = os.path.basename(pdf_path)
            
            return qa_pairs
            
        except json.JSONDecodeError as e:
            _log.error(f"Ошибка парсинга JSON: {str(e)}")
            _log.debug(f"Ответ модели: {response.text[:500]}...")
            return []
        except Exception as e:
            _log.error(f"Ошибка при обработке: {str(e)}")
            return []
    
    def process_all_pdfs(
        self,
        pdf_dir: str,
        output_file: str,
        delay_seconds: int = 2
    ) -> int:
        """
        Обрабатывает все PDF файлы в директории.
        
        Args:
            pdf_dir: Директория с PDF файлами
            output_file: Путь к выходному JSON файлу
            delay_seconds: Задержка между запросами (для соблюдения лимитов API)
            
        Returns:
            Количество сгенерированных QA пар
        """
        pdf_files = list(Path(pdf_dir).glob("*.pdf"))
        
        if not pdf_files:
            _log.error(f"Не найдено PDF файлов в директории: {pdf_dir}")
            return 0
        
        _log.info(f"Найдено {len(pdf_files)} PDF файлов для обработки")
        
        all_qa_pairs = []
        successful_files = 0
        failed_files = 0
        
        for idx, pdf_file in enumerate(pdf_files, 1):
            _log.info(f"[{idx}/{len(pdf_files)}] Обработка: {pdf_file.name}")
            
            try:
                qa_pairs = self.process_pdf_with_gemini(str(pdf_file))
                
                if qa_pairs:
                    all_qa_pairs.extend(qa_pairs)
                    successful_files += 1
                    _log.info(f"Успешно обработан файл {pdf_file.name}")
                else:
                    failed_files += 1
                    _log.warning(f"Файл {pdf_file.name} не дал QA пар")
                
                # Задержка между запросами
                if idx < len(pdf_files):
                    time.sleep(delay_seconds)
                    
            except Exception as e:
                failed_files += 1
                _log.error(f"Ошибка при обработке файла {pdf_file.name}: {str(e)}")
                continue
        
        # Сохраняем результаты
        if all_qa_pairs:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(all_qa_pairs, f, ensure_ascii=False, indent=2)
            
            _log.info(f"Результаты сохранены в: {output_file}")
            _log.info(f"Всего сгенерировано QA пар: {len(all_qa_pairs)}")
            _log.info(f"Успешно обработано файлов: {successful_files}")
            _log.info(f"Не удалось обработать: {failed_files}")
            
            # Статистика по категориям
            categories = {}
            for qa in all_qa_pairs:
                cat = qa.get('category', 'UNKNOWN')
                categories[cat] = categories.get(cat, 0) + 1
            
            _log.info("Статистика по категориям:")
            for cat, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
                _log.info(f"  {cat}: {count}")
        else:
            _log.error("Не получено ни одной QA пары")
        
        return len(all_qa_pairs)
    
    # ========== ФУНКЦИИ ДЛЯ ОЧИСТКИ РЕЗУЛЬТАТОВ (из клетки 5) ==========
    
    @staticmethod
    def clean_qa_results(
        input_file: str,
        output_file: Optional[str] = None,
        remove_keys: List[str] = None
    ) -> int:
        """
        Очищает результаты QA от ненужных ключей.
        
        Args:
            input_file: Путь к входному JSON файлу
            output_file: Путь к выходному файлу (если None, перезаписывает входной)
            remove_keys: Список ключей для удаления (по умолчанию ['source_file'])
            
        Returns:
            Количество обработанных записей
        """
        if remove_keys is None:
            remove_keys = ['source_file']
        
        if output_file is None:
            output_file = input_file
        
        try:
            # Загружаем данные
            with open(input_file, 'r', encoding='utf-8') as f:
                qa_data = json.load(f)
            
            # Очищаем данные
            cleaned_data = []
            for item in qa_data:
                if isinstance(item, dict):
                    for key in remove_keys:
                        if key in item:
                            del item[key]
                cleaned_data.append(item)
            
            # Сохраняем результат
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(cleaned_data, f, ensure_ascii=False, indent=2)
            
            _log.info(f"Ключи {remove_keys} удалены из файла {input_file}")
            _log.info(f"Обработано записей: {len(cleaned_data)}")
            
            return len(cleaned_data)
            
        except FileNotFoundError:
            _log.error(f"Файл {input_file} не найден")
            return 0
        except json.JSONDecodeError:
            _log.error(f"Не удалось декодировать JSON из файла {input_file}")
            return 0
        except Exception as e:
            _log.error(f"Ошибка при очистке результатов: {str(e)}")
            return 0

