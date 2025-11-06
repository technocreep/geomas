# Отчет об интеграции OCR модуля в архитектуру geomas

## Выполненные изменения

### 1. Структурная интеграция
- **Перемещение**: OCR модуль интегрирован в `geomas/core/ocr_modules/`
- **Архитектура**: Сохранена модульная структура OCR с адаптацией под geomas паттерны
- **Структура**:
  ```
  geomas/core/ocr_modules/
  ├── __init__.py           # Lazy-loaded API экспорт
  ├── api.py               # High-level API с geomas логированием
  ├── models/              # OCR адаптеры и базовые классы
  │   ├── adapters/        # Marker, MinerU, OlmOCR, QwenVL
  │   ├── base.py         
  │   └── batch.py
  ├── conversion/          # Конвертация документов
  ├── io/                  # I/O операции
  ├── pdf_preproc/         # Предобработка PDF
  ├── runtime/             # Оркестрация выполнения  
  └── standardize/         # Нормализация к Markdown
  ```

### 2. API интеграция
- **Создан**: `geomas/core/api/ocr.py` - high-level OCR API 
- **Классы**: 
  - `OcrApi` - основной API класс с поддержкой всех адаптеров
  - `process_document_ocr()` - функция для одного документа
  - `process_documents_ocr()` - функция для множественных документов
- **Паттерны**: Следует архитектуре geomas с использованием логирования и репозиториев

### 3. Repository pattern
- **Создан**: `geomas/core/repository/ocr_repository.py`
- **Функции**:
  - Lazy-loading OCR адаптеров
  - Управление конфигурациями
  - Константы и настройки по умолчанию
  - Проверка поддерживаемых форматов файлов
- **Конфигурация**: `geomas/core/config/ocr-config.yaml`

### 4. CLI интеграция
- **Команда `geomas ocr`**:
  - Обработка файла или папки
  - Выбор OCR адаптера
  - Настройка параметров (batch_size, language, directories)
  - Интегрированное логирование
- **Команда `geomas ocr-adapters`**:
  - Показ доступных адаптеров

### 5. Зависимости
- **Базовые**: добавлены в основной `pyproject.toml`
- **Расширенные**: вынесены в `[project.optional-dependencies.ocr]`
- **Установка**: `pip install -e .[ocr]` для полного функционала

## Тестирование

1. **Импорт репозитория**: 
   ```python
   from geomas.core.repository.ocr_repository import get_ocr_adapter_names
   print('Available adapters:', get_ocr_adapter_names())
   # → Available adapters: ['marker', 'mineru', 'olmocr', 'qwen_vl']
   ```

2. **API создание**:
   ```python
   from geomas.core.api.ocr import OcrApi
   api = OcrApi('marker')
   print(f'OCR API создан с адаптером: {api.adapter_name}')
   # → OCR API создан с адаптером: marker
   ```

3. **CLI команды**:
   ```bash
   geomas --help               # OCR команды видны
   geomas ocr-adapters         # Показывает все 4 адаптера
   ```

## Как использовать

### Python API
```python
from geomas.core.api.ocr import OcrApi, process_document_ocr

# Простое использование
result = process_document_ocr("document.pdf", adapter_name="marker")

# Расширенное использование  
api = OcrApi("marker")
results = api.process_documents(
    ["doc1.pdf", "doc2.pdf"],
    output_dir="output/markdown",
    batch_size=16,
    language="ru"
)
```

### CLI
```bash
# Обработка файла
geomas ocr document.pdf --adapter marker --output-dir output/

# Обработка папки с настройками
geomas ocr ./documents/ --adapter mineru --batch-size 16 --language ru

# Показать адаптеры
geomas ocr-adapters
```

## Архитектурные принципы

**Repository Pattern** - для конфигураций и адаптеров  
**Dependency Injection** - через параметры функций  
**Модульность** - четкое разделение ответственности  
**Логирование** - интегрировано с geomas logger  
**CLI Integration** - через typer с помощью и валидацией  
**Configuration** - YAML конфиги по примеру других модулей  

