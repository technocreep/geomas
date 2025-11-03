# GEOMAS

**GEOMAS** — комплексная система для обработки геологических документов, извлечения семантических сущностей и обучения языковых моделей для ответов на вопросы о месторождениях полезных ископаемых.

## 📋 Описание

GEOMAS реализует полный пайплайн от PDF-документов до обученной LLM модели:

1. **Парсинг PDF** документов в текстовые чанки
2. **Извлечение сущностей** с помощью дообученной BERT NER модели
3. **Генерация QA-пар** на основе извлеченных сущностей
4. **Форматирование в instruct-формат** для дообучения LLM
5. **Supervised Fine-Tuning (SFT)** для обучения модели отвечать на геологические вопросы

## 🚀 Быстрый старт

### Установка

```bash
# Клонируйте репозиторий
git clone <repository-url>
cd geomas

# Установите в режиме разработки
pip install -e .
```

### Проверка системы

```bash
geomas health
```

Команда выводит информацию о:
- Версии Python, PyTorch, Unsloth
- Доступности CUDA
- Доступных GPU устройствах

## 📖 Пайплайн инструкции

Полная пошаговая инструкция по использованию пайплайна доступна в:
- **[INSTRUCT_FINETUNING_GUIDE.md](INSTRUCT_FINETUNING_GUIDE.md)** — детальное руководство по instruct fine-tuning
- **[BERT_NER_GUIDE.md](BERT_NER_GUIDE.md)** — руководство по BERT NER модели

## 🔧 Основные команды

### 1. Парсинг PDF документов

Преобразование PDF-файлов в JSON формат с чанками текста:

```bash
geomas makedataset ./source_pdfs ./parse_results
```

**Результат:** `./parse_results/chunks.json` — массив текстовых чанков

---

### 2. Конвертация аннотаций (опционально)

Если у вас есть аннотации из Label Studio для обучения BERT NER:

```bash
geomas convert-annotations ./АННОТАЦИИ ./bert_training_data
```

**Результат:** `./bert_training_data/bert_training_data.json` — данные для обучения BERT

---

### 3. Обучение BERT NER модели (опционально)

Дообучение BERT модели для извлечения геологических сущностей:

```bash
geomas train-bert-ner ./bert_training_data \
    --model-name DeepPavlov/rubert-base-cased \
    --output-dir ./bert_ner_model \
    --epochs 30 \
    --batch-size 8 \
    --learning-rate 2e-5
```

**Параметры:**
- `--model-name` — модель BERT для дообучения (по умолчанию: `DeepPavlov/rubert-base-cased`)
- `--output-dir` — директория для сохранения модели
- `--epochs` — количество эпох обучения
- `--batch-size` — размер батча
- `--learning-rate` — learning rate

**Результат:** Обученная модель в `./bert_ner_model/`

---

### 4. Извлечение сущностей

Использование обученной BERT NER модели для извлечения сущностей из текста:

```bash
# Из командной строки
geomas extract-entities ./bert_ner_model \
    --text "Месторождение Агинское содержит золото 43,7 г/т и серебро 19,1 г/т."

# Из файла
geomas extract-entities ./bert_ner_model \
    --input-file input.txt \
    --output-file entities.json
```

**Результат:** JSON с извлеченными сущностями и их типами

**Типы сущностей:**
- `GENERAL_INFO` — Общая информация
- `ORE_COMPONENT` — Полезные компоненты (золото, серебро)
- `RESOURCE_POTENTIAL` — Запасы и ресурсы
- `ORE_BODIES` — Рудные тела
- `MINERALOGICAL` — Минералогический состав
- `TECHNOLOGICAL` — Технологические параметры
- `STUDY_INFO` — История изучения
- И другие (всего 17 типов)

---

### 5. Генерация QA-пар

Создание вопросов-ответов на основе извлеченных сущностей:

```bash
geomas generate-qa-pairs \
    ./parse_results/chunks.json \
    ./bert_ner_model \
    ./qa_pairs.json \
    --num-pairs 2
```

**Параметры:**
- `chunks.json` — файл с чанками (из команды `makedataset`)
- `bert_ner_model` — путь к обученной BERT NER модели
- `--num-pairs` — количество QA-пар на сущность (по умолчанию: 2)
- `--no-context` — не добавлять контекст в ответы

**Результат:** `./qa_pairs.json` — массив QA-пар

---

### 6. Форматирование в instruct-формат

Преобразование QA-пар в формат для instruct fine-tuning:

```bash
geomas format-instruct-dataset \
    ./qa_pairs.json \
    ./instruct_dataset.json \
    --format-type alpaca \
    --split-ratio 0.9
```

**Параметры:**
- `--format-type` — формат данных (`alpaca` или `chat`, по умолчанию: `alpaca`)
- `--split-ratio` — соотношение train/validation (по умолчанию: 0.9)
- `--no-context` — не включать контекст в примеры

**Результат:**
- `./instruct_dataset_train.json` — тренировочный датасет (90%)
- `./instruct_dataset_val.json` — валидационный датасет (10%)

---

### 7. Supervised Fine-Tuning (SFT)

Дообучение LLM на instruct-датасете:

```bash
# Linux/Mac
export CUDA_VISIBLE_DEVICES=0

geomas train-sft \
    mistral-7b-4bit \
    ./instruct_dataset_train.json \
    --tag geological_qa_v1 \
    --max-seq-length 2048
```

```powershell
# Windows PowerShell
$env:CUDA_VISIBLE_DEVICES="0"

geomas train-sft `
    mistral-7b-4bit `
    instruct_dataset_train.json `
    --tag geological_qa_v1 `
    --max-seq-length 2048
```

**Параметры:**
- `mistral-7b-4bit` — модель для дообучения (см. список ниже)
- `--tag` — тег эксперимента для MLflow
- `--max-seq-length` — максимальная длина последовательности (по умолчанию: 2048)

**Доступные модели:**
```python
"mistral-7b-4bit"     # Mistral 7B (4-bit quantization)
"mistral-7b"          # Mistral 7B (full precision)
"minitron-8b"         # NVIDIA Mistral NeMo Minitron 8B
"mistral-nemo-12b"    # Mistral NeMo Base 12B
"qwen3-14b"           # Qwen3 14B (4-bit)
"gemma-3-1b-4bit"     # Gemma 3 1B (4-bit)
"gemma-3-1b"          # Gemma 3 1B
"gemma-3-27b"         # Gemma 3 27B
"qwen3-30b-4bit"      # Qwen3 30B (4-bit)
```

**Результат:** Обученная модель сохраняется в `../models/sft-{model-name}/`

---

## 📊 Полный пример пайплайна

```bash
# Шаг 1: Парсинг PDF
geomas makedataset ./geological_pdfs ./parse_results

# Шаг 2: Генерация QA-пар
geomas generate-qa-pairs \
    parse_results/chunks.json \
    D:\bert_ner_output_v10_2812examples \
    qa_pairs_final.json \
    --num-pairs 2

# Шаг 3: Форматирование в instruct-формат
geomas format-instruct-dataset \
    qa_pairs_final.json \
    instruct_dataset_final.json \
    --format-type alpaca \
    --split-ratio 0.9

# Шаг 4: Дообучение LLM
export CUDA_VISIBLE_DEVICES=0
geomas train-sft \
    mistral-7b-4bit \
    instruct_dataset_final_train.json \
    --tag geological_model_production \
    --max-seq-length 2048
```

---

## 🔄 Continued Pretraining (старый метод)

Для continued pretraining используйте команду `train`:

```bash
export CUDA_VISIBLE_DEVICES=0
geomas train MODEL DATASET_PATH TAG
```

### Настройка MLflow

Перед использованием `train` необходимо запустить MLflow:

```bash
docker ps -a | grep mlflow
```

Должны быть запущены 3 контейнера:
- `mlflow_postgres` — PostgreSQL на порту `5432`
- `mlflow_minio` — MinIO (S3) на порту `9000`
- `mlflow-mlflow-1` — MLflow сервер на порту `5000`

Создайте файл `.env` в директории `geomas`:

```env
MLFLOW_TRACKING_URI=http://127.0.0.1:5000
MLFLOW_S3_ENDPOINT_URL=http://127.0.0.1:9000
AWS_ACCESS_KEY_ID=***
AWS_SECRET_ACCESS_KEY=***
```

---

## 🛠️ Разработка

### Установка зависимостей для разработки

```bash
pip install -e ".[dev]"
```

### Код-стиль

Используйте Makefile для проверки и форматирования кода:

```bash
make lint    # Проверка кода с Ruff
make fix     # Автоисправление ошибок
make format  # Форматирование кода
make sort    # Сортировка импортов
make all     # Запуск всех проверок
```

---

## 📁 Структура проекта

```
geomas/
├── geomas/
│   ├── cli.py                    # CLI интерфейс (Typer)
│   ├── core/
│   │   ├── data/
│   │   │   ├── annotation_converter.py    # Конвертация аннотаций
│   │   │   ├── qa_generator.py            # Генерация QA-пар
│   │   │   └── instruct_formatter.py       # Форматирование в instruct
│   │   ├── inference/
│   │   │   └── bert_ner_inference.py      # BERT NER inference
│   │   ├── learning/
│   │   │   ├── bert_ner_trainer.py         # Обучение BERT NER
│   │   │   ├── sft_trainer.py             # SFT обучение
│   │   │   └── continued_pretrain.py     # Continued pretraining
│   │   ├── rag_modules/
│   │   │   └── convertation/
│   │   │       └── pdf_to_json.py         # PDF парсинг
│   │   └── utils.py                       # Утилиты и константы
│   └── api/                               # API модули
├── INSTRUCT_FINETUNING_GUIDE.md          # Инструкция по instruct fine-tuning
├── BERT_NER_GUIDE.md                      # Инструкция по BERT NER
└── README.md                              # Этот файл
```

---

## 🐛 Troubleshooting

### Проблема: CUDA out of memory при SFT

**Решение:** Уменьшите `max_seq_length`:
```bash
--max-seq-length 1024
```

### Проблема: Низкое качество QA-пар

**Решение:**
- Уменьшите `--num-pairs` до 1
- Проверьте качество BERT NER модели
- Увеличьте количество эпох обучения BERT

### Проблема: BERT NER предсказывает только 'O' labels

**Решение:**
- Проверьте баланс B-/I- labels в данных
- Убедитесь, что данные используют правильную WordPiece токенизацию
- См. `BERT_NER_GUIDE.md` для детальной диагностики

### Проблема: Ошибка при загрузке модели

**Решение:** Проверьте путь к модели и наличие всех файлов:
```bash
ls ./bert_ner_model/
# Должны быть: config.json, model.safetensors, label2id.json, id2label.json
```

---

## 📚 Дополнительная документация

- **[INSTRUCT_FINETUNING_GUIDE.md](INSTRUCT_FINETUNING_GUIDE.md)** — полное руководство по instruct fine-tuning пайплайну
- **[BERT_NER_GUIDE.md](BERT_NER_GUIDE.md)** — руководство по обучению и использованию BERT NER
- **[ANNOTATION_CONVERTER_GUIDE.md](ANNOTATION_CONVERTER_GUIDE.md)** — конвертация аннотаций из Label Studio

---

## 📝 Лицензия

См. файл [LICENSE](LICENSE) в корне проекта.

---

## 🤝 Вклад в проект

При добавлении новых типов сущностей:
1. Обновите шаблоны вопросов в `geomas/core/data/qa_generator.py`
2. Добавьте инструкции в `geomas/core/data/instruct_formatter.py`
3. Переобучите BERT NER модель с новыми метками

---

## 📧 Контакты

Для вопросов и предложений создавайте issues в репозитории.
