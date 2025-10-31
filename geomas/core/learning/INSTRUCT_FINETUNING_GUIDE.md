# Руководство по Instruct Fine-Tuning для геологических текстов

Полный пайплайн от PDF-документов до обученной LLM модели для ответов на вопросы о геологических месторождениях.

## 📋 Обзор пайплайна

```
PDF документы
    ↓
[1] Парсинг → chunks.json
    ↓
[2] Entity Extraction (BERT NER) → entities
    ↓
[3] QA Generation → qa_pairs.json
    ↓
[4] Instruct Formatting → instruct_dataset_train.json + instruct_dataset_val.json
    ↓
[5] Supervised Fine-Tuning → Обученная LLM
```

## 🚀 Быстрый старт

### Шаг 1: Парсинг PDF документов

```bash
geomas makedataset ./source_pdfs ./parse_results
```

Результат: `./parse_results/chunks.json` с чанками текста.

### Шаг 2: Конвертация аннотаций (опционально)

Если у вас есть аннотации из Label Studio:

```bash
geomas convert-annotations ./АННОТАЦИИ ./bert_training_data
```

Результат: `./bert_training_data/bert_training_data.json`

### Шаг 3: Обучение BERT NER модели (опционально)

Если вы хотите переобучить BERT NER:

```bash
geomas train-bert-ner ./bert_training_data \
    --model-name DeepPavlov/rubert-base-cased \
    --output-dir ./bert_ner_model \
    --epochs 30 \
    --batch-size 8 \
    --learning-rate 2e-5
```

Результат: `./bert_ner_model/` с обученной моделью.

### Шаг 4: Генерация QA-пар

```bash
geomas generate-qa-pairs \
    ./parse_results/chunks.json \
    ./bert_ner_model_folder \
    ./qa_pairs.json \
    --num-pairs 2
```

Результат: `./qa_pairs.json` с вопросами и ответами.

### Шаг 5: Форматирование для instruct fine-tuning

```bash
geomas format-instruct-dataset \
    ./qa_pairs.json \
    ./instruct_dataset.json \
    --format-type alpaca \
    --split-ratio 0.9
```

Результат:
- `./instruct_dataset_train.json` (90% данных)
- `./instruct_dataset_val.json` (10% данных)

### Шаг 6: Supervised Fine-Tuning

```bash
export CUDA_VISIBLE_DEVICES=0

geomas train-sft \
    mistral-7b-4bit \
    ./instruct_dataset_train.json \
    --tag geological_instruct \
    --max-seq-length 2048
```

Результат: Обученная модель в `../models/sft-mistral-7b-v0.3-bnb-4bit/`

## 📊 Формат данных

### QA Pairs Format (`qa_pairs.json`)

```json
[
  {
    "question": "Каковы содержания полезных компонентов в золота 43,7 г/т?",
    "answer": "золота 43,7 г/т, серебра 19,1 г/т...",
    "entity_type": "ORE_COMPONENT",
    "source_text": "Месторождение Агинское...",
    "context": "Средние по месторождению содержания составляют..."
  }
]
```

### Instruct Dataset Format (`instruct_dataset_train.json`)

```json
[
  {
    "instruction": "Опиши содержания полезных компонентов в руде.\n\nВопрос: Каковы содержания...",
    "input": "Средние по месторождению содержания составляют...",
    "output": "золота 43,7 г/т, серебра 19,1 г/т..."
  }
]
```

## 🎯 Поддерживаемые типы геологических сущностей

BERT NER модель распознаёт 17 типов сущностей:

1. **GENERAL_INFO** - Общие сведения о месторождении
2. **ORE_COMPONENT** - Полезные компоненты руд
3. **RESOURCE_POTENTIAL** - Запасы и ресурсный потенциал
4. **ORE_FORMATION** - Рудная формация / тип оруденения
5. **MINERALOGICAL** - Минералогические признаки
6. **TECHNOLOGICAL** - Технологические характеристики
7. **STRATIGRAPHY** - Стратиграфия и типы пород
8. **STRUCTURAL_TECTONIC** - Структурно-тектонические характеристики
9. **ORE_BODIES** - Морфология и залегание рудных тел
10. **ORE_COMPOSITION** - Состав руд
11. **GEODYNAMIC** - Геодинамические характеристики
12. **GEO_CHEMICAL** - Геохимические признаки
13. **METALLOGENIC_CHAR** - Металлогенические характеристики
14. **METASOMATIC** - Метасоматические изменения
15. **FORMATION_CONDITIONS** - Условия формирования
16. **STUDY_INFO** - Изученность месторождения
17. **INFO_SOURCES** - Источники информации

## 🔧 Конфигурация

### Модели для SFT

Доступные модели определены в `geomas/core/utils.py`:

```python
ALLOWED_MODELS = {
    "mistral-7b-4bit": "unsloth/mistral-7b-v0.3-bnb-4bit",
    # Добавьте другие модели здесь
}
```

### Параметры обучения

Конфигурации хранятся в `geomas/core/repository/data/`:
- `cpt-mistral-7b-v03-bnb-4bit.yaml`
- `cpt-gemma-3-1b-pt.yaml`
- и другие...

### Извлечение сущностей из текста

```bash
geomas extract-entities \
    ./bert_ner_model_folder \
    --input-file input.txt \
    --output-file entities.json
```

Или напрямую из командной строки:

```bash
geomas extract-entities \
    ./bert_ner_model_folder \
    --text "Месторождение Агинское находится в Камчатском крае. Содержание золота 43,7 г/т."
```

## 🎓 Примеры использования

### Пример 1: Полный пайплайн от PDF до обученной модели

```bash
# 1. Парсинг
geomas makedataset ./pdfs ./parse_results

# 2. Генерация QA
geomas generate-qa-pairs \
    ./parse_results/chunks.json \
    ./bert_ner_model_folder \
    ./qa_pairs.json

# 3. Форматирование
geomas format-instruct-dataset \
    ./qa_pairs.json \
    ./instruct_dataset.json

# 4. Обучение
geomas train-sft \
    mistral-7b-4bit \
    ./instruct_dataset_train.json \
    --tag my_experiment
```

### Пример 2: Только генерация QA из существующих чанков

```bash
geomas generate-qa-pairs \
    ./my_chunks.json \
    ./my_bert_model \
    ./my_qa_pairs.json \
    --num-pairs 3
```

### Пример 3: Тестирование модели

```python
from geomas.core.inference.bert_ner_inference import load_bert_ner_model

# Загрузить модель
ner_model = load_bert_ner_model("./bert_ner_model_folder")

# Извлечь сущности
text = "Месторождение Агинское содержит золото 43,7 г/т и серебро 19,1 г/т."
result = ner_model.extract_entities(text)

# Вывести результаты
for entity in result.entities:
    print(f"{entity.label}: {entity.text} (confidence: {entity.confidence:.3f})")
```

## 🐛 Troubleshooting

### Проблема: Низкая confidence в извлечённых сущностях

**Решение**: 
1. Увеличьте количество эпох обучения BERT NER
2. Добавьте больше аннотированных данных
3. Проверьте баланс B-/I- labels в данных

### Проблема: QA-пары слишком длинные

**Решение**: Настройте `max_context_length` в `QAGenerator`:

```python
qa_generator = QAGenerator(
    add_context=True,
    max_context_length=300  # Уменьшите это значение
)
```

### Проблема: Out of memory при SFT

**Решение**:
1. Уменьшите `max_seq_length` (например, до 1024)
2. Уменьшите batch size в конфигурации
3. Используйте gradient checkpointing







