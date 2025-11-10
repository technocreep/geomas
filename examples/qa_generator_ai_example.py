#!/usr/bin/env python3
"""
Тестовый скрипт для проверки QAGeneratorAI.
"""

import os
import sys
from pathlib import Path

# Добавляем корневую директорию в PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent))

from geomas.core.data.qa_generator_ai import QAGeneratorAI


def print_separator(title: str):
    """Печатает разделитель с заголовком."""
    print("\n" + "="*80)
    print(f" {title}")
    print("="*80 + "\n")


def test_1_pdf_splitting():
    """
    Тест 1: Проверка разбиения PDF на части (из клетки 3 ноутбука).
    """
    print_separator("ТЕСТ 1: Разбиение PDF на части")
    
    # Путь к тестовому PDF файлу
    # input_pdf = input("Введите путь к PDF файлу для разбиения (или Enter для пропуска): ").strip()
    input_pdf = r"Агинское _ Дальневосточный _ ПРАЙМ ЗОЛОТО.pdf"
    
    if not input_pdf or not os.path.exists(input_pdf):
        print("⚠ PDF файл не указан или не найден. Пропускаем тест разбиения.")
        return None
    
    # Параметры разбиения
    output_dir = "./test_split_pdfs"
    pages_per_file = 5
    
    print(f"📄 Входной файл: {input_pdf}")
    print(f"📁 Директория вывода: {output_dir}")
    print(f"📊 Страниц в каждом файле: {pages_per_file}")
    
    # Создаем генератор (API ключ не нужен для разбиения PDF)
    try:
        generator = QAGeneratorAI(api_key="dummy_key_for_splitting")
    except Exception:
        # Если не удалось инициализировать из-за отсутствия зависимостей, используем статические методы
        print("⚠ Используем статические методы для разбиения")
    
    # Получаем информацию о PDF
    print("\n📋 Информация о файле:")
    num_pages, file_size = QAGeneratorAI.get_pdf_info(input_pdf)
    print(f"  • Количество страниц: {num_pages}")
    print(f"  • Размер файла: {file_size}")
    
    if num_pages == 0:
        print("❌ Не удалось получить информацию о PDF файле")
        return None
    
    # Запрашиваем диапазон страниц
    # start_page = input(f"\nНачальная страница (1-{num_pages}, Enter = 1): ").strip()
    # end_page = input(f"Конечная страница (1-{num_pages}, Enter = {num_pages}): ").strip()
    
    # start_page = int(start_page) if start_page else None
    # end_page = int(end_page) if end_page else None
    start_page = None  # Начинаем с первой страницы
    end_page = None    # До последней страницы
    
    # Разбиваем PDF
    print("\n⏳ Разбиение PDF файла...")
    generator = QAGeneratorAI(api_key="dummy_key")
    created_files = generator.split_pdf_by_pages(
        input_pdf=input_pdf,
        output_dir=output_dir,
        pages_per_file=pages_per_file,
        start_page=start_page,
        end_page=end_page,
        prefix="test_part"
    )
    
    if created_files:
        print(f"\n✅ УСПЕШНО! Создано файлов: {len(created_files)}")
        print(f"📁 Файлы сохранены в: {output_dir}")
        print("\n📝 Список созданных файлов:")
        for f in created_files[:5]:  # Показываем первые 5
            print(f"  • {os.path.basename(f)}")
        if len(created_files) > 5:
            print(f"  ... и ещё {len(created_files) - 5} файлов")
        return output_dir
    else:
        print("❌ Не удалось разбить PDF файл")
        return None


def test_2_qa_generation(pdf_dir: str = None):
    """
    Тест 2: Генерация QA пар через Gemini API (из клетки 2 ноутбука).
    """
    print_separator("ТЕСТ 2: Генерация QA пар через Gemini API")
    
    # Запрашиваем API ключ
    # api_key = input("Введите API ключ Google Gemini (или Enter для пропуска): ").strip()
    api_key = "API-KEY HERE"  # Пропускаем тест если не указан ключ
    
    if not api_key:
        print("⚠ API ключ не указан. Пропускаем тест генерации QA.")
        return None
    
    # Запрашиваем директорию с PDF
    if pdf_dir is None:
        # pdf_dir = input("Введите путь к директории с PDF файлами: ").strip()
        pdf_dir = ""  # Будет пропущено если не указано
    
    if not pdf_dir or not os.path.exists(pdf_dir):
        print("❌ Директория не указана или не существует")
        return None
    
    # Параметры генерации
    output_file = "./test_qa_results.json"
    model_name = "gemini-2.5-flash-lite"
    
    print(f"\n📁 Директория с PDF: {pdf_dir}")
    print(f"💾 Выходной файл: {output_file}")
    print(f"🤖 Модель: {model_name}")
    
    # Создаем генератор
    try:
        generator = QAGeneratorAI(
            api_key=api_key,
            model_name=model_name,
            temperature=0.2
        )
    except Exception as e:
        print(f"❌ Ошибка инициализации генератора: {e}")
        return None
    
    # Обрабатываем PDF файлы
    print("\n⏳ Начинаем обработку PDF файлов...")
    print("   (это может занять несколько минут)")
    
    num_qa_pairs = generator.process_all_pdfs(
        pdf_dir=pdf_dir,
        output_file=output_file,
        delay_seconds=2
    )
    
    if num_qa_pairs > 0:
        print(f"\n✅ УСПЕШНО! Сгенерировано QA пар: {num_qa_pairs}")
        print(f"💾 Результаты сохранены в: {output_file}")
        
        # Показываем примеры QA пар
        import json
        with open(output_file, 'r', encoding='utf-8') as f:
            qa_data = json.load(f)
        
        print("\n📝 Примеры сгенерированных QA пар:")
        for i, qa in enumerate(qa_data[:3], 1):
            print(f"\n  Пример {i}:")
            print(f"    ❓ Вопрос: {qa.get('question', 'N/A')}")
            print(f"    ✅ Ответ: {qa.get('answer', 'N/A')[:100]}...")
            print(f"    🏷️  Категория: {qa.get('category', 'N/A')}")
            print(f"    📄 Источник: {qa.get('source_file', 'N/A')}")
        
        return output_file
    else:
        print("❌ Не удалось сгенерировать QA пары")
        return None


def test_3_clean_results(qa_file: str = None):
    """
    Тест 3: Очистка результатов от метаданных (из клетки 5 ноутбука).
    """
    print_separator("ТЕСТ 3: Очистка результатов от метаданных")
    
    # Запрашиваем файл с QA парами
    if qa_file is None:
        # qa_file = input("Введите путь к JSON файлу с QA парами (или Enter для пропуска): ").strip()
        qa_file = ""  # Будет пропущено если не указано
    
    if not qa_file or not os.path.exists(qa_file):
        print("⚠ Файл не указан или не существует. Пропускаем тест очистки.")
        return
    
    print(f"📄 Входной файл: {qa_file}")
    
    # Показываем, какие ключи будут удалены
    remove_keys = ['source_file']
    print(f"🗑️  Будут удалены ключи: {remove_keys}")
    
    # Создаем резервную копию
    backup_file = qa_file + ".backup"
    import shutil
    shutil.copy2(qa_file, backup_file)
    print(f"💾 Создана резервная копия: {backup_file}")
    
    # Показываем данные до очистки
    import json
    with open(qa_file, 'r', encoding='utf-8') as f:
        data_before = json.load(f)
    
    print(f"\n📊 До очистки:")
    if data_before and isinstance(data_before[0], dict):
        print(f"  Ключи в первой записи: {list(data_before[0].keys())}")
    
    # Очищаем результаты
    print("\n⏳ Очистка данных...")
    num_cleaned = QAGeneratorAI.clean_qa_results(
        input_file=qa_file,
        output_file=qa_file,
        remove_keys=remove_keys
    )
    
    if num_cleaned > 0:
        print(f"\n✅ УСПЕШНО! Обработано записей: {num_cleaned}")
        
        # Показываем данные после очистки
        with open(qa_file, 'r', encoding='utf-8') as f:
            data_after = json.load(f)
        
        print(f"\n📊 После очистки:")
        if data_after and isinstance(data_after[0], dict):
            print(f"  Ключи в первой записи: {list(data_after[0].keys())}")
        
        # Показываем пример
        print("\n📝 Пример очищенной записи:")
        if data_after:
            example = data_after[0]
            for key, value in example.items():
                if isinstance(value, str) and len(value) > 100:
                    value = value[:100] + "..."
                print(f"  {key}: {value}")
        
        print(f"\n💡 Резервная копия сохранена в: {backup_file}")
    else:
        print("❌ Не удалось очистить данные")


def main():
    """
    Главная функция: запускает все тесты последовательно.
    """
    print("""
╔════════════════════════════════════════════════════════════════════════════════╗
║                      ТЕСТИРОВАНИЕ QAGeneratorAI                                ║
║                                                                                ║
║  Проверка трех основных функциональностей:                                     ║
║  1. Разбиение PDF на части (клетка 3 из ноутбука)                              ║
║  2. Генерация QA пар через Gemini API (клетка 2 из ноутбука)                   ║
║  3. Очистка результатов от метаданных (клетка 5 из ноутбука)                   ║
╚════════════════════════════════════════════════════════════════════════════════╝
    """)
    
    # Тест 1: Разбиение PDF
    split_dir = test_1_pdf_splitting()
    
    # Тест 2: Генерация QA
    qa_file = test_2_qa_generation(pdf_dir=split_dir)
    
    # Тест 3: Очистка результатов
    test_3_clean_results(qa_file=qa_file)
    
    # Итоговая информация
    print_separator("ТЕСТИРОВАНИЕ ЗАВЕРШЕНО")
    print("\nПримечания:")
    print("  • Для полного теста необходим API ключ Google Gemini")
    print("  • Для разбиения PDF требуется библиотека PyPDF2")
    print("  • Для работы с Gemini требуется библиотека google-generativeai")
    print()


if __name__ == "__main__":
    main()
