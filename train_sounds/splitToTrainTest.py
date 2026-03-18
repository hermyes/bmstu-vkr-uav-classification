#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import csv
import shutil
import argparse
import random
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Set, Any
from collections import defaultdict

try:
    import soundfile as sf
except ImportError:
    print("Ошибка: требуется библиотека 'soundfile'. Установите её с помощью: pip install soundfile")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


LABEL_MAP = {
    "background": 0,
    "drone": 1,
    "helicopter": 2,
    "airplane": 3,
}


def get_source_id(filepath: Path) -> str:
    """
    Извлекает source_id (идентификатор источника) из имени файла.
    Правило: берется все до последнего символа '_' (перед индексом сегмента).
    Примеры:
      heli_ABC123_0001.wav -> heli_ABC123
      drone_XY77_0003.wav  -> drone_XY77
      plainname.wav        -> plainname
    """
    stem = filepath.stem
    if "_" in stem:
        return stem.rsplit("_", 1)[0]
    return stem


def clean_class_name(dir_name: str) -> str:
    """
    Очищает имя директории, удаляя суффикс '_train_1sec'.
    Пример: 'airplane_train_1sec' -> 'airplane'
    """
    suffix = "_train_1sec"
    if dir_name.endswith(suffix):
        return dir_name[: -len(suffix)]
    return dir_name


def normalize_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> Tuple[float, float, float]:
    """
    Нормализует доли выборок, чтобы они в сумме давали 1.0.
    """
    total = train_ratio + val_ratio + test_ratio
    if total <= 0:
        raise ValueError("Сумма долей должна быть положительным числом.")
    if not (0.99 <= total <= 1.01):
        logging.warning(f"Сумма долей равна {total:.4f}, нормализуем к 1.0")
    return train_ratio / total, val_ratio / total, test_ratio / total


def ensure_min_splits(source_ids: List[str], train_r: float, val_r: float) -> Tuple[List[str], List[str], List[str]]:
    """
    Разделяет списки идентификаторов на train/val/test с защитой от утечек данных.
    Гарантированно оставляет хотя бы один элемент в val/test, если их достаточно.
    
    - Если n == 1: только train
    - Если n == 2: train + test
    - Если n >= 3: пытается выделить минимум 1 в val и 1 в test
    """
    n = len(source_ids)
    if n == 0:
        return [], [], []

    if n == 1:
        return source_ids, [], []
    if n == 2:
        return [source_ids[0]], [], [source_ids[1]]

    n_train = max(1, int(round(n * train_r)))
    n_val = max(1, int(round(n * val_r)))

    if n_train + n_val >= n:
        n_val = max(1, n - n_train - 1)

    if n_train + n_val >= n:
        n_train = max(1, n - n_val - 1)

    train_ids = source_ids[:n_train]
    val_ids = source_ids[n_train : n_train + n_val]
    test_ids = source_ids[n_train + n_val :]
    return train_ids, val_ids, test_ids


def safe_link_or_copy(src: Path, dst: Path, symlink: bool) -> None:
    """
    Проверяет наличие файла назначения и создает символическую ссылку (если symlink=True)
    или копирует файл, избегая перезаписи. Создает все нужные поддиректории.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if symlink:
        try:
            os.symlink(src.resolve(), dst)
        except OSError as e:
            raise RuntimeError(f"Не удалось создать символическую ссылку для {src} -> {dst}: {e}")
    else:
        shutil.copy2(src, dst)


def write_json(path: Path, obj: Any) -> None:
    """
    Утилита для создания папки и записи данных (объекта) в JSON по указанному пути.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def process_dataset(args: argparse.Namespace) -> None:
    """
    Основной воркер: обрабатывает входную директорию, группирует файлы по классам 
    и source_id, разбивает на train/val/test без утечек, копирует/линкует файлы 
    и генерирует общие метаданные для каждого разбиения.
    """
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists() or not input_dir.is_dir():
        logging.error(f"Входная директория не существует: {input_dir}")
        sys.exit(1)

    train_r, val_r, test_r = normalize_ratios(args.train_ratio, args.val_ratio, args.test_ratio)
    random.seed(args.seed)

    dataset: Dict[str, Dict[str, List[Path]]] = defaultdict(lambda: defaultdict(list))
    found_class_dirs: List[Path] = []

    logging.info(f"Сканирование входной директории: {input_dir}")
    for class_dir in sorted(input_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        if not class_dir.name.endswith("_train_1sec"):
            continue

        class_name = clean_class_name(class_dir.name)
        found_class_dirs.append(class_dir)

        files = sorted(class_dir.glob(f"*{args.ext}"))
        for f in files:
            src_id = get_source_id(f)
            dataset[class_name][src_id].append(f)

    if not dataset:
        logging.error(f"Аудиофайлы с расширением {args.ext} не найдены в папках '*_train_1sec' в {input_dir}")
        sys.exit(1)

    write_json(output_dir / "label_map.json", LABEL_MAP)

    splits: Dict[str, List[Tuple[str, str, Path]]] = {"train": [], "val": [], "test": []}

    classes_for_stats = sorted(set(list(dataset.keys()) + list(LABEL_MAP.keys())))

    for class_name, sources in dataset.items():
        source_ids = sorted(list(sources.keys()))
        random.shuffle(source_ids)

        train_ids, val_ids, test_ids = ensure_min_splits(source_ids, train_r, val_r)

        for sid in train_ids:
            splits["train"].extend([(class_name, sid, f) for f in sources[sid]])
        for sid in val_ids:
            splits["val"].extend([(class_name, sid, f) for f in sources[sid]])
        for sid in test_ids:
            splits["test"].extend([(class_name, sid, f) for f in sources[sid]])

    stats_files: Dict[str, Dict[str, int]] = {s: defaultdict(int) for s in splits.keys()}
    stats_sources: Dict[str, Dict[str, Set[str]]] = {s: defaultdict(set) for s in splits.keys()}

    output_dir.mkdir(parents=True, exist_ok=True)

    for split_name in ["train", "val", "test"]:
        split_items = splits[split_name]
        split_dir = output_dir / split_name
        audio_dir = split_dir / "audio"
        audio_dir.mkdir(parents=True, exist_ok=True)

        manifest: List[Dict[str, Any]] = []
        logging.info(f"Обработка разбиения {split_name} ({len(split_items)} файлов)...")

        for idx, (class_name, source_id, src_path) in enumerate(split_items):
            fname = f"{idx:06d}{args.ext}"
            dst_path = audio_dir / fname

            try:
                safe_link_or_copy(src_path, dst_path, args.symlink)
            except RuntimeError as e:
                logging.error(str(e))
                sys.exit(1)

            duration_sec = 0.0
            sample_rate = 0
            try:
                info = sf.info(str(src_path))
                duration_sec = float(info.duration)
                sample_rate = int(info.samplerate)
            except Exception as e:
                logging.warning(f"Не удалось считать информацию об аудио из {src_path}: {e}")

            if class_name not in LABEL_MAP:
                logging.warning(
                    f"Класс '{class_name}' отсутствует в LABEL_MAP. "
                    f"Будет установлен label_id=-1. Рассмотрите возможность добавления его в LABEL_MAP."
                )
                label_id = -1
            else:
                label_id = LABEL_MAP[class_name]

            manifest.append(
                {
                    "id": idx,
                    "file": f"audio/{fname}",
                    "label": class_name,
                    "label_id": label_id,
                    "source_id": source_id,
                    "duration_sec": duration_sec,
                    "sample_rate": sample_rate,
                    "orig_path": str(src_path.as_posix()),
                }
            )

            stats_files[split_name][class_name] += 1
            stats_sources[split_name][class_name].add(source_id)

        write_json(split_dir / "manifest.json", manifest)

    print("\n" + "=" * 60)
    print("СТАТИСТИКА РАЗБИЕНИЯ ДАТАСЕТА (файлы + уникальные source_id)")
    print("=" * 60)

    for split_name in ["train", "val", "test"]:
        print(f"\nРазбиение: {split_name.upper()}")
        print(f"{'Класс':<15} | {'Файлы':<10} | {'Уникальные источники':<15}")
        print("-" * 50)

        for cls in classes_for_stats:
            n_files = stats_files[split_name].get(cls, 0)
            n_sources = len(stats_sources[split_name].get(cls, set()))
            print(f"{cls:<15} | {n_files:<10} | {n_sources:<15}")

            if split_name in ["val", "test"] and n_files == 0 and cls in dataset:
                logging.warning(f"Класс '{cls}' ПУСТ в разбиении {split_name}!")

    print("\nГотово!")
    print(f"Результат записан в: {output_dir}")
    print(f"Таблица классов: {output_dir / 'label_map.json'}")
    print(f"Манифест train: {output_dir / 'train' / 'manifest.json'}")
    print(f"Манифест val:   {output_dir / 'val' / 'manifest.json'}")
    print(f"Манифест test:  {output_dir / 'test' / 'manifest.json'}")


def build_argparser() -> argparse.ArgumentParser:
    """
    Создает и возвращает парсер аргументов командной строки.
    """
    p = argparse.ArgumentParser(
        description="Разделение аудиодатасета на train/val/test (без утечек по source_id) "
                    "и экспорт JSON манифестов со всеми аудио-файлами для каждого разбиения."
    )
    p.add_argument("--input_dir", type=str, default="TRAIN_SOUNDS", help="Входная директория с папками '*_train_1sec'.")
    p.add_argument("--output_dir", type=str, default="dataset_out", help="Выходная директория.")
    p.add_argument("--train_ratio", type=float, default=0.7, help="Доля обучающей выборки (train).")
    p.add_argument("--val_ratio", type=float, default=0.15, help="Доля валидационной выборки (val).")
    p.add_argument("--test_ratio", type=float, default=0.15, help="Доля тестовой выборки (test).")
    p.add_argument("--seed", type=int, default=42, help="Случайный посев (для воспроизводимости).")
    p.add_argument("--symlink", action="store_true", help="Создавать символические ссылки вместо копирования.")
    p.add_argument("--ext", type=str, default=".wav", help="Расширение аудиофайлов (по умолчанию .wav).")
    return p


if __name__ == "__main__":
    parser = build_argparser()
    args = parser.parse_args()
    process_dataset(args)