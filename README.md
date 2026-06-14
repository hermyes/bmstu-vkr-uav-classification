# BMSTU-VKR-UAV-Classification

Система классификации звуков летательных аппаратов (ВКР, МГТУ им. Н.Э. Баумана).

Проект реализует полный baseline-конвейер:

1. загрузка аудио и унификация формата;
2. предобработка и фильтрация сигнала;
3. сегментация и эвристический выбор информативного окна;
4. извлечение признаков (MFCC + спектральные + энергетические);
5. обучение и оценка SVM;
6. инференс по одному файлу и формирование JSON-результата.

## Классы

- `background` -> `0`
- `drone` -> `1`
- `helicopter` -> `2`
- `airplane` -> `3`

## Текущие данные

Основной датасет для обучения:

- `train_sounds/dataset_out`
  - `train/manifest.json`
  - `val/manifest.json`
  - `test/manifest.json`
  - `label_map.json`

Подробный снимок структуры проекта:  
- [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)

Журнал текущей разработки и следующего шага:  
- [DEVELOPMENT_LOG.md](DEVELOPMENT_LOG.md)

## Структура репозитория

```text
BMSTU-VKR-UAV-Classification/
├─ configs/
├─ data/
├─ models/
├─ reports/
├─ scripts/
├─ src/
├─ tests/
├─ train_sounds/
├─ DEVELOPMENT_LOG.md
├─ PROJECT_STRUCTURE.md
├─ README.md
└─ requirements.txt
```

## Установка

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Быстрый старт

### 1) Проверка датасета

```bash
python scripts/prepare_dataset.py --dataset train_sounds/dataset_out
```

### 2) Обучение baseline SVM

```bash
python scripts/train_svm.py --config configs/svm_baseline.yaml --dataset train_sounds/dataset_out
```

Артефакты:

- `models/svm_baseline.pkl`
- `models/scaler.pkl`
- `models/label_map.json`
- `reports/metrics/svm_val_metrics.json`

### 3) Оценка на test split

```bash
python scripts/evaluate.py --config configs/svm_baseline.yaml --dataset train_sounds/dataset_out --model models/svm_baseline.pkl --scaler models/scaler.pkl --label-map models/label_map.json
```

Артефакты:

- `reports/metrics/test_metrics.json`
- `reports/figures/confusion_matrix.png`

### 4) Инференс по одному файлу

```bash
python scripts/infer.py --audio path/to/file.wav --model models/svm_baseline.pkl --scaler models/scaler.pkl --label-map models/label_map.json --config configs/inference.yaml
```

Пример ответа:

```json
{
  "target_type": "drone",
  "label_id": 1,
  "raw_label_id": 1,
  "raw_label_name": "drone",
  "confidence": 0.87,
  "signal_level": 0.63,
  "time_start": 0.0,
  "time_end": 1.0,
  "selected_windows": 1,
  "decision": "accepted",
  "audio_path": "path/to/file.wav"
}
```

## Конфигурации

- `configs/svm_baseline.yaml`:
  - параметры датасета;
  - предобработка;
  - признаки;
  - grid-search для SVM;
  - выходные директории.
- `configs/inference.yaml`:
  - предобработка;
  - сегментация и эвристика;
  - порог уверенности (`confidence_threshold`).

## Тесты

```bash
pytest -q
```

Покрыты базовые проверки:

- загрузка и ресемплинг аудио;
- предобработка и сегментация;
- извлечение признаков;
- end-to-end inference pipeline на синтетическом сигнале.

## Презентационный Режим

Создание наглядного отчета по инференсу (графики + markdown):

```bash
python scripts/present_infer.py \
  --audio path/to/file.wav \
  --model models/svm_baseline.pkl \
  --scaler models/scaler.pkl \
  --label-map models/label_map.json \
  --config configs/inference.yaml
```

Подробности: `PRESENTATION_MODE.md`.

## Статистика Предположений Эвристики

`scripts/evaluate.py` также считает качество угадывания класса только эвристикой (без предсказания класса ML-моделью).

- включено по умолчанию в `evaluate`;
- сохраняется в:
  - `reports/metrics/test_metrics.json` (поле `heuristic_type_guess`);
  - `reports/metrics/heuristic_type_guess_metrics.json`.

При необходимости можно отключить:

```bash
python scripts/evaluate.py ... --skip-heuristic-type-stats
```

## Streamlit Приложение Для Презентации

Запуск интерактивного демо-интерфейса:

```bash
python -m streamlit run streamlit_app.py
```

Подробная инструкция:
- `STREAMLIT_DEMO.md`
