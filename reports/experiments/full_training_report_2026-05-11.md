# Полноразмерное обучение и сравнение моделей

Дата: `2026-05-11`  
Датасет: `train_sounds/dataset_out`  
Объем: `train=6857`, `val=1467`, `test=1476`  
Признаки: `144`

## 1) Итог полноразмерного обучения SVM

Модель: `SVC (RBF) + calibrate probabilities`

- Лучшие параметры (по CV на train): `C=10.0`, `gamma=scale`, `kernel=rbf`
- `val`:
  - Accuracy: `0.9489`
  - F1-macro: `0.9469`
- `test`:
  - Accuracy: `0.9383`
  - F1-macro: `0.9318`

Артефакты:

- `models/svm_baseline.pkl`
- `models/scaler.pkl`
- `models/label_map.json`
- `models/feature_names.json`
- `reports/metrics/svm_val_metrics.json`
- `reports/metrics/test_metrics.json`
- `reports/figures/confusion_matrix.png`

Архивные копии метрик этого запуска:

- `reports/metrics/svm_val_metrics_full_2026-05-11.json`
- `reports/metrics/svm_test_metrics_full_2026-05-11.json`
- `reports/metrics/svm_training_summary_full_2026-05-11.json`

## 2) Сравнение моделей (полный датасет)

Отбор гиперпараметров выполнен по `val F1-macro`.

| Модель | Val F1-macro | Test F1-macro | Test Accuracy |
|---|---:|---:|---:|
| `gradient_boosting` | 0.9578 | 0.9312 | 0.9417 |
| `svm_rbf` | 0.9538 | 0.9309 | 0.9383 |
| `random_forest` | 0.9518 | 0.9416 | 0.9485 |
| `svm_linear` | 0.9261 | 0.8805 | 0.8997 |
| `logreg` | 0.9198 | 0.8810 | 0.8997 |
| `knn` | 0.8767 | 0.8886 | 0.9031 |

Вывод:

- Лучшая по `val` модель: `gradient_boosting`.
- Лучшая по `test` метрикам (`F1-macro` и `accuracy`) на этом прогоне: `random_forest`.
- SVM показывает стабильно высокий результат и остается хорошим baseline.

Артефакты сравнения:

- `reports/metrics/model_comparison.json`
- `reports/experiments/model_comparison.md`
- `models/comparison/*.pkl`

## 3) Практическая рекомендация

1. Для дальнейших экспериментов и воспроизводимости оставить baseline: `models/svm_baseline.pkl`.
2. Для лучшего качества на текущем test-пуле рассмотреть `models/comparison/random_forest.pkl` как кандидат production-модели.
3. Перед интеграцией с часовыми записями проверить поведение обеих моделей на длинном аудио в режиме оконного анализа.
