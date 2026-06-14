# Сравнение моделей классификации

Дата: 2026-05-11
Датасет: `train_sounds\dataset_out`

## Итоговая таблица

| Модель | Val F1-macro | Test F1-macro | Test Accuracy | Время (с) | Параметры |
|---|---:|---:|---:|---:|---|
| `gradient_boosting` | 0.9578 | 0.9312 | 0.9417 | 2893.3 | `{"n_estimators": 250, "learning_rate": 0.1, "max_depth": 3}` |
| `svm_rbf` | 0.9538 | 0.9309 | 0.9383 | 199.8 | `{"C": 5.0, "gamma": "scale"}` |
| `random_forest` | 0.9518 | 0.9416 | 0.9485 | 17.4 | `{"n_estimators": 300, "max_depth": null, "min_samples_leaf": 2}` |
| `svm_linear` | 0.9261 | 0.8805 | 0.8997 | 102.7 | `{"C": 1.0}` |
| `logreg` | 0.9198 | 0.8810 | 0.8997 | 1.4 | `{"C": 2.0}` |
| `knn` | 0.8767 | 0.8886 | 0.9031 | 1.7 | `{"n_neighbors": 3, "weights": "distance", "p": 1}` |

## Лучшая модель по val

- Модель: `gradient_boosting`
- Val F1-macro: `0.9578`
- Test F1-macro: `0.9312`
- Test Accuracy: `0.9417`
- Путь к артефакту: `models\comparison\gradient_boosting.pkl`

## Примечание

- Модели сохраняются, чтобы не переобучать их повторно.
- Отбор гиперпараметров выполнен по `val` (метрика `F1-macro`).