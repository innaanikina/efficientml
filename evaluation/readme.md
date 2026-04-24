# Эксперимент 4. Оценка качества на WikiText-2

Запуск из корня репозитория:

```bash
python -m evaluation.evaluate_wikitext2
```

Скрипт выполняет:

- загрузку test сплита датасета `WikiText-2`;
- оценку perplexity базовой fp16-модели `unsloth/Llama-3.2-1B-Instruct`;
- оценку perplexity модели с квантованными линейными слоями;
- замер скорости оценки в tokens/s;
- замер скорости генерации в tokens/s;
- сравнение качества и скорости базовой и квантованной моделей.

Результаты сохраняются в:

- `evaluation/results/wikitext2_eval.json`
- `evaluation/results/wikitext2_eval.md`

Параметры оценки задаются в `evaluation/wikitext2_config.py`.
