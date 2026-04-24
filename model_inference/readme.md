# Week 4. Инференс квантизованной LLaMA

Запуск из корня репозитория

```bash
python -m model_inference.llama_quantized_inference
```

Скрипт выполняет следующее:

- Загружает `unsloth/Llama-3.2-1B-Instruct` в fp16 на CUDA;
- Заменяет слои `nn.Linear` на `QuantizedLinear`;
- Оставляет `lm_head` в fp16;
- Проверяет forward pass на тестовом промпте;
- Жадно генерирует короткую последовательность;
- Сохраняет отчет в `model_inference/results/llama_quantized_report.json`.

Финальная конфигурация модели находится в файле `model_inference/llama_inference_config.py`.
