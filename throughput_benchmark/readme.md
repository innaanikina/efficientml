Запуск бенчмарка:

Из корня репозитория:
`python -m throughput_benchmark.benchmark_matmul`

Параметры запуска задаются в `throughput_benchmark/benchmark_config.py`.
Для выбора реализации квантизованного умножения поменяйте:

```python
KERNEL_NAME = "rowwise_int4"
```

Результаты сохраняются в `throughput_benchmark/results/` с именем, включающим `KERNEL_NAME`.
