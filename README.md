### Запуск контейнера

Dockerfile совместим с драйверами cuda 12.3 или новее

В docker-compose.yaml можно пробросить желаемый порт (для triton-vis, например) или добавить команду запуска jupyter notebook

```bash
docker compose up -d
```


### Запуск тестов

```bash
pytest -sv
```
### Бенчмарки

Запуск бенчмарка для _quantize_rowwise_int4:

```bash
python triton_kernels/quantize_kernel.py
```