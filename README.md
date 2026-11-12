# Movie Recommender ML

Передайте `user_id`, чтобы получить список фильмов с названиями. Item-to-item модель считает косинусное сходство фильмов по разреженной матрице оценок MovieLens 1M. Запросы принимает FastAPI.

## Запуск в Docker

~~~sh
docker build -t movie-recommender-ml .
docker run --rm -p 127.0.0.1:8000:8000 movie-recommender-ml
~~~

При сборке Docker скачивает MovieLens 1M с сайта GroupLens, проверяет SHA-256 и обучает модель. Архив остаётся в промежуточном этапе сборки. В финальный образ копируется обученная модель. После запуска контейнера откройте [Swagger UI](http://localhost:8000/docs).

Чтобы получить три рекомендации для пользователя 6040, выполните запрос из другого терминала.

~~~sh
curl --fail-with-body "http://127.0.0.1:8000/recommendations?user_id=6040&k=3"
~~~

~~~powershell
curl.exe --fail-with-body "http://127.0.0.1:8000/recommendations?user_id=6040&k=3"
~~~

~~~json
{
  "user_id": 6040,
  "recommendations": [
    {"movie_id": 1221, "title": "Godfather: Part II, The (1974)"},
    {"movie_id": 1234, "title": "Sting, The (1973)"},
    {"movie_id": 1267, "title": "Manchurian Candidate, The (1962)"}
  ]
}
~~~

`k` задаёт размер выдачи от 1 до 50, по умолчанию 10. Сервис берёт кандидатов из обучающего каталога и исключает фильмы, оценённые пользователем в обучающей истории. Если в истории есть оценки и ни одна не положительная, сервис выдаёт популярные фильмы.

Для неизвестного `user_id` API отвечает HTTP 404. Через `GET /health` можно проверить, загружена ли модель.

## Офлайн-оценка

Для всех пользователей взяли одну дату разбиения. Профиль строится по оценкам от четырёх звёзд до этой даты. Для теста берём фильмы из обучающего каталога, которым пользователь поставил от четырёх звёзд после разбиения. Фильмы с оценками до разбиения исключаем. В оценку вошли 1 122 пользователя с такими фильмами.

| Метод | Recall@10 | NDCG@10 |
|---|---:|---:|
| Популярные фильмы | 0,054 | 0,227 |
| Item-to-item cosine | **0,062** | **0,257** |

Дата разбиения, размеры выборок, размер каталога и метрики есть в [отчёте](metrics.json).

## Повторить оценку и запустить тесты

В Windows запустите PowerShell с Python 3.13.

~~~powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe download_data.py
.\.venv\Scripts\python.exe evaluate.py --user-id 6040 --model-output artifacts/recommender.joblib
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m uvicorn api:app
~~~

На Linux и macOS используйте `python3.13 -m venv .venv` и `.venv/bin/python`.

Тесты работают на небольших синтетических данных. Они проверяют разбиение по времени, метрики, порядок выдачи, исключение уже оценённых фильмов и ответы API. Затем [GitHub Actions](.github/workflows/ci.yml) собирает контейнер на MovieLens 1M и выполняет запрос из README.

## Источник данных

Данные предоставила группа GroupLens Research. [MovieLens 1M](https://grouplens.org/datasets/movielens/1m/)

Описание датасетов F. Maxwell Harper и Joseph A. Konstan, 2015. [The MovieLens Datasets: History and Context](https://doi.org/10.1145/2827872)

Код и отчёт опубликованы для учебного использования. Данные и модель подготавливаются при локальной сборке. [Условия GroupLens](https://files.grouplens.org/datasets/movielens/ml-1m-README.txt)
