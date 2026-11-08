# Alarm suppression in a zero-waste zinc plant: simulation code and results

Code, parameter tables and results for the article «Подавление тревог-следствий в АСУ ТП безотходного цинкового производства с замкнутыми рециклами: сравнение правил по графу технологической схемы и по времени на имитационных данных» (Suppression of consequential alarms in a zero-waste zinc plant with closed recycles: graph-based versus time-based rules on simulated data).

The data are synthetic; no real plant data were used. Every model parameter is either taken from an open source or marked as an assumption (see `results/params_*.csv`).

## Состав
- `alarm_sim.py` — имитационная модель, варианты B0/B1/B2/P/P2, метрики M1–M8, Монте-Карло, анализ чувствительности.
- `extra_checks.py` — S9 при W = 180 мин, окно эталонной разметки (10/30/60 мин), определение времени распространения, представление N5 как запаздывания.
- `make_figures.py` — рисунки 1–3.
- `results/` — таблицы результатов (runs, summary, reductions, tests, montecarlo, sensitivity, extra_*, params_*), `graph_report.txt`, `config.json`.
- `figs/` — рисунки.

## Зависимости
Python 3.10+; `pip install -r requirements.txt` (numpy, pandas, networkx, scipy, matplotlib).

## Воспроизведение (зерно 12345, расчёт детерминирован)
```
python alarm_sim.py --runs 100 --mc 150 --mc-runs 3 --sens --sens-runs 30 --out results
python extra_checks.py results
python make_figures.py results figs
```
Время: основной эксперимент с Монте-Карло около 10 минут, чувствительность и дополнительные проверки — ещё 20–30 минут на одном ядре.

Повторный запуск с нуля даёт те же таблицы (`runs.csv`, `summary.csv`, `reductions.csv`, `tests.csv`, `priorities_B1.csv`, `montecarlo.csv`), что приведены в статье.

## Параметры
Параметры узлов, дуг и переменных с указанием источников или пометкой «допущение» — в `results/params_nodes.csv`, `params_edges.csv`, `params_tags.csv`.

## Лицензия
MIT.
