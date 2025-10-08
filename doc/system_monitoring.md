# Мониторинг системных ресурсов

Документ описывает встроенный модуль мониторинга ресурсов, который завершает требования Этапа 1 к наблюдаемости и автоматическому восстановлению сервиса.

## Обзор
- Модуль реализован в `monitoring/resource_monitor.py` и запускается как фоновая задача FastAPI-приложения при старте. 【F:monitoring/resource_monitor.py†L1-L236】【F:src/main.py†L324-L370】
- Контролируются четыре ключевые метрики: общая загрузка CPU, использование оперативной памяти, RSS процесса приложения и заполненность диска. При превышении порогов отправляются алерты через существующий модуль уведомлений (`monitoring/alerts.py`). 【F:monitoring/resource_monitor.py†L104-L192】
- Для предотвращения ложных срабатываний используются серийные пороги (`warn_streak`, `crit_streak`), а также режим автоматического завершения процесса (`exit_on_crit`) для перезапуска супервизором. 【F:monitoring/resource_monitor.py†L40-L94】【F:monitoring/resource_monitor.py†L167-L192】

## Метрики и пороги по умолчанию
| Метрика | Порог предупреждения | Критический порог | Примечание |
| --- | --- | --- | --- |
| CPU (общая загрузка) | 85 % | 95 % | `RESOURCE_MONITOR_CPU_WARN` / `RESOURCE_MONITOR_CPU_CRIT` |
| RAM (системная) | 85 % | 95 % | `RESOURCE_MONITOR_RAM_WARN` / `RESOURCE_MONITOR_RAM_CRIT` |
| RSS процесса | 2048 MB | 3072 MB | `RESOURCE_MONITOR_RSS_WARN_MB` / `RESOURCE_MONITOR_RSS_CRIT_MB` |
| Диск (путь `RESOURCE_MONITOR_DISK_PATH`) | 90 % | 97 % | `RESOURCE_MONITOR_DISK_WARN` / `RESOURCE_MONITOR_DISK_CRIT` |

Дополнительно собираются: загрузка CPU процесса, swap, средняя нагрузка за 1 минуту, число потоков. Эти значения включаются в контекст алертов и heartbeat-сообщений. 【F:monitoring/resource_monitor.py†L96-L166】

## Поведение
1. **Heartbeat:** в состоянии «OK» модуль отправляет информационное сообщение каждые `RESOURCE_MONITOR_INFO_INTERVAL_SEC` секунд (по умолчанию 600 с), что позволяет контролировать живость фоновой задачи. 【F:monitoring/resource_monitor.py†L194-L214】
2. **Предупреждение:** при достижении порога `warn` подряд `warn_streak` раз (по умолчанию 2) отправляется предупреждение и пишется запись в лог. 【F:monitoring/resource_monitor.py†L167-L192】
3. **Критика:** при достижении порога `crit` подряд `crit_streak` раз (по умолчанию 3) отправляется критический алерт; если `exit_on_crit=true`, приложение завершает работу с кодом 90, чтобы внешняя система перезапустила контейнер/процесс. 【F:monitoring/resource_monitor.py†L167-L192】
4. **Восстановление:** когда показатели возвращаются в норму `recovery_streak` раз подряд, отправляется информационное уведомление о восстановлении. 【F:monitoring/resource_monitor.py†L167-L192】

## Конфигурация
Все параметры настраиваются переменными окружения (или `.env`), значения по умолчанию указаны в скобках:

- `RESOURCE_MONITOR_ENABLED` (`true`) – включает/отключает модуль.
- `RESOURCE_MONITOR_INTERVAL_SEC` (`30`) – интервал опроса метрик.
- `RESOURCE_MONITOR_INFO_INTERVAL_SEC` (`600`) – частота heartbeat-сообщений в состоянии OK.
- `RESOURCE_MONITOR_WARN_STREAK` (`2`) – сколько подряд измерений должно превысить порог `warn`.
- `RESOURCE_MONITOR_CRIT_STREAK` (`3`) – сколько подряд измерений должно превысить порог `crit`.
- `RESOURCE_MONITOR_RECOVERY_STREAK` (`2`) – количество измерений ниже порога для фиксации восстановления.
- `RESOURCE_MONITOR_EXIT_ON_CRIT` (`false`) – завершать ли процесс после критического события.
- `RESOURCE_MONITOR_DISK_PATH` (`/`) – путь, по которому оценивается заполненность диска.
- `RESOURCE_MONITOR_CPU_WARN` / `RESOURCE_MONITOR_CPU_CRIT` – пороги CPU, %.
- `RESOURCE_MONITOR_RAM_WARN` / `RESOURCE_MONITOR_RAM_CRIT` – пороги RAM, %.
- `RESOURCE_MONITOR_RSS_WARN_MB` / `RESOURCE_MONITOR_RSS_CRIT_MB` – пороги RSS процесса, MB.
- `RESOURCE_MONITOR_DISK_WARN` / `RESOURCE_MONITOR_DISK_CRIT` – пороги заполненности диска, %.

## Интеграция с алертингом
Модуль использует `alert_info`, `alert_warn`, `alert_crit` из `monitoring/alerts.py`; поэтому достаточно настроить Telegram-бота, SMTP или оба канала в `configs/exec.yaml` либо переменных окружения, чтобы получать уведомления. 【F:monitoring/alerts.py†L188-L366】

## Проверка
1. Убедиться, что при старте приложения в логах появляется сообщение `Resource monitor started`.
2. Вызвать нагрузку (например, заполнить диск в тестовой среде) и проверить отправку предупреждения/критики.
3. После снятия нагрузки монитор должен зафиксировать восстановление и отправить heartbeat в течение ближайших `info_interval_sec` секунд.

Модуль является частью базового self-check контура и должен оставаться включённым во всех окружениях, кроме unit-тестов и локальной отладки.
