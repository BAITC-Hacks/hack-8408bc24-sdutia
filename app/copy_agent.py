"""Bilingual copy for the agent control room."""

from app.i18n import tr as base_tr

COPY = {
    "policy": ("Режим агента", "Agent mode"),
    "policy_auto": ("Авто (LLM, если есть ключ)", "Auto (LLM when a key is available)"),
    "policy_llm": ("LLM", "LLM"),
    "policy_rules": ("Правила (без LLM)", "Rules (without LLM)"),
    "quick_first": ("31 янв 2026 — первый выпуск", "31 Jan 2026 — first issue"),
    "quick_february": ("10 фев 2026", "10 Feb 2026"),
    "quick_note": ("Быстрый выбор даты; затем нажмите «Запустить агента».", "Choose a date, then select Run agent."),
    "analysis_pending": ("Здесь появится результат вашего запуска.", "The result of your run will appear here."),
    "saved_example": ("Пример: сохранённый выпуск {date}", "Example: saved forecast issued on {date}"),
    "your_result": ("Результат вашего запуска · {date}", "Your run result · {date}"),
    "why_updated": ("Почему уточнено: {reason}", "Why updated: {reason}"),
    "details": ("Подробнее", "Details"),
    "original_analysis": ("Исходный текст анализа", "Original analysis text"),
    "original_language": ("Текст агента приведён на языке исходного выпуска.", "Agent narrative is shown in its original language; this issue has no English summary."),
    "additional_analysis": ("Дополнительные замечания", "Additional notes"),
    "general_analysis": ("Общий анализ выпуска", "Analysis of this issue"),
    "agent_decision": ("Решение агента", "Agent decision"),
    "waiting_result": ("Выполняется…", "Running…"),
    "unknown_step": ("Выполнение инструмента: {tool}", "Tool: {tool}"),
    "no_key": ("LLM-ключ не задан — агент работает по правилам. Используются те же инструменты и та же модель прогноза.", "No LLM key is configured — the agent uses rules, with the same tools and forecasting model."),
    "no_key_count": ("LLM-ключ не задан — агент работает по правилам: те же {count} инструментов и та же модель прогноза.", "No LLM key is configured — the agent uses rules: the same {count} tools and forecasting model."),
    "inspect_scada": ("Проверка данных SCADA", "Check SCADA data"),
    "list_weather_runs": ("Какие прогнозы погоды уже опубликованы", "Find weather forecasts already published"),
    "fetch_weather": ("Загрузка прогнозов погоды", "Load weather forecasts"),
    "validate_weather": ("Проверка качества погоды", "Check weather data quality"),
    "run_forecast": ("Расчёт ML-модели", "Run the forecasting model"),
    "analyze_forecast": ("Анализ результата", "Analyze the result"),
    "publish_forecast": ("Публикация прогноза", "Publish the forecast"),
    "check_for_updates": ("Проверка новых прогнозов погоды", "Check for newer weather forecasts"),
}


def tr(key: str, lang: str = "RU", **values) -> str:
    if key in COPY:
        return COPY[key][1 if lang == "EN" else 0].format(**values)
    return base_tr(key, lang, **values)


def tool_title(tool: str, lang: str) -> str:
    aliases = {"predict": "run_forecast", "publish": "publish_forecast", "check_new_runs": "check_for_updates"}
    key = aliases.get(tool, tool)
    return tr(key, lang) if key in COPY else tr("unknown_step", lang, tool=tool)
