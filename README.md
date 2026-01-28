# LLM Agent Factory

Синтетический датасет AI-агентов для обучения моделей генерации агентов.

## Описание

Проект генерирует структурированные описания AI-агентов (AgentSpec) путём комбинирования **692 доменов** и **36 ролей**. Каждый агент содержит:
- `agent_id` — уникальный идентификатор
- `display_name` — человекочитаемое имя
- `persona` — характер и экспертиза агента (2-3 предложения)
- `description` — что делает агент и формат вывода
- `role_id` — роль агента
- `domain` — предметная область
- `tools` — список инструментов

## Структура проекта

```
LLM-Agent-Factory/
├── config/                      # Конфигурации
│   ├── domain.json              # 692 домена (Wikipedia-based)
│   ├── role_id.json             # 36 ролей агентов
│   └── tool.json                # 10 доступных инструментов
├── dataset/                     # Сгенерированные данные
│   ├── agents/                  # Сырые агенты (до обработки)
│   │   ├── agents_eng/          # EN, temperature 0.7
│   │   ├── agents_rus/          # RU, temperature 0.7
│   │   └── agents_temp_03_big/  # EN, temperature 0.3
│   ├── agents_norm/             # Нормализованные агенты (после curation)
│   │   ├── agents_eng/
│   │   ├── agents_rus/
│   │   └── agents_temp_03_big/
│   └── agent_tasks/             # Задачи для каждого агента
│       ├── agents_eng/
│       ├── agents_rus/
│       └── agents_temp_03_big/
├── script/                      # Скрипты генерации
│   ├── agent_generator.py       # Генерация агентов
│   ├── agent_deduplicator.py    # Дедупликация через эмбеддинги
│   ├── dataset_curator.py       # LLM-курация датасета
│   ├── add_meta_agents.py       # Добавление мета-агентов
│   └── task_generator.py        # Генерация задач
├── pyproject.toml
└── README.md
```

## Датасеты

| Датасет | Язык | Temperature | Агентов | Описание |
|---------|------|-------------|---------|----------|
| `agents_eng` | English | 0.7 | ~18000 | Основной англоязычный датасет |
| `agents_rus` | Russian | 0.7 | ~17000 | Русскоязычный датасет (persona, description, display_name) |
| `agents_temp_03_big` | English | 0.3 | ~19000 | Датасет с низкой температурой (более детерминированный) |

## Пайплайн генерации

```
1. agent_generator.py     → dataset/agents/           # Сырая генерация
2. agent_deduplicator.py  → (in-place)                # Дедупликация
3. dataset_curator.py     → dataset/agents_norm/      # LLM-курация
4. add_meta_agents.py     → (добавляет meta.json)     # Мета-агенты
5. task_generator.py      → dataset/agent_tasks/      # Генерация задач
```

### 1. Генерация агентов
Для каждой комбинации `domain × role` LLM решает:
- Логична ли комбинация (coder + Cooking = нет)
- Если да — генерирует 1-3 агента

### 2. Дедупликация
- Sentence Transformers (`all-mpnet-base-v2`)
- FAISS для поиска ближайших соседей
- Union-Find для кластеризации дубликатов
- Порог схожести: 0.87

### 3. LLM-курация
- Добавление недостающих специалистов
- Удаление семантических дубликатов
- Исправление tools (researcher → web_search, tutor → [])
- Улучшение persona и description

### 4. Мета-агенты
11 агентов для оркестрации мультиагентных систем:
- coordinator, router, summarizer, tool_runner
- verifier, safety_guard, memory_manager, planner
- evaluator, recovery_handler, state_keeper

### 5. Генерация задач
Для каждого агента генерируется 11 задач разной сложности:
- Простые вопросы новичка
- Сложный анализ
- Запросы на действие
- Творческие задачи
- Супер-сложные задачи

## Установка

```bash
# Основные зависимости
pip install -e .

# С поддержкой дедупликации (FAISS + sentence-transformers)
pip install -e ".[dedup]"
```

**Требования:** Python >= 3.11

## Формат данных

### AgentSpec (dataset/agents_norm/\*/\*.json)
```json
{
  "domain": "Music",
  "generated_at": "2025-01-27T10:00:00",
  "total_agents": 15,
  "agents": [
    {
      "agent_id": "music_theory_tutor",
      "display_name": "Music Theory Tutor",
      "persona": "A patient and methodical music educator...",
      "description": "Explains music theory concepts...",
      "role_id": "tutor",
      "domain": "Music",
      "tools": [],
      "input_schema": {},
      "output_schema": {},
      "raw": {}
    }
  ]
}
```

### Tasks (dataset/agent_tasks/\*/tasks.json)
```json
    {
      "agent_id": "air_sports_info_retriever",
      "display_name": "Air Sports Info Retriever",
      "domain": "Air sports",
      "role_id": "retriever",
      "tasks": [
        "What's the minimum age requirement to get a private pilot license for ultralight aircraft in the US? Why would I choose THIS agent over 1000 other AI specialists?",
        "Can you tell me the current wind conditions at the popular paragliding site in Birgit, Switzerland? Why would I choose THIS agent over 1000 other AI specialists?",
        "Compare the safety regulations for skydiving in Australia vs. Canada, focusing on altitude limits and mandatory equipment. Why would I choose THIS agent over 1000 other AI specialists?",
        "Find the next three international hot air balloon festivals, list their dates, locations, and any specific flight restrictions due to local airspace. Why would I choose THIS agent over 1000 other AI specialists?",
        "Book a weather briefing for my upcoming wingsuit jump in Patagonia on March 12, including wind speed, thermal activity, and precipitation forecast. Why would I choose THIS agent over 1000 other AI specialists?",
        "Generate a checklist with the latest FAA equipment specifications for a new powered paragliding wing and format it as a markdown table ready to copy. Why would I choose THIS agent over 1000 other AI specialists?",
        "I'm nervous about my first solo hang gliding flight tomorrow. Give me a short motivational speech that includes a quick safety tip about checking the launch site wind. Why would I choose THIS agent over 1000 other AI specialists?",
        "Write a short blog intro (150 words) announcing the upcoming European Aerobatic Championship, highlighting the unique aircraft types and the importance of weather monitoring. Why would I choose THIS agent over 1000 other AI specialists?",
        "My glider club is planning a cross-country contest from Lake Tahoe to Reno. I need a detailed route plan that avoids restricted airspace, includes optimal thermals based on forecast, and suggests refueling points. Why would I choose THIS agent over 1000 other AI specialists?",
        "We are organizing a skydiving training weekend in Texas and need a risk assessment report that includes local weather patterns for June, emergency landing zones within 30 miles, and compliance with FAA Part 105. Why would I choose THIS agent over 1000 other AI specialists?",
        "Perform a multi-step calculation to determine the optimal glide ratio for a new sailplane design under varying wind shear conditions forecasted for the upcoming competition in Chile, incorporating real-time atmospheric data, and compare it against the performance of the current world record holder. Provide the full computation steps and sources. Why would I choose THIS agent over 1000 other AI specialists?"
      ]
    },
```

## Роли агентов

| Роль | Описание | Инструменты |
|------|----------|-------------|
| `general` | Общий помощник | — |
| `researcher` | Исследователь | web_search |
| `coder` | Программист | code_interpreter |
| `tutor` | Обучает и объясняет | — |
| `advisor` | Даёт рекомендации | — |
| `critic` | Критический анализ | — |
| `fact_checker` | Проверка фактов | web_search |
| `summarizer` | Суммаризация | — |
| `translator` | Перевод | — |
| ... | (36 ролей всего) | |

## Инструменты

- `web_search` — поиск в интернете
- `code_interpreter` — выполнение кода
- `file_search` — поиск в документах
- `vector_search` — семантический поиск
- `image_generation` — генерация изображений
- `shell` — системные команды
- `computer_use` — взаимодействие с UI
- `apply_patch` — модификация кода
- `function_calling` — вызов внешних API
- `remote_mcp_servers` — внешние серверы инструментов

## Статистика

- **Доменов:** 692 (из Wikipedia categories)
- **Ролей:** 36
- **Агентов на датасет:** ~22000
- **Задач на агента:** 11
- **Всего агентов:** ~дохуя (3 датасета)
- **Всего задач:** ~еще больше чем дохуя
