# UAV Flight Telemetry Analyzer

**Команда:** TheDivass
**Версiя:** 1.8
**Завдання:** BEST - Система аналiзу телеметрiї та 3D-вiзуалiзацiї польотiв БПЛА

**TEST IT:** [https://uav-telemetry-analyzer.streamlit.app/](https://uav-telemetry-analyzer-ca5vg3dsvdb4ld4595ldxh.streamlit.app/)

---

## Опис

Веб-додаток для автоматизованого розбору бiнарних лог-файлiв польотних контролерiв Ardupilot (.BIN), обчислення кiнематичних метрик та iнтерактивної 3D-вiзуалiзацiї траєкторiї польоту.

### Можливостi

- **Парсинг телеметрiї** - зчитування GPS, IMU, BAT повiдомлень з .BIN файлiв через pymavlink. Автоматичний fallback на AHR2 (EKF) при вiдсутностi GPS та на ACC+GYR при вiдсутностi IMU.
- **Обчислення метрик** - загальна дистанцiя (Haversine), макс. горизонтальна/вертикальна швидкiсть, макс. прискорення, набiр висоти, тривалiсть польоту. Трапецiєвидне iнтегрування прискорень IMU для демонстрацiї дрейфу.
- **3D-вiзуалiзацiя (ENU)** - iнтерактивний 3D-графiк траєкторiї з конвертацiєю WGS-84 → ENU, динамiчним колоруванням за швидкiстю або часом, маркерами старту/фiнiшу, SRTM-рельєфом мiсцевостi.
- **Профiль швидкостi та висоти** - часовi графiки горизонтальної/вертикальної швидкостi з адаптивним масштабуванням.
- **GPS vs IMU порiвняння** - наочна демонстрацiя дрейфу при подвiйному iнтегруваннi акселерометра.
- **AI-звiт (експертна дiагностика)** - аналiз польоту у форматi ArduPilot-дiагноста: Root Cause Analysis (метод "5 Чому"), рiвень ризику (NOMINAL/WARNING/CRITICAL), Action Plan. Google Gemini API з fallback на gemini-2.0-flash та шаблонний звiт.
- **Детекцiя аномалiй** - евристичне виявлення рiзких втрат висоти, перевищення швидкостi, високих прискорень.
- **2D-карта траєкторiї** - iнтерактивна карта Folium/OpenStreetMap з маркерами старту/фiнiшу та кольоровими фазами польоту.
- **Автодетекцiя фаз польоту** - евристичне визначення злiт/круїз/зависання/посадка на основi швидкостей та висоти.
- **Монiторинг батареї** - графiк напруги та струму батареї у часi.
- **Якiсть GPS** - графiк HDOP та кiлькостi супутникiв у часi для оцiнки надiйностi позицiонування.
- **Профiль прискорення** - вiзуалiзацiя динамiчних навантажень на БПЛА з порогом P95.
- **Порiвняння польотiв** - паралельне порiвняння метрик двох рiзних польотiв.
- **Експорт PDF** - завантаження повного звiту у форматi PDF.
- **Налаштування** - одиницi швидкостi (км/год, м/с), колiрна схема, згладжування, фiльтрацiя за часовим дiапазоном, експорт метрик у CSV.

---

## Технологiчний стек

| Компонент | Технологiя | Обгрунтування |
|-----------|-----------|---------------|
| Парсинг .BIN | **pymavlink** | Офiцiйна бiблiотека MAVLink/Ardupilot, пiдтримує всi типи повiдомлень |
| Обробка даних | **pandas, numpy** | Стандарт для табличних даних та числових обчислень |
| Веб-iнтерфейс | **Streamlit** | Мiнiмальний boilerplate, вбудованi вiджети, iнтерактивнiсть |
| 3D-графiки | **Plotly** | Iнтерактивнi 3D Scatter + Surface, hover-пiдказки, zoom/rotate |
| Рельєф | **srtm.py, scipy** | Безкоштовнi NASA SRTM данi, iнтерполяцiя висот |
| 2D-карта | **Folium, streamlit-folium** | OpenStreetMap, маркери, фази польоту на реальнiй картi |
| AI-аналiз | **Google Gemini API** | Безкоштовний тариф, fallback 2.5-flash → 2.0-flash → шаблон |
| PDF-звiт | **fpdf2** | Генерацiя PDF з кирилицею (DejaVu шрифти) |
| Тестування | **pytest** | Юнiт-тести для coordinates, metrics, flight phases |
| Контейнеризацiя | **Docker** | Вiдтворюване середовище, простий деплой |

---

## Архiтектура

```
app.py                  # Streamlit entry point
src/
  __init__.py           # Публiчний API пакету (__all__)
  parser.py             # .BIN → GPS + IMU DataFrames (fallbacks: AHR2, ACC+GYR)
  coordinates.py        # Haversine, WGS-84 → ENU конвертацiя
  metrics.py            # Трапецiєвидне iнтегрування, метрики, детекцiя фаз польоту
  visualization.py      # 3D Plotly, SRTM terrain, 2D Folium карта, GPS якiсть, прискорення
  ai_report.py          # Gemini API + шаблонний звiт, детекцiя аномалiй, PDF-експорт
assets/
  DejaVuSans*.ttf       # Шрифти для PDF-генерацiї (Unicode/кирилиця)
docs/
  theory.md             # Математичне обгрунтування (WGS-84, ENU, Haversine, кватернiони)
tests/
  test_coordinates.py   # Тести Haversine, WGS-84 → ENU
  test_metrics.py       # Тести iнтегрування, метрик, фаз польоту
```

---

## Запуск

### Локально

```bash
# 1. Клонувати репозиторiй
git clone https://github.com/Vatsonio/uav-telemetry-analyzer
cd uav-telemetry-analyzer

# 2. Створити вiртуальне середовище
python -m venv venv
source venv/bin/activate  # Linux/macOS
# або: venv\Scripts\activate  # Windows

# 3. Встановити залежностi
pip install -r requirements.txt

# 4. Запустити додаток
streamlit run app.py
```

Додаток вiдкриється на http://localhost:8501

### Docker

```bash
docker compose up --build
```

### Gemini API (опцiонально)

Для AI-звiту введiть API ключ в sidebar або задайте змiнну середовища:

```bash
export GEMINI_API_KEY=your_key_here
```

Безкоштовний ключ: [https://aistudio.google.com/apikey](https://aistudio.google.com/app/api-keys)

---

## Вхiднi данi

Додаток працює з бiнарними .BIN лог-файлами Ardupilot. Покладiть файли в кореневу директорiю проєкту або завантажте через iнтерфейс.

---

## Вiдповiднiсть вимогам ТЗ

### MVP (40%)

| Вимога | Статус | Реалiзацiя |
|--------|--------|------------|
| Парсинг .BIN логiв Ardupilot (GPS, IMU) | ✅ | `src/parser.py` - pymavlink, автоматичнi fallbacks (AHR2, ACC+GYR) |
| Частоти семплювання та одиницi вимiрювань | ✅ | `src/parser.py:_compute_info()` - GPS/IMU Hz, одиницi в metadata |
| Структурований DataFrame | ✅ | pandas DataFrame з колонками time_s, lat, lng, alt, speed, vz, acc_x/y/z |
| Макс. горизонтальна/вертикальна швидкiсть | ✅ | `src/metrics.py:compute_flight_metrics()` з фiльтрацiєю викидiв (P99) |
| Макс. прискорення | ✅ | Повний вектор прискорення мiнус гравiтацiя |
| Макс. набiр висоти | ✅ | `alt.max() - alt[0]` |
| Тривалiсть польоту | ✅ | З GPS timestamp |
| Дистанцiя через Haversine | ✅ | `src/coordinates.py:haversine()` - числово стабiльна формула |
| Швидкостi з IMU через трапецiєвидне iнтегрування | ✅ | `src/metrics.py:trapezoidal_integrate()` + `velocity_from_imu()` |
| WGS-84 → ENU конвертацiя | ✅ | `src/coordinates.py:wgs84_to_enu()` - лiнеаризована + ECEF задокументовано |
| 3D-вiзуалiзацiя з колоруванням за швидкiстю/часом | ✅ | `src/visualization.py` - Plotly 3D Scatter, 6 colorscale |

### Алгоритмiчна база (20%)

| Вимога | Статус | Реалiзацiя |
|--------|--------|------------|
| Haversine formula | ✅ | `src/coordinates.py:haversine()` |
| Трапецiєвидне iнтегрування | ✅ | `src/metrics.py:trapezoidal_integrate()` |
| WGS-84 / ENU системи координат | ✅ | `src/coordinates.py:wgs84_to_enu()` + `docs/theory.md` |
| Теоретичне обгрунтування | ✅ | `docs/theory.md` - WGS-84, ENU, Haversine vs Vincenty, кватернiони, дрейф IMU, фiльтрацiя викидiв, детекцiя фаз |

### Nice-to-have (15%)

| Вимога | Статус | Реалiзацiя |
|--------|--------|------------|
| Веб-застосунок (Streamlit) | ✅ | `app.py` - завантаження файлiв, налаштування, iнтерактивнi графiки |
| AI-асистент (LLM) | ✅ | `src/ai_report.py` - Gemini API, Root Cause Analysis, Risk Level, Action Plan |
| Детекцiя аномалiй | ✅ | Рiзка втрата висоти, перевищення швидкостi, високе прискорення |
| 2D-карта траєкторiї | ✅ | Folium/OpenStreetMap з фазами польоту |
| Автодетекцiя фаз польоту | ✅ | Злiт / круїз / зависання / посадка |
| Монiторинг батареї | ✅ | Графiк напруги та струму |
| SRTM рельєф мiсцевостi | ✅ | NASA SRTM данi пiд 3D траєкторiєю |
| GPS vs IMU порiвняння | ✅ | Демонстрацiя дрейфу iнтегрування |
| Якiсть GPS (HDOP/супутники) | ✅ | Графiк HDOP та NSats у часi |
| Профiль прискорення | ✅ | Модуль прискорення без гравiтацiї, порiг P95 |
| Порiвняння двох польотiв | ✅ | Таблиця порiвняння метрик |
| Docker | ✅ | `Dockerfile` + `docker-compose.yml` |
| Юнiт-тести | ✅ | 22 тести (pytest): coordinates, metrics, flight phases |
| Експорт метрик CSV | ✅ | Кнопка завантаження в sidebar |
| Експорт звiту PDF | ✅ | PDF з метриками, графiками, AI-звiтом (fpdf2 + DejaVu), компактний layout |

### Архiтектура та чистота коду (10%)

| Вимога | Статус | Реалiзацiя |
|--------|--------|------------|
| Модульна архiтектура | ✅ | 5 модулiв з чiтким розподiлом вiдповiдальностi (SRP) |
| Документованi публiчнi API | ✅ | Docstrings (NumPy-style) для всiх публiчних функцiй |
| Type hints | ✅ | Типiзацiя параметрiв та повернених значень |
| `__init__.py` з експортами | ✅ | Явний `__all__` для контрольованого API пакету |
| Fallback-логiка | ✅ | GPS→AHR2, IMU→ACC+GYR, Gemini→шаблон |
| Фiльтрацiя викидiв | ✅ | P99/P95 перцентилi для GPS-глiтчiв |
| Кешування даних | ✅ | `st.session_state` для уникнення повторного парсингу |

### Документацiя / Презентацiя (15%)

| Вимога | Статус | Реалiзацiя |
|--------|--------|------------|
| README з iнструкцiєю запуску | ✅ | Локальний запуск + Docker |
| Обгрунтування вибору стеку | ✅ | Таблиця технологiй з поясненнями |
| Теоретичне обгрунтування | ✅ | `docs/theory.md` - WGS-84, ENU, Haversine, кватернiони, дрейф IMU |
| Таблиця вiдповiдностi ТЗ | ✅ | Повна таблиця з посиланнями на код |

---

## Тестування

```bash
python -m pytest tests/ -v
```

22 юнiт-тести: Haversine, WGS-84→ENU, трапецiєвидне iнтегрування, метрики польоту, детекцiя фаз.

---

## Алгоритмiчна база

Детальне математичне обгрунтування - у [`docs/theory.md`](docs/theory.md):

- **WGS-84 → ENU** - лiнеаризоване перетворення через дотичну площину (для малих дистанцiй <10 км), повний ECEF варiант задокументовано
- **Haversine vs Vincenty** - обчислення дистанцiї по великому колу, порiвняння сферичної та елiпсоїдної моделi
- **Трапецiєвидне iнтегрування** - чисельне iнтегрування прискорень IMU у швидкiсть з аналiзом дрейфу
- **Кватернiони vs Ейлер** - пояснення gimbal lock та переваг кватернiонного представлення орiєнтацiї
- **Фiльтрацiя викидiв (P99/P95)** - перцентильне обмеження для GPS-глiтчiв
- **Детекцiя фаз польоту** - евристичний алгоритм з 7 крокiв

## Скріншоти

<img src="https://media.discordapp.net/attachments/1151832689620557876/1490451174392991986/screencapture-uav-telemetry-analyzer-ca5vg3dsvdb4ld4595ldxh-streamlit-app-2026-04-05-23_40_24.png?ex=69d41a53&is=69d2c8d3&hm=bf3751b74505c195ff1ef0dea580f3aa5de4b08984f9402d9b4644401b00d4d9" width="100%" alt="UAV Telemetry Analyzer v1.4">

## Ліцензія

MIT
