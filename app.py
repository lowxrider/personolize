import streamlit as st
import pandas as pd
import sqlite3
import random
from faker import Faker
import openai

# ----------------------------- Конфигурация страницы -----------------------------
st.set_page_config(page_title="Сервис персонализированных e-mail рассылок", layout="wide")

# ----------------------------- CSS: перенос сайдбара вправо -----------------------
st.markdown(
    """
    <style>
        [data-testid=\"stSidebar\"] {order: 2;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------- Глобальные данные ------------------------
FAKER = Faker("ru_RU")
ROASTS = ["Светлая", "Средняя", "Тёмная"]
BREW_METHODS = ["Френч‑пресс", "Эспрессо", "Пуровер", "Аэропресс"]
PRODUCTS = [
    "Эфиопия Сидамо", "Кения АА", "Колумбия Супремо", "Гватемала Антигуа",
    "Бразилия Сантос", "Индонезия Ява", "Уганда Буген", "Перу Куско",
    "Ямайка Блу Маунтин", "Коста‑Рика Тарразу", "Мексика Чьяпас", "Вьетнам Робуста",
    "Панама Гейша", "Индия Мономзари", "Суматра Мандхелинг", "Никарагуа Финка",
    "Эквадор Чимборазо", "Гондурас Окоапо", "Бразилия Шоколадный микс", "Кения Киллдегум",
    "Кения Печикиа"
]
COFFEE_TYPES = ["Арабика", "Робуста", "Бленд"]
SEGMENTS = ["новый", "лояльный", "риск ухода"]

# ----------------------------- Функция генерации данных ---------------------------
def generate_coffee_customers(n: int) -> pd.DataFrame:
    rows = []
    for idx in range(1, n + 1):
        first = FAKER.first_name()
        last = FAKER.last_name()
        middle = FAKER.middle_name()
        birth_date = FAKER.date_of_birth(minimum_age=18, maximum_age=80)
        rows.append({
            "customer_id": idx,
            "last_name": last,
            "first_name": first,
            "middle_name": middle,
            "email": "stas.krdpltsv+test666@gmail.com",
            "birth_date": birth_date,
            "preferred_language": "RU",
            "segment": random.choice(SEGMENTS),
            "last_purchase_date": FAKER.date_between(start_date='-6M', end_date='today'),
            "favorite_roast": random.choice(ROASTS),
            "favorite_coffee_type": random.choice(COFFEE_TYPES),
            "favorite_product": random.choice(PRODUCTS),
            "last_order_product": random.choice(PRODUCTS),
            "preferred_brew_method": random.choice(BREW_METHODS),
            "lifetime_value": round(random.uniform(5000, 90000), 2),
        })
    return pd.DataFrame(rows)

# ----------------------------- Сайдбар ----------------------------------
with st.sidebar:
    st.header("Настройки LLM")
    api_key = st.text_input("Ключ OpenAI", type="password")
    if st.button("Сохранить ключ") and api_key:
        st.session_state["OPENAI_API_KEY"] = api_key
    if st.button("Очистить ключ"):
        st.session_state.pop("OPENAI_API_KEY", None)
    model = st.selectbox("Модель", ["gpt-4o", "gpt-4", "gpt-3.5-turbo"])
    temperature = st.slider("Температура", 0.0, 1.0, value=0.7, step=0.05)
    top_p = st.slider("Креативность (top_p)", 0.0, 1.0, value=0.9, step=0.05)
    max_tokens = st.number_input("Макс. токенов", min_value=50, max_value=4096, value=512, step=50)
    st.session_state.update({
        "OPENAI_MODEL": model,
        "OPENAI_TEMPERATURE": temperature,
        "OPENAI_TOP_P": top_p,
        "OPENAI_MAX_TOKENS": max_tokens,
    })

# ----------------------------- Основные вкладки -----------------------------------
TAB_TITLES = ["Датасет CRM", "Генератор писем", "Библиотека промптов", "Метрики"]
tab1, tab2, tab3, tab4 = st.tabs(TAB_TITLES)

# ----------------------------- Вкладка 1: Датасет ---------------------------------
with tab1:
    st.subheader("Создание CRM-датасета")
    n = st.number_input("Число записей", 100, 10000, 1000, 100)
    if st.button("Сгенерировать датасет"):
        df = generate_coffee_customers(n)
        st.session_state["crm_df"] = df
        conn = sqlite3.connect("crm_seed.db")
        df.to_sql("customers", conn, if_exists="replace", index=False)
        conn.close()
        st.success(f"Сгенерировано {len(df)} записей")
    if "crm_df" in st.session_state:
        st.dataframe(st.session_state["crm_df"], use_container_width=True)
        st.download_button(
            "Скачать CSV",
            data=st.session_state["crm_df"].to_csv(index=False).encode("utf-8"),
            file_name="coffee_crm.csv"
        )

# ----------------------------- Вкладка 2: Генератор писем -----------------------------------
with tab2:
    st.subheader("Генерация персонализированного контента")
    # Поле для вывода ответа от модели
    response = st.text_area(
        label="Ответ модели",
        value=st.session_state.get("tab2_response", ""),
        height=200,
        key="tab2_response"
    )
    # Кнопка для выполнения запроса
    if st.button("Запросить четверостишие", key="btn_tab2"):
        if "OPENAI_API_KEY" not in st.session_state:
            st.error("Укажите ключ OpenAI в настройках сайдбара.")
        else:
            openai.api_key = st.session_state["OPENAI_API_KEY"]
            try:
                resp = openai.ChatCompletion.create(
                    model=st.session_state["OPENAI_MODEL"],
                    messages=[
                        {"role": "system", "content": "Вы — поэт, создающий четверостишия на русском языке."},
                        {"role": "user", "content": "расскажи четверостишие"}
                    ],
                    temperature=st.session_state["OPENAI_TEMPERATURE"],
                    top_p=st.session_state["OPENAI_TOP_P"],
                    max_tokens=st.session_state["OPENAI_MAX_TOKENS"]
                )
                content = resp.choices[0].message.content
                st.session_state["tab2_response"] = content
            except Exception as e:
                st.error(f"Ошибка при запросе к OpenAI: {e}")
# ----------------------------- Вкладка 3: Библиотека промптов (заглушка): Библиотека промптов (заглушка) ----------
with tab3:
    st.subheader("Библиотека промптов — в разработке")

# ----------------------------- Вкладка 4: Метрики (заглушка) ----------------------
with tab4:
    st.subheader("Метрики — в разработке")
