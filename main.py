import pandas as pd
import joblib
import streamlit as st
from catboost import CatBoostClassifier
import seaborn as sns
import matplotlib.pyplot as plt


model = CatBoostClassifier()
model.load_model("catboost_model.cbm")

scaler = joblib.load("scaler.pkl")

FEATURE_COLS = joblib.load("feature_cols.pkl")

num_cols = ['Age', 'Flight Distance', 'Arrival Delay in Minutes']

results_df = pd.read_csv('results.csv')


mapping_gender = {'Мужчина': 0, 'Женщина': 1}
mapping_customer = {'Нелояльный клиент': 0, 'Лояльный клиент': 1}
mapping_travel = {'Личное путешествие': 0, 'Командировка': 1}
mapping_class = {'Эконом': 0, 'Эконом Плюс': 1, 'Бизнес': 2}



def make_feature_row(
    gender,
    customer_type,
    travel_type,
    ticket_class,
    age,
    flight_distance,
    arrival_delay,
    inflight_wifi,
    ease_booking,
    food_drink,
    online_boarding,
    inflight_ent,
    onboard_service,
    leg_room,
    checkin_service,
    inflight_service,
    cleanliness
):
    data = {
        'Gender': mapping_gender[gender],
        'Customer Type': mapping_customer[customer_type],
        'Type of Travel': mapping_travel[travel_type],
        'Class': mapping_class[ticket_class],
        'Age': age,
        'Flight Distance': flight_distance,
        'Inflight wifi service': inflight_wifi,
        'Ease of Online booking': ease_booking,
        'Food and drink': food_drink,
        'Online boarding': online_boarding,
        'Inflight entertainment': inflight_ent,
        'On-board service': onboard_service,
        'Leg room service': leg_room,
        'Checkin service': checkin_service,
        'Inflight service': inflight_service,
        'Cleanliness': cleanliness,
        'Arrival Delay in Minutes': arrival_delay
    }

    df = pd.DataFrame([data])
    df = df.reindex(columns=FEATURE_COLS, fill_value=0)
    df[num_cols] = scaler.transform(df[num_cols])

    return df



def predict_satisfaction(**kwargs):
    X_user = make_feature_row(**kwargs)

    label = int(model.predict(X_user)[0])
    proba = float(model.predict_proba(X_user)[0, 1])

    return label, proba


st.title("Прогноз удовлетворённости пассажира авиакомпании")

st.header("Основная информация")

gender = st.selectbox("Пол", ["Мужчина", "Женщина"])
customer_type = st.selectbox("Тип клиента", ["Нелояльный клиент", "Лояльный клиент"])
travel_type = st.selectbox("Тип путешествия", ["Личное путешествие", "Командировка"])
ticket_class = st.selectbox("Класс обслуживания", ["Эконом", "Эконом Плюс", "Бизнес"])

age = st.number_input("Возраст", min_value=1, max_value=100, value=30)
flight_distance = st.number_input("Дальность перелёта (км)", min_value=1, max_value=10000, value=1000)
arrival_delay = st.number_input("Задержка прибытия (мин.)", min_value=0, max_value=300, value=0)

st.header("Оценки качества сервиса (1 = очень плохо, 5 = отлично)")

inflight_wifi = st.slider("Wi-Fi на борту", 1, 5, 1)
ease_booking = st.slider("Удобство онлайн-бронирования", 1, 5, 1)
food_drink = st.slider("Еда и напитки", 1, 5, 1)
online_boarding = st.slider("Онлайн-посадка", 1, 5, 1)
inflight_ent = st.slider("Развлечения на борту", 1, 5, 1)
onboard_service = st.slider("Обслуживание на борту", 1, 5, 1)
leg_room = st.slider("Место для ног", 1, 5, 1)
checkin_service = st.slider("Регистрация на рейс", 1, 5, 1)
inflight_service = st.slider("Услуги на борту", 1, 5, 1)
cleanliness = st.slider("Чистота", 1, 5, 1)

if st.button("Спрогнозировать удовлетворённость"):
    label, proba = predict_satisfaction(
        gender=gender,
        customer_type=customer_type,
        travel_type=travel_type,
        ticket_class=ticket_class,
        age=age,
        flight_distance=flight_distance,
        arrival_delay=arrival_delay,
        inflight_wifi=inflight_wifi,
        ease_booking=ease_booking,
        food_drink=food_drink,
        online_boarding=online_boarding,
        inflight_ent=inflight_ent,
        onboard_service=onboard_service,
        leg_room=leg_room,
        checkin_service=checkin_service,
        inflight_service=inflight_service,
        cleanliness=cleanliness
    )

    st.subheader("Результат")

    if label == 1:
        st.success(f"Пассажир, скорее всего, будет УДОВЛЕТВОРЁН. \nВероятность: {proba:.2%}")
    else:
        st.error(f"Пассажир может быть НЕУДОВЛЕТВОРЁН. \nВероятность удовлетворённости: {proba:.2%}")

importance_cols = [
    'Wi-Fi на борту',
    'Удобство онлайн-бронирования',
    'Еда и напитки',
    'Онлайн-посадка',
    'Развлечения на борту',
    'Обслуживание на борту',
    'Место для ног',
    'Регистрация',
    'Услуги на борту',
    'Чистота',
    'Задержка прибытия (мин)'
]

segment_cols = [
    'Пол',
    'Возраст',
    'Тип путешествия',
    'Тип клиента',
    'Класс обслуживания',
    'Дальность перелёта'
]

def select_categories(results_df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    mask = pd.Series(True, index=results_df.index)

    for col, val in filters.items():
        if val != "Не выбрано":
            mask &= (results_df[col] == val)

    return results_df[mask].copy()


def analyze_single_category(row: pd.Series):
    st.subheader("Категория пассажиров")
    for col in segment_cols:
        st.write(f"**{col}:** {row[col]}")

    imps = row[importance_cols]

    top3 = imps.sort_values(ascending=False).head(3)
    bottom3 = imps.sort_values(ascending=True).head(3)

    st.subheader("Топ-3 наиболее важных факторов")
    for feat, val in top3.items():
        st.write(f"**{feat}**")

    st.subheader("Топ-3 наименее важных факторов")
    for feat, val in bottom3.items():
        st.write(f"**{feat}**")


def analyze_multiple_categories(subset: pd.DataFrame):
    imps = subset[importance_cols]

    top1 = imps.idxmax(axis=1)

    top3 = imps.apply(lambda s: s.sort_values(ascending=False).head(3).index.tolist(), axis=1)
    bottom3 = imps.apply(lambda s: s.sort_values(ascending=True).head(3).index.tolist(), axis=1)

    top1_counts = top1.value_counts()

    other_imp_counts = pd.Series(0, index=importance_cols)
    for idx, feats in top3.items():
        for f in feats:
            if f != top1[idx]:
                other_imp_counts[f] += 1
    other_imp_counts = other_imp_counts[other_imp_counts > 0]

    not_imp_counts = pd.Series(0, index=importance_cols)
    for feats in bottom3:
        for f in feats:
            not_imp_counts[f] += 1
    not_imp_counts = not_imp_counts[not_imp_counts > 0]

    st.subheader(f"Найдено категорий: {len(subset)}")

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    palette = sns.color_palette("Blues", n_colors=10)

# Top-1
    top1_counts.plot(
        kind="pie",
        autopct="%1.1f%%",
        ax=axes[0],
        colors=palette,
        textprops={'fontsize': 8},
        pctdistance=0.8,
        labeldistance=1.1
    )
    axes[0].set_ylabel("")
    axes[0].set_title("Самые важные факторы (Top-1)")

    # Other important
    if len(other_imp_counts) > 0:
        other_imp_counts.plot(
            kind="pie",
            autopct="%1.1f%%",
            ax=axes[1],
            colors=palette,
            textprops={'fontsize': 8},
            pctdistance=0.8,
            labeldistance=1.1
        )
        axes[1].set_ylabel("")
        axes[1].set_title("Другие значимые факторы (входящие в Top-3)")
    else:
        axes[1].set_axis_off()

    # Not important
    if len(not_imp_counts) > 0:
        not_imp_counts.plot(
            kind="pie",
            autopct="%1.1f%%",
            ax=axes[2],
            colors=palette,
            textprops={'fontsize': 8},
            pctdistance=0.8,
            labeldistance=1.1
        )
        axes[2].set_ylabel("")
        axes[2].set_title("Наименее важные факторы")
    else:
        axes[2].set_axis_off()

    st.pyplot(fig)


def generate_report_streamlit(results_df: pd.DataFrame):
    st.title("Отчет о важных признаках для разных категорий")

    st.header("Фильтры категории")

    options = {"Не выбрано"}  

    gender = st.selectbox("Пол", ["Не выбрано"] + sorted(results_df["Пол"].unique().tolist()))
    age = st.selectbox("Возраст", ["Не выбрано"] + sorted(results_df["Возраст"].unique().tolist()))
    travel = st.selectbox("Тип путешествия", ["Не выбрано"] + sorted(results_df["Тип путешествия"].unique().tolist()))
    customer = st.selectbox("Тип клиента", ["Не выбрано"] + sorted(results_df["Тип клиента"].unique().tolist()))
    cls = st.selectbox("Класс обслуживания", ["Не выбрано"] + sorted(results_df["Класс обслуживания"].unique().tolist()))
    dist = st.selectbox("Дальность перелёта", ["Не выбрано"] + sorted(results_df["Дальность перелёта"].unique().tolist()))

    if st.button("Сформировать отчёт"):
        filters = {
            "Пол": gender,
            "Возраст": age,
            "Тип путешествия": travel,
            "Тип клиента": customer,
            "Класс обслуживания": cls,
            "Дальность перелёта": dist
        }

        subset = select_categories(results_df, filters)

        if len(subset) == 0:
            st.error("Категории не найдены для выбранных условий.")
            return

        if len(subset) == 1:
            analyze_single_category(subset.iloc[0])
        else:
            analyze_multiple_categories(subset)

generate_report_streamlit(results_df)




