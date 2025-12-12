import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

st.set_page_config(page_title="Car Price ML App", layout="wide")

# --------------------------------------------------------
# 1. Загрузка данных
# --------------------------------------------------------
train_url = "https://raw.githubusercontent.com/Murcha1990/MLDS_ML_2022/main/Hometasks/HT1/cars_train.csv"
test_url  = "https://raw.githubusercontent.com/Murcha1990/MLDS_ML_2022/main/Hometasks/HT1/cars_test.csv"

@st.cache_data
def load_data():
    df_train = pd.read_csv(train_url)
    df_test = pd.read_csv(test_url)
    return df_train, df_test

df_train, df_test = load_data()


# --------------------------------------------------------
# 2. ДОБАВЛЕН ТВОЙ ПРЕПРОЦЕССИНГ
# --------------------------------------------------------

# 2.1 сравнение средних и медиан
train_mean = df_train.select_dtypes(include="number").mean()
train_median = df_train.select_dtypes(include="number").median()

test_mean = df_test.select_dtypes(include="number").mean()
test_median = df_test.select_dtypes(include="number").median()

mean_diff = train_mean - test_mean
median_diff = train_median - test_median

# 2.2 удаление дублей
df_train = df_train.drop_duplicates(
    subset=[col for col in df_train.columns if col != "selling_price"],
    keep="first"
)

# 2.3 парсинг mileage / engine / max_power + torque
KGFM_TO_NM = 9.80665

for dataset in [df_train, df_test]:

    # mileage, engine, max_power → первые числа
    for col in ["mileage", "engine", "max_power"]:
        dataset[col] = dataset[col].astype(str).str.extract(r"(\d+\.?\d*)")
        dataset[col] = dataset[col].astype(float)

    # torque подробно
    dataset["torque_raw"] = dataset["torque"].astype(str)

    dataset["torque"] = dataset["torque_raw"].str.extract(r"(\d+\.?\d*)").astype(float)

    # rpm — второе число в строке torque
    dataset["max_torque_rpm"] = (
        dataset["torque_raw"].str.findall(r"(\d+\.?\d*)").str[1].astype(float)
    )

    # перевод kgm → Nm
    dataset["torque"] = dataset["torque"].where(
        ~dataset["torque_raw"].str.contains("kgm", case=False, na=False),
        dataset["torque"] * KGFM_TO_NM
    )

for dataset in [df_train, df_test]:
    num_cols = dataset.select_dtypes(include="number").columns
    for col in num_cols:
        dataset[col] = dataset[col].fillna(dataset[col].median())

    cat_cols = dataset.select_dtypes(exclude="number").columns
    for col in cat_cols:
        mode_val = dataset[col].mode().iloc[0]
        dataset[col] = dataset[col].fillna(mode_val)

# --------------------------------------------------------
# Streamlit вывод среднего / медианного отличия
# --------------------------------------------------------
st.title("🚗 ML App — Ridge Regression + EDA + Preprocessing")

st.header("📘 Разница средних и медиан между train и test")

st.write("### Разница средних:")
st.write(mean_diff)

st.write("### Разница медиан:")
st.write(median_diff)


# --------------------------------------------------------
# 3. EDA
# --------------------------------------------------------
st.header("📊 Exploratory Data Analysis")

num_cols = df_train.select_dtypes(include=np.number).columns

col1, col2 = st.columns(2)

with col1:
    st.subheader("Распределение цены (selling_price)")
    fig, ax = plt.subplots()
    sns.histplot(df_train["selling_price"], kde=True, ax=ax)
    st.pyplot(fig)

with col2:
    st.subheader("Корреляционная матрица")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(df_train[num_cols].corr(), cmap="coolwarm", annot=False)
    st.pyplot(fig)

st.subheader("Boxplot числовых признаков")
fig, ax = plt.subplots(figsize=(10, 4))
sns.boxplot(data=df_train[num_cols], ax=ax)
plt.xticks(rotation=45)
st.pyplot(fig)


# --------------------------------------------------------
# 4. Обучение модели
# --------------------------------------------------------
st.header("⚙ Обучение Ridge Regression")

X = df_train.drop(columns=["selling_price"])
y = df_train["selling_price"]

numeric_features = X.select_dtypes(include=np.number).columns.tolist()

model = Pipeline([
    ("scaler", StandardScaler()),
    ("ridge", Ridge(alpha=1.0))
])

model.fit(X[numeric_features], y)
st.success("Модель успешно обучена!")


# --------------------------------------------------------
# 5. Предсказание
# --------------------------------------------------------
st.header("📝 Предсказание цены автомобиля")

tab1, tab2 = st.tabs(["Загрузить CSV", "Ввести вручную"])

# Загрузка CSV
with tab1:
    uploaded = st.file_uploader("Загрузите CSV", type=["csv"])
    if uploaded:
        df_input = pd.read_csv(uploaded)
        df_input["pred_price"] = model.predict(df_input[numeric_features])
        st.dataframe(df_input)

# Ручной ввод
with tab2:
    st.subheader("Введите значения:")

    user_data = {}
    for col in numeric_features:
        user_data[col] = st.number_input(
            col, value=float(df_train[col].median())
        )

    if st.button("Предсказать"):
        df_input = pd.DataFrame([user_data])
        pred = model.predict(df_input)[0]
        st.subheader(f"💰 Предсказанная цена: **{pred:.2f}**")


# --------------------------------------------------------
# 6. Визуализация весов
# --------------------------------------------------------
st.header("📈 Веса Ridge Regression (коэффициенты)")

ridge = model.named_steps["ridge"]

weights = pd.DataFrame({
    "feature": numeric_features,
    "weight": ridge.coef_
}).sort_values("weight")

st.dataframe(weights)

fig, ax = plt.subplots(figsize=(8, 6))
sns.barplot(data=weights, x="weight", y="feature", ax=ax)
plt.title("Влияние признаков на цену")
st.pyplot(fig)
