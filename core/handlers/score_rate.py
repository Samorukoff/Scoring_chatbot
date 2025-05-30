import logging
from datetime import datetime
import joblib

import pandas as pd

from aiogram import Router, F
from aiogram.fsm.context import FSMContext
from aiogram.filters import StateFilter
from aiogram.types import CallbackQuery, Message, InlineKeyboardMarkup, InlineKeyboardButton

from core.handlers.states import Form
from core.handlers.score_test import validate_and_store

import httpx
from lxml import etree

router = Router()
logging.basicConfig(level=logging.INFO)

# Выгрузка курса доллара
async def get_usd_rate_cbr() -> float:
    url = "https://www.cbr.ru/scripts/XML_daily.asp"

    async with httpx.AsyncClient() as client:
        response = await client.get(url)

    tree = etree.fromstring(response.content)

    for valute in tree.findall("Valute"):
        char_code = valute.find("CharCode").text
        if char_code == "USD":
            value_str = valute.find("Value").text.replace(',', '.')
            return float(value_str)
    
    raise RuntimeError("Курс доллара не найден.")

@router.message(Form.monthly_balance)
async def result(message: Message, state: FSMContext):
    if await validate_and_store(message, state, "monthly_balance", float, 0):
        await state.set_state(Form.result)

        # Достаем данные
        data = await state.get_data()

        # Базовые числовые признаки
        usd_rate = await get_usd_rate_cbr()

        input_data = {
            'Age': int(data['age']),
            'Annual_Income': round(float(data['annual_income']) / usd_rate, 2),
            'Monthly_Inhand_Salary': round(float(data['monthly_salary']) / usd_rate, 2),
            'Num_Bank_Accounts': int(data['bank_accounts']),
            'Num_Credit_Card': int(data['credit_cards']),
            'Interest_Rate': round(float(data['interest_rate']), 2),
            'Delay_from_due_date': int(data['delay_from_due_date']),
            'Num_of_Delayed_Payment': int(data['delayed_payments']),
            'Changed_Credit_Limit': round(float(data['changed_credit_limit']) / usd_rate, 2),
            'Num_Credit_Inquiries': int(data['credit_inquiries']),
            'Outstanding_Debt': round(float(data['outstanding_debt']) / usd_rate, 2),
            'Credit_Utilization_Ratio': round(float(data['credit_util_ratio']), 2),
            'Credit_History_Age': int(data['credit_history_age']),
            'Total_EMI_per_month': round(float(data['total_emi']) / usd_rate, 2),
            'Amount_invested_monthly': round(float(data['monthly_investment']) / usd_rate, 2),
            'Monthly_Balance': round(float(data['monthly_balance']) / usd_rate, 2),
        }

        # --------- Добавим категориальные признаки ---------
        # Инициализируем все возможные категориальные поля значением False
        one_hot_columns = [
            # Месяцы
            'Month_January', 'Month_February', 'Month_March', 'Month_May', 'Month_June', 'Month_July', 'Month_August',

            # Occupation
            'Occupation_Architect', 'Occupation_Developer', 'Occupation_Doctor', 'Occupation_Engineer',
            'Occupation_Entrepreneur', 'Occupation_Journalist', 'Occupation_Lawyer', 'Occupation_Manager',
            'Occupation_Mechanic', 'Occupation_Media_Manager', 'Occupation_Musician', 'Occupation_Scientist',
            'Occupation_Teacher', 'Occupation_Writer', 'Occupation________',

            # Payment_of_Min_Amount
            'Payment_of_Min_Amount_No', 'Payment_of_Min_Amount_Yes',

            # Payment_Behaviour
            'Payment_Behaviour_High_spent_Large_value_payments',
            'Payment_Behaviour_High_spent_Medium_value_payments',
            'Payment_Behaviour_High_spent_Small_value_payments',
            'Payment_Behaviour_Low_spent_Large_value_payments',
            'Payment_Behaviour_Low_spent_Medium_value_payments',
            'Payment_Behaviour_Low_spent_Small_value_payments'
        ]

        for col in one_hot_columns:
            input_data[col] = False

        # Установка текущего месяца
        month = datetime.now().strftime('%B')  # May, June и т.п.
        month_col = f'Month_{month}'
        if month_col in one_hot_columns:
            input_data[month_col] = True

        # Установка occupation
        occ_col = data.get('occupation')
        if occ_col in one_hot_columns:
            input_data[occ_col] = True

        # Установка payment behaviour
        pay_col = data.get('payment_behaviour')
        if pay_col in one_hot_columns:
            input_data[pay_col] = True

        # Установка Payment_of_Min_Amount
        if data.get('payment_of_min_amount') is True:
            input_data['Payment_of_Min_Amount_Yes'] = True
        elif data.get('payment_of_min_amount') is False:
            input_data['Payment_of_Min_Amount_No'] = True

        # Определяем результат
        X_input = pd.DataFrame([input_data])

        # Восстанавливаем порядок признаков
        feature_order = joblib.load("core/model/_feature_order.pkl")
        X_input = X_input[feature_order]

        # Определяем кредитный рейтинг
        model = joblib.load("core/model/_stacking_model.pkl")
        labels = {0: "Низкий", 1: "Средний", 2: "Высокий"}
        prediction_index = int(model.predict(X_input)[0])
        prediction = labels.get(prediction_index, "Неопределенный")

        # Определяем числовой рейтинг исходя из вероятностей
        probs = model.predict_proba(X_input)
        probs = probs[0]
        logging.info(f'Вероятности: {probs}')
        rate = int((probs[0] * 0 + probs[1] * 0.5 + probs[2] * 1) * 999)

        # Вывод ответов
        lines = ["📋 *Введённые данные:*"]
        row = X_input.iloc[0]

        for col, val in row.items():
            if val in [False, 0, 0.0, None] or pd.isna(val):
                continue  # Пропускаем "пустые" значения
            if val is True:
                display_val = "✅"
            else:
                display_val = f"`{val}`"
            lines.append(f"• *{col}*: {display_val}")

        answer_text = "\n".join(lines)

        await message.answer(
            f'Ваш кредитный рейтинг: *{prediction}*\n\n*{rate} баллов*\n\n{answer_text}',
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup(inline_keyboard=[
                                [InlineKeyboardButton(text="Вернуться в начало", callback_data="go_to_start")]])
        )

# Начальный экран
@router.callback_query(F.data == 'go_to_start')
async def start(call: CallbackQuery, state: FSMContext):
    await call.answer()
    await state.clear()
    await call.message.answer('Добро пожаловать в ФБИ-Банк!\n\n'
    'Для получения своего кредитного рейтинга вам необходимо пройти небольшое тестирование.'
    'По его результатам мы сможем оценить ваш уровень кредитоспособности для дальнейшего кредитования\n\n'
    'Если готовы начать тестирование, нажмите кнопку "СТАРТ"', reply_markup = InlineKeyboardMarkup(inline_keyboard=
                                     [[InlineKeyboardButton(text='▶️ СТАРТ', callback_data='start_test')]]))