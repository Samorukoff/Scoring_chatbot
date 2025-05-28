import logging
import re
from aiogram import Router, F, CommandStart
from aiogram.fsm.context import FSMContext
from aiogram.filters import StateFilter
from aiogram.types import CallbackQuery, Message, InlineKeyboardMarkup, InlineKeyboardButton

from handlers.states import Form

router = Router()
logging.basicConfig(level=logging.INFO)

# Начальный экран
@router.message(CommandStart)
async def start(message: Message):
    await message.answer('Добро пожаловать в ФБИ-Банк!\n' \
    'Для получения своего кредитного рейтинга вам необходимо пройти небольшое тестирование.' \
    'По его результатам мы сможем оценить ваш уровень кредитоспособности для дальнейшего кредитования\n\n'
    'Если готовы начать тестирование, нажмите кнопку "СТАРТ"', reply_markup = InlineKeyboardMarkup(inline_keyboard=
                                     [[InlineKeyboardButton(text='▶️ СТАРТ', callback_data='start_test')]]))


# Универсальная функция валидации чисел
async def validate_and_store(message: Message, state: FSMContext, field, cast_type=float, min_val=None, max_val=None):
    try:
        value = cast_type(message.text.replace(',', '.'))
        if min_val is not None and value < min_val:
            raise ValueError
        if max_val is not None and value > max_val:
            raise ValueError
        await state.update_data({field: value})
        return True
    except ValueError:
        await message.answer("❗ Пожалуйста, введите корректное числовое значение.")
        return False

# Вопросы
@router.callback_query(F.data('start_test'))
async def ask_age(call: CallbackQuery, state: FSMContext):
    await call.answer()
    await state.set_state(Form.age)
    await call.message.answer('Сколько вам лет?')

@router.message(Form.age)
async def ask_income(message: Message, state: FSMContext):
    if await validate_and_store(message, state, "age", int, 14, 120):
        await state.set_state(Form.occupation)
        await message.answer("Укажите вашу специальность из предложенного списка:",
                             reply_markup = InlineKeyboardMarkup(inline_keyboard=[
                                [InlineKeyboardButton(text="Инженер", callback_data="Occupation_Engineer")],
                                [InlineKeyboardButton(text="Врач", callback_data="Occupation_Doctor")],
                                [InlineKeyboardButton(text="Преподаватель", callback_data="Occupation_Teacher")],
                                [InlineKeyboardButton(text="Юрист", callback_data="Occupation_Lawyer")],
                                [InlineKeyboardButton(text="Художник/дизайнер", callback_data="Occupation_Artist")],
                                [InlineKeyboardButton(text="Ученый", callback_data="Occupation_Scientist")],
                                [InlineKeyboardButton(text="Бухгалтер", callback_data="Occupation_Accountant")],
                                [InlineKeyboardButton(text="Продавец", callback_data="Occupation_Salesperson")],
                                [InlineKeyboardButton(text="Менеджер", callback_data="Occupation_Manager")],
                                [InlineKeyboardButton(text="Медсестра", callback_data="Occupation_Nurse")],
                                [InlineKeyboardButton(text="Другое", callback_data="Occupation________")]
                            ]))

@router.callback_query(Form.occupation)
async def ask_income(call: CallbackQuery, state: FSMContext):
    await call.answer()
    value = str(call.data)
    await state.update_data({'occupation': value})
    await state.set_state(Form.annual_income)
    await call.message.answer("Каков ваш годовой доход (в рублях)?")

@router.message(Form.annual_income)
async def ask_salary(message: Message, state: FSMContext):
    if await validate_and_store(message, state, "annual_income", float, 0):
        await state.set_state(Form.monthly_salary)
        await message.answer("Охарактеризуйте ваше финансовое поведение",
                             reply_markup=InlineKeyboardMarkup(inline_keyboard=[
                                [InlineKeyboardButton(text="Высокие траты / Большие выплаты", callback_data="Payment_Behaviour_High_spent_Large_value_payments")],
                                [InlineKeyboardButton(text="Высокие траты / Medium payments", callback_data="Payment_Behaviour_High_spent_Medium_value_payments")],
                                [InlineKeyboardButton(text="Высокие траты / Small payments", callback_data="Payment_Behaviour_High_spent_Small_value_payments")],
                                [InlineKeyboardButton(text="Низкие траты / Большие выплаты", callback_data="Payment_Behaviour_Low_spent_Large_value_payments")],
                                [InlineKeyboardButton(text="Низкие траты / Средние выплаты", callback_data="Payment_Behaviour_Low_spent_Medium_value_payments")],
                                [InlineKeyboardButton(text="Низкие траты / Маленькие выплаты", callback_data="Payment_Behaviour_Low_spent_Small_value_payments")],
                            ]))

@router.callback_query(Form.payment_behaviour)
async def ask_salary(call: CallbackQuery, state: FSMContext):
    await call.answer()
    value = str(call.data)
    await state.update_data({'payment_behaviour': value})
    await state.set_state(Form.monthly_salary)
    await call.message.answer("Какая сумма остаётся у вас на руках в месяц после всех платежей?")

@router.message(Form.monthly_salary)
async def ask_accounts(message: Message, state: FSMContext):
    if await validate_and_store(message, state, "monthly_salary", float, 0):
        await state.set_state(Form.bank_accounts)
        await message.answer("Сколько у вас банковских счетов?")

@router.message(Form.bank_accounts)
async def ask_cards(message: Message, state: FSMContext):
    if await validate_and_store(message, state, "bank_accounts", int, 0):
        await state.set_state(Form.credit_cards)
        await message.answer("Сколько у вас кредитных карт?")

@router.message(Form.credit_cards)
async def ask_interest(message: Message, state: FSMContext):
    if await validate_and_store(message, state, "credit_cards", int, 0):
        await state.set_state(Form.interest_rate)
        await message.answer("Какая процентная ставка по вашим займам?")

@router.message(Form.interest_rate)
async def ask_delay_days(message: Message, state: FSMContext):
    if await validate_and_store(message, state, "interest_rate", float, 0, 100):
        await state.set_state(Form.delay_from_due_date)
        await message.answer("В среднем на сколько дней вы опаздываете с оплатой?")

@router.message(Form.delay_from_due_date)
async def ask_delayed_count(message: Message, state: FSMContext):
    if await validate_and_store(message, state, "delay_from_due_date", float, 0):
        await state.set_state(Form.delayed_payments)
        await message.answer("Сколько раз вы задерживали платежи?")

@router.message(Form.delayed_payments)
async def ask_credit_change(message: Message, state: FSMContext):
    if await validate_and_store(message, state, "delayed_payments", int, 0):
        await state.set_state(Form.changed_credit_limit)
        await message.answer("Укажите изменение кредитного лимита (например, -2000 или 5000)")

@router.message(Form.changed_credit_limit)
async def ask_inquiries(message: Message, state: FSMContext):
    if await validate_and_store(message, state, "changed_credit_limit", float):
        await state.set_state(Form.credit_inquiries)
        await message.answer("Сколько раз вы запрашивали кредитные предложения за последний год?")

@router.message(Form.credit_inquiries)
async def ask_debt(message: Message, state: FSMContext):
    if await validate_and_store(message, state, "credit_inquiries", int, 0):
        await state.set_state(Form.outstanding_debt)
        await message.answer("Какова ваша текущая задолженность (в рублях)?")

@router.message(Form.outstanding_debt)
async def ask_utilization(message: Message, state: FSMContext):
    if await validate_and_store(message, state, "outstanding_debt", float, 0):
        await state.set_state(Form.credit_util_ratio)
        await message.answer("Какой процент от доступного кредита вы используете? (0–100)")

@router.message(Form.credit_util_ratio)
async def ask_history(message: Message, state: FSMContext):
    if await validate_and_store(message, state, "credit_util_ratio", float, 0):
        await state.set_state(Form.credit_history_age)
        await message.answer("Каков возраст вашей кредитной истории? (в месяцах)")

@router.message(Form.credit_history_age)
async def ask_emi(message: Message, state: FSMContext):
    if await validate_and_store(message, state, "credit_history_age", int, 0):
        await state.set_state(Form.total_emi)
        await message.answer("Сколько вы платите ежемесячно по всем кредитам (EMI)?")

@router.message(Form.total_emi)
async def ask_investment(message: Message, state: FSMContext):
    if await validate_and_store(message, state, "total_emi", float, 0):
        await state.set_state(Form.monthly_investment)
        await message.answer("Сколько вы инвестируете ежемесячно? (в рублях)")

@router.message(Form.monthly_investment)
async def ask_balance(message: Message, state: FSMContext):
    if await validate_and_store(message, state, "monthly_investment", float, 0):
        await state.set_state(Form.monthly_balance)
        await message.answer("Какой у вас средний остаток на счету в конце месяца?")