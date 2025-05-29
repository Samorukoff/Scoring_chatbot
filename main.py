from aiogram import Bot, Dispatcher
from aiogram.types import BotCommand
import logging
import asyncio

from core.handlers.score_test import router as test_router
from core.handlers.score_rate import router as rate_router

from core.settings import settings



async def start():
    bot = Bot(token=settings.bots.bot_token)
    dp = Dispatcher()

    # Подключаем все роутеры
    dp.include_router(test_router)
    dp.include_router(rate_router)

    # Установка команд для бота
    await bot.set_my_commands([BotCommand(command="start", description="Запустить бота")])
    
    try:
        logging.info("Бот запущен. Ожидание событий...")
        await dp.start_polling(bot)
    except (KeyboardInterrupt, asyncio.CancelledError):
        logging.info("Бот остановлен вручную.")
    except Exception:
        logging.exception("Произошла критическая ошибка во время работы бота.")
    finally:
        await bot.session.close()
        logging.info("Сессия бота закрыта.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    try:
        asyncio.run(start())
    except KeyboardInterrupt:
        logging.info("Завершение через KeyboardInterrupt (Ctrl+C)")