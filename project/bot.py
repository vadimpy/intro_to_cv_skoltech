from telebot.async_telebot import AsyncTeleBot
import numpy as np
import cv2

from docscan import get_doc

bot = AsyncTeleBot('5920217051:AAGe58_5lJVSvygKM3ki9-pAL3II7AP_Mos')

@bot.message_handler(content_types=["photo"])
async def process_image(message):
    fileID = message.photo[-1].file_id
    file_info = await bot.get_file(fileID)

    bin_img = await bot.download_file(file_info.file_path)
    nparr = np.frombuffer(bin_img, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    doc = get_doc(img)

    if doc is not None:
        res, img_encode = cv2.imencode('.jpg', doc)
        data_encode = np.array(img_encode)
        byte_encode = data_encode.tobytes()
        await bot.send_photo(message.chat.id, byte_encode)

import asyncio
asyncio.run(bot.polling())
