from collections import deque
from PIL import Image
import numpy as np
from typing import List
#Вариант №2
encoding: str = 'utf-8'

class Pixel:
    pointer: int = 0

    def __init__(self, rgba: np.array):
        self.rgba: np.array = rgba
        self.order = Pixel.pointer
        Pixel.pointer += 1

class BruyndonckxMethod:
    def __init__(self, empty_image_path: str, filled_image_path: str):
        self.empty_image_path: str = empty_image_path
        self.filled_image_path: str = filled_image_path
        self.delta_l = 5
        self.occupancy: int = 0

    @staticmethod
    def str_to_bits(message: str):
        result = []
        for num in list(message.encode(encoding=encoding)):
            result.extend([(num >> x) & 1 for x in range(7, -1, -1)])
        return result

    @staticmethod
    def bits_to_str(bits: list) -> str:
        chars = []
        for b in range(len(bits) // 8):
            byte = bits[b * 8:(b + 1) * 8]
            chars.append(chr(int(''.join([str(bit) for bit in byte]), 2)))
        return ''.join(chars)

    def __func(self):
        g = {}
        for i in range(64):
            group = np.random.choice(['1A', '1B']) if i < 32 else np.random.choice(['2A', '2B'])
            g.setdefault(group, []).append(i)
        return g

    #Этот метод modification_brightness принимает список пикселей sorted_block_pixels, которые представляют собой отсортированный блок пикселей изображения.
    # Он изменяет яркость пикселей в блоке в
    # зависимости от значения бита bit, который указывает, должна ли яркость увеличиться или уменьшиться для скрытия информации.
    def modification_brightness(self, sorted_block_pixels: List[Pixel], bit:int):
        g = self.__func()
        arr = np.asarray([pixel.rgba for pixel in sorted_block_pixels],dtype=np.uint8)
        delta_l = self.delta_l
        sign = 1 if bit else -1
        g1_arr = arr[(g['1A'] + g['1B'])]
        arr[g['1A'], 3] -= (np.mean(arr[g['1A'], 3]) - (np.mean(g1_arr[:, 3]) + (sign * arr[g['1B']].shape[0] *delta_l / g1_arr.shape[0]))).astype(np.uint8)
        arr[g['1B'], 3] -= (np.mean(arr[g['1B'], 3]) - (np.mean(g1_arr[:, 3]) - (sign * arr[g['1A']].shape[0] *delta_l / g1_arr.shape[0]))).astype(np.uint8)

        g2_arr = arr[(g['2A'] + g['2B'])]
        arr[g['2A'], 3] -= (np.mean(arr[g['2A'], 3]) - (np.mean(g2_arr[:, 3]) + (sign * arr[g['2B']].shape[0] *delta_l / g2_arr.shape[0]))).astype(np.uint8)
        arr[g['2B'], 3] -= (np.mean(arr[g['2B'], 3]) - (np.mean(g2_arr[:, 3]) - (sign * arr[g['2A']].shape[0] *delta_l / g2_arr.shape[0]))).astype(np.uint8)
        for i, pixel in enumerate(arr):
            sorted_block_pixels[i].rgba = pixel

    #Встраивает сообщение в изображение с использованием метода Брюндонкса.
    def embed(self, message: str, key_generator: int):
        np.random.seed(key_generator)
        with Image.open(self.empty_image_path).convert('RGBA') as img:
            picture = np.asarray(img, dtype=np.uint8).astype(np.uint8)
            picture[:, :, 3] = (0.299 * picture[:, :, 0] + 0.587 * picture[:,
:, 1] + 0.114 * picture[:, :, 2]).astype(int)
        height, width = picture.shape[0], picture.shape[1]
        message_bits = self.str_to_bits(message)
        message_bits_length = len(message_bits)
        if message_bits_length > (height // 8) * (width // 8):
            raise ValueError('Размер сообщения превышает размер контейнера!')
        message_bits = deque(message_bits)
        for i in range(8, height - 7, 8):
            for j in range(8, width - 7, 8):
                old_block = picture[i - 8: i, j - 8: j].copy()
                old_size = old_block.shape
                old_block = old_block.reshape(-1, 4)
                new_block = sorted([Pixel(pixel) for pixel in old_block],
key=lambda obj: obj.rgba[3])
                bit = message_bits.popleft()
                self.modification_brightness(new_block, bit)
                new_block = sorted(new_block, key=lambda obj: obj.order)
                new_block = (np.asarray([pixel.rgba for pixel in new_block],
dtype=np.uint8)).reshape(old_size)
                picture[i - 8: i, j - 8: j] = new_block[:, :]
                self.occupancy += 1
                if self.occupancy == message_bits_length:
                    Image.fromarray(picture,
'RGBA').save(self.filled_image_path, 'PNG')
                    np.random.seed()
                    return

    #Извлекает сообщение из изображения, используя метод Брюндонкса.

    def recover(self, key_generator: int):
        np.random.seed(key_generator)
        with Image.open(self.filled_image_path).convert('RGBA') as img:
            picture = np.asarray(img, dtype=np.uint8)
        height, width = picture.shape[0], picture.shape[1]
        message_bits = []
        for i in range(8, height - 7, 8):
            for j in range(8, width - 7, 8):
                modified_block = picture[i - 8: i, j - 8: j].copy()
                modified_block = modified_block.reshape(-1, 4)
                modified_block = sorted([Pixel(pixel) for pixel in modified_block], key=lambda pixel: np.uint8(
                    0.299 * pixel.rgba[0] + 0.587 * pixel.rgba[1] + 0.114 * pixel.rgba[2]))
                g = self.__func()
                arr = np.asarray([pixel.rgba for pixel in modified_block],
                                 dtype=np.uint8)
                if (np.mean(arr[g['1A'], 3]) - np.mean(arr[g['1B'], 3]) > 0) and \
                    (np.mean(arr[g['2A'], 3]) - np.mean(arr[g['2B'], 3]) > 0):message_bits.append(1)
                else:
                    message_bits.append(0)
                if len(message_bits) == self.occupancy:
                    np.random.seed()
                    message = self.bits_to_str(message_bits)
                    return message

#Вычисляет метрики качества между пустым и заполненным изображениями,
# такие как максимальное абсолютное отклонение, норма Минковского и среднее квадратичное отклонение.

#Загружает изображение и сообщение из файлов.
#Создает объект класса BruyndonckxMethod.
#Встраивает сообщение в изображение и сохраняет результат.
#Извлекает сообщение из встроенного изображения.
#Вычисляет и выводит метрики качества изображения после встраивания сообщения.
def metrics(empty_image_path: str, filled_image_path: str):
    with Image.open(empty_image_path).convert('RGBA') as img:
        empty = np.asarray(img, dtype=np.uint8).astype(np.uint8)
        empty[:, :, 3] = (0.299 * empty[:, :, 0] + 0.587 * empty[:, :, 1] +
                          0.114 * empty[:, :, 2]).astype(np.uint8)
    with Image.open(filled_image_path).convert('RGBA') as img:
        full = np.asarray(img, dtype=np.uint8)

    H, W = empty.shape[0], empty.shape[1]
    maxD = np.sum((empty - full) * (empty - full)) / np.sum((empty * empty))
    Lp = 1 / maxD
    MSE = np.sum((empty - full) ** 2) / (W * H)

    print('Максимальное абсолютное отклонение:{}'.format(maxD))
    print('Норма Минковского = {}'.format(Lp))
    print('Среднее квадратичное отклонение:{}'.format(MSE))

empty_image_path = 'input/old_image.png'
filled_image_path = 'output/new_image.png'

with open('message.txt', mode='r', encoding=encoding) as file:
    message = file.read()
key = 21532
b = BruyndonckxMethod(empty_image_path, filled_image_path)
b.embed(message, key)
recovered_message = b.recover(key)
print('Зашированное сообщение:{}'.format(recovered_message))
metrics(empty_image_path, filled_image_path)