import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import re

PROMPT_WORD_EXAMPLES = [
    # Сокращения и модификации
    "уши", "прошка", "50к",
    # Разговорное написание и диминутивы
    "чё", "гамно", "тыща", "бачи", "ясненько", "скучновато",
    # Раздельное, слитное и дефисное написание
    "коллцентры", "по больше", "шьююбки",
    # Согласование и предлоги
    "моделе", "на голову приходит", "с старой",
    # Орфографические ошибки и опечатки
    "типо", "пачему", "учпеть",
    # Повторяющиеся слова и слоги (включил два варианта)
    "ноно", "ноно там ещё",
    # Отсутствие ё
    "причем",
    # Буквенные наращения
    "16тый",
    # Комбинация
    "обращаи тесь",
    # Игра слов
    "вымираты",
    # Окказионализмы
    "вкдилдодрон",
    # Неологизмы
    "дистрибутив", "фейк",
    # Иноязычные слова и варваризмы
    "диджтл", "комьюнити",
    # Жаргон
    "бабки", "жопожник",
    # Синлексемы
    "голосовое",
    # Теги и междометия (ищем то, что было ДО замены на теги)
    "что то", "вауууу",
    # Сложные случаи (опечатки в словарях)
    "павербанк", "коммьюнити", "скил",
    # Аббревиатуры (как сохраняемые, так и те, что расшифровывали)
    "оаэ", "тт", "озу", "впн", "фио", "мб", "лс", "нг", "кз"
]

PROMPT_SENT_EXAMPLES = {
    "original": [
        "Во первых у нас учатся дети от 2 до 17 лет",
        "Ну он фотает все что красиво выглядит да и часто ходит где то так часто фотает что ему даже с флешкой 128 Гб не хватает",
        "оооо кста как то с ним на вписке был",
        "Вк дерьмо ебанное",
        "Мб козловский и петров еще",
        "но может это и вброс китацев но пока почти все сходится с старой информацией",
        "В украине не работает даже с впн",
    ],
    "normalized": [
        "Во-первых <hyphen> у нас учатся дети от 2 до 17 лет",
        "Ну он фотографирует всё что красиво выглядит да и часто ходит где-то <hyphen> так часто фотографирует что ему даже с флешкой 128 гигабайт не хватает",
        "<interjection> кстати как-то <hyphen> с ним на вписке был",
        "Вк дерьмо ебанное",
        "Может|быть Козловский и Петров ещё",
        "Но может это и вброс китайцев но пока почти всё сходится со старой информацией"
        "В Украине не работает даже с ВПН",
    ]
}




def get_train_test(
    n: int, 
    random_state: int = 42, 
    test_size: float = 0.3, 
    shuffle=False
    ):
    """Load original and normalized sentences, split them into train, test

    Args:
        n (int): Number of sentences used from original dataset
        random_state (int, optional): Defaults to 42.
        test_size (float, optional): Defaults to 0.3.

    Returns:
        tuple: X_train, X_test, y_train, y_test
    """
    original, normalized = get_original_normalized(n)
    X_train, X_test, y_train, y_test = train_test_split(
        original,
        normalized,
        random_state=random_state,
        shuffle=shuffle,
        test_size=test_size
    )

    return X_train, X_test, y_train, y_test

def get_original_normalized(n: int, exclude_prompt_examples: bool = True):
    """Load original and normalized sentences

    Args:
        n (int): Number of sentences to load

    Returns:
        tuple: original, normalized
    """    
    df = pd.read_csv(Path(__file__).parent / "dataset.csv")
    original, normalized = df.original.to_list(), df.normalized.to_list()
    
    if not exclude_prompt_examples:
        return original[:n], normalized[:n]
    
    filtered_original = []
    filtered_normalized = []
    for orig, norm in zip(original, normalized):
        word_cond = not any(word in orig.split() for word in PROMPT_WORD_EXAMPLES)
        sent_cond = not any(sentence == orig for sentence in PROMPT_SENT_EXAMPLES["original"])
        
        if word_cond and sent_cond:
            filtered_original.append(orig)
            filtered_normalized.append(norm)

    return filtered_original[:n], filtered_normalized[:n]