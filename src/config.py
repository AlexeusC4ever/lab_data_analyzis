# PATH
RAW_PATH = 'data/raw/'


# COLS
TARGET_COLS = ['Артериальная гипертензия', 'ОНМК', 'Стенокардия, ИБС, инфаркт миокарда', 'Сердечная недостаточность', 'Прочие заболевания сердца']
ID_COL = 'ID'
EDU_COL = 'Образование'
SEX_COL = 'Пол'
CAT_COLS = [
    'Пол', 'Семья', 'Этнос', 'Национальность', 'Религия', 'Образование', 
    'Профессия', 'Статус Курения', 'Алкоголь',
    'Время засыпания', 'Время пробуждения'
]
OHE_COLS = [
    'Пол', 'Вы работаете?', 'Выход на пенсию', 'Прекращение работы по болезни', 'Сахарный диабет', 'Гепатит',
    'Онкология', 'Хроническое заболевание легких', 'Бронжиальная астма', 'Туберкулез легких ', 'ВИЧ/СПИД',
    'Регулярный прим лекарственных средств', 'Травмы за год', 'Переломы','Пассивное курение', 'Сон после обеда', 
    'Спорт, клубы', 'Религия, клубы'
]
REAL_COLS = ['Возраст курения', 'Сигарет в день', 'Возраст алког', 'Частота пасс кур']

path_to_splitted_train_data = 'data/processed/train.pkl'
path_to_splitted_train_data_target = 'data/processed/train_target.pkl'
path_to_splitted_val_data = 'data/processed/val.pkl'
path_to_splitted_val_data_target = 'data/processed/val_target.pkl'

name_of_featured_train_data = '/featured_train_data.pkl'
name_of_featured_val_data = '/featured_val_data.pkl'
name_of_featured_t_encoded_train_data = '/featured_train_data_t_encoded.pkl'
name_of_featured_t_encoded_val_data = '/featured_val_data_t_encoded.pkl'

path_to_target_encoder = 'src/models/tar_encoder.pkl'

name_catboost_model = 'catboost_model.pkl'
name_lr_model = 'lr_model.pkl'

num_classes = 5