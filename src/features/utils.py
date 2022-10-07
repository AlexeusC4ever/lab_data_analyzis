import category_encoders as ce
import src.config as cfg

# class TargetEncoder(BaseEstimator, TransformerMixin):
class TargetEncoder:
    
    def __init__(self, cat_cols, num_classes=1):
        self.te = [ce.TargetEncoder(drop_invariant=False)] * num_classes
        self.cat_cols = cat_cols
        self.num_classes = num_classes
        
        
    def fit(self, x, y):
        for target_num in range(self.num_classes):
            target_col = cfg.TARGET_COLS[target_num]
            self.te[target_num].fit(x[self.cat_cols], y[target_col])

        
    def transform(self, x, y=None):
        target_encoded_x = x.copy()
        for target_num in range(self.num_classes):
            target_col = cfg.TARGET_COLS[target_num]
            encoded_cols = self.te[target_num].transform(x[self.cat_cols])
            
            target_encoded_x[[col + "/" + target_col for col in self.cat_cols]] = encoded_cols

        target_encoded_x = target_encoded_x.drop(self.cat_cols, axis=1)
        return target_encoded_x
            
    def fit_transform(self, x, y=None):
        self.fit(x, y)
        return self.transform(x, y)
        
        
        
        
def second_hand_smoke_count(value):
    if value == '1-2 раза в неделю':
        return 1.5
    if value == '2-3 раза в день':
        return 17.5
    if value == '3-6 раз в неделю':
        return 4.5
    if value == '4 и более раз в день':
        return 28.0
    if value == 'не менее 1 раза в день':
        return 7.0
    return 0.0
    
    
def fill_real_cols_with_zero(df):
    df = df.fillna(0.)
    return df

def fill_real_cols(data):
    data.loc[:, cfg.REAL_COLS[:-1]] = fill_real_cols_with_zero(data.loc[:, cfg.REAL_COLS[:-1]])
    data['Частота пасс кур'] = [second_hand_smoke_count(i) for i in data['Частота пасс кур']]
    return data


class empty_column_filler:
    def second_hand_smoke_count(self, value):
        if value == '1-2 раза в неделю':
            return 1.5
        if value == '2-3 раза в день':
            return 17.5
        if value == '3-6 раз в неделю':
            return 4.5
        if value == '4 и более раз в день':
            return 28.0
        if value == 'не менее 1 раза в день':
            return 7.0
        return 0.0
    
    
    def fill_real_cols_with_zero(self, df):
        df = df.fillna(0.)
        return df

    def fill_real_cols(self, data):
        data.loc[:, cfg.REAL_COLS[:-1]] = fill_real_cols_with_zero(data.loc[:, cfg.REAL_COLS[:-1]])
        data['Частота пасс кур'] = [second_hand_smoke_count(i) for i in data['Частота пасс кур']]
        return data


    def fit(self, x, y=None):
        x = self.fill_real_cols(x)

    def transform(self, x, y=None):
        x = self.fill_real_cols(x)
        return x

    def fit_transform(self, x, y=None):
        return self.transform(x)
       