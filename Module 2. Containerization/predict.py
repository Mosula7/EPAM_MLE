import pandas as pd
import lightgbm as lgbm
import sys
from datetime import datetime
import os


model_name = sys.argv[1]
data_name = sys.argv[2]
data = pd.read_csv(os.path.join('data', data_name))
model = lgbm.Booster(model_file=os.path.join('models', model_name))
pred = pd.DataFrame(model.predict(data), columns=['PRED'])

pred_name = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
pred.to_csv(os.path.join('data', f'pred_{pred_name}.csv'))