import pickle
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import make_pipeline


def load_model(model_path):
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model


categorical = ['lead_source']
numeric = ['number_of_courses_viewed', 'annual_income']

# df[categorical] = df[categorical].fillna('NA')
# df[numeric] = df[numeric].fillna(0)
# 
# train_dict = df[categorical + numeric].to_dict(orient='records')
# 
# pipeline = make_pipeline(
#     DictVectorizer(),
#     LogisticRegression(solver='liblinear')
# )
# 
# pipeline.fit(train_dict, y_train)

pipeline = load_model('pipeline_v1.bin')

def predict(df):
    df[categorical] = df[categorical].fillna('NA')
    df[numeric] = df[numeric].fillna(0)

    dicts = df[categorical + numeric].to_dict(orient='records')
    y_pred = pipeline.predict_proba(dicts)[:, 1]

    return y_pred

if __name__ == '__main__':
    test_data = pd.DataFrame([
    {
    "lead_source": "paid_ads",
    "number_of_courses_viewed": 2,
    "annual_income": 79276.0
    }])

    predictions = predict(test_data)
    print(predictions)


