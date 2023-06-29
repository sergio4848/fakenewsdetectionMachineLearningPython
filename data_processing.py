import pandas as pd

def load_and_merge_data():
    # Fake haber veri kümesini yükle
    fake_data = pd.read_csv("fake.csv")
    fake_data["label"] = 1  # Fake haberleri temsil etmek için etiket değeri olarak 1 kullanalım

    # Gerçek haber veri kümesini yükle
    true_data = pd.read_csv("true.csv")
    true_data["label"] = 0  # Gerçek haberleri temsil etmek için etiket değeri olarak 0 kullanalım

    # Veri kümesini birleştirme
    data = pd.concat([fake_data, true_data], ignore_index=True)

    return data
