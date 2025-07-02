import pandas as pd

class Pasien:
    def __init__(self, fitur_dict):
        self.data = fitur_dict

    def to_dataframe(self):
        return pd.DataFrame([self.data])
