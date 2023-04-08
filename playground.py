import json
import re
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor

ROOT_PATH = "datasets/access"


class NginxAI(object):
    def __init__(self):
        self.label_encoder = LabelEncoder()
        pass

    def parse_request(self, request_string):
        matches = re.match(
            r"(\S+) - - \[(\S+ \S+)\] \"(\S+) (\S+) (\S+)\" (\S+) (\S+) \"(\S+)\" \"(.+)\"",
            request_string,
        )
        request_dict = {
            "ip_address": matches.group(1),
            "timestamp": matches.group(2),
            "method": matches.group(3),
            "url": matches.group(4),
            "protocol": matches.group(5),
            "status_code": matches.group(6),
            "response_length": matches.group(7),
            "referer": matches.group(8),
            "user_agent": matches.group(9),
        }
        return request_dict

    def load_data(self, name: str, root_path=ROOT_PATH):
        path = os.path.join(root_path, f"{name}.txt")
        json_array = []
        with open(path, "r") as f:
            for line in f:
                json_array.append(self.parse_request(line))
        json_data_str = json.dumps(json_array)
        return pd.read_json(json_data_str, orient="records")

    def run(self):
        df = self.load_data("access")
        df["label"] = df["user_agent"].apply(
            lambda x: 1 if x == "Python/3.10 websockets/10.3" else 0
        )
        # df["label"] = df["ip_address"].apply(
        #     lambda x: 1 if x == "172.70.54.90" else 0
        # )

        label_encoder = LabelEncoder()
        self.label_encoder_values = {}
        categorical_features = [
            "ip_address",
            "timestamp",
            "method",
            "url",
            "protocol",
            "referer",
            "user_agent",
        ]
        for feature in categorical_features:
            df[feature] = label_encoder.fit_transform(df[feature])
            self.label_encoder_values[feature] = label_encoder.classes_

        X_train, X_test, y_train, y_test = train_test_split(
            df.drop("label", axis=1),
            df["label"],
            test_size=0.2,
            random_state=42
        )

        model = RandomForestRegressor(n_estimators=10, random_state=42)

        model.fit(X_train, y_train)

        test_data = pd.DataFrame(
            [
                {
                    "ip_address": "172.74.54.99",
                    "timestamp": "08/Apr/2023:06:56:58 +0000",
                    "method": "POST",
                    "url": "/favicon.ico",
                    "protocol": "HTTP/1.1",
                    "status_code": 200,
                    "response_length": 588,
                    "referer": "-",
                    "user_agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Safari/537.36",
                }
            ]
        )

        for feature in categorical_features:
            test_data[feature] = np.searchsorted(
                self.label_encoder_values[feature], test_data[feature]
            )

        # print(test_data.head())

        prediction = model.predict(test_data)

        print("Prediction: {}".format(prediction[0]))


NginxAI().run()
