import pandas as pd
import json
import bentoml
from bentoml.frameworks.sklearn import SklearnModelArtifact
#from bentoml.service.artifacts.common import PickleArtifact
#from bentoml.handlers import DataframeHandler
from bentoml.adapters import DataframeInput, FileInput

@bentoml.artifacts([
                    SklearnModelArtifact("model_a"),
                    SklearnModelArtifact("ml")
                    ])
@bentoml.env(pip_packages=["scikit-learn", "pandas"])
class CreditPrediction(bentoml.BentoService):

    @bentoml.api(input=DataframeInput(), batch=True)
    def predict(self, df):
        """
        predict expects pandas.Series as input
        """  
        # print("***********")
        # print("df", df.iloc[0,:].T)
        # print("***********")
        #print("colunas", df.columns)
        df = self.artifacts.model_a.transform(df)
        print(df)
        return self.artifacts.ml.predict(df)
