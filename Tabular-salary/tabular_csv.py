
from bentoml import env, api, artifacts, BentoService
from bentoml.frameworks.fastai import Fastai1ModelArtifact
from bentoml.adapters import DataframeInput


@env(pip_packages=['fastai'])
@artifacts([Fastai1ModelArtifact('model')])
class FastaiTabularModel(BentoService):
    
    @api(input=DataframeInput(), batch=True)
    def predict(self, df):
        results = []
        for _, row in df.iterrows():       
            prediction = self.artifacts.model.predict(row)
            results.append(prediction[0].obj)
        return results
