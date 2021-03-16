
import bentoml,json
import pandas as pd
from bentoml.frameworks.fastai import FastaiModelArtifact
from bentoml.adapters import FileInput,DataframeInput
@bentoml.artifacts([FastaiModelArtifact('learner')])
@bentoml.env(infer_pip_packages=True)
class FastaiTabularModel(bentoml.BentoService):
    
    @bentoml.api(input=DataframeInput(), batch=True)  ## preisa ser True
    def predict(self, df):
        dl = self.artifacts.learner.dls.test_dl(df.iloc[:1,:],rm_type_tfms=None, num_workers=0,with_input=True)
        print("dl.xs= ",dl.xs)
        result  = self.artifacts.learner.get_preds(dl=dl)
        return [[result]]
