from __future__ import print_function

import time
import sys
from io import StringIO
import os
import shutil

import argparse
import csv
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Binarizer, StandardScaler, OneHotEncoder

#from sagemaker_containers.beta.framework import (
#    content_types, encoders, env, modules, transformer, worker)

# Since we get a headerless CSV file we specify the column names here.
feature_columns_names = [
    'sex', # M, F, and I (infant)
    'length', # Longest shell measurement
    'diameter', # perpendicular to length
    'height', # with meat in shell
    'whole_weight', # whole abalone
    'shucked_weight', # weight of meat
    'viscera_weight', # gut weight (after bleeding)
    'shell_weight'] # after being dried

feature_columns_names = [
       'credito_coaplicante',
	   'credito_fiador',
       'financiamento_outro_lugar',
	   'tipo_emprego', 
	   'conta_corrente_valor',
       'outros_creditos_aqui',
	   'finance__credits__other_banks',
       'numero_dependentes',
	   'investimentos_valor',
	   'duracao_residencia',
       'valor_solicitado',
	   'tem_telefone', 
	   'duracao_emprego',
       'duracao_credito',
	   'historico_credito',
	   'financiamento_outros_bens',
       'proposito_credito',
	   'tipo_residencia',
	   'receita_disponivel']




feature_columns_dtype = {
       'credito_coaplicante'            : "float64",
	   'credito_fiador'                 : "float64", 
       'financiamento_outro_lugar'      : "float64",
	   'tipo_emprego'                   : "category" ,
	   'conta_corrente_valor'           : "category" ,
       'outros_creditos_aqui'           : "float64",
	   'finance__credits__other_banks'  : "float64",
       'numero_dependentes'             : "float64",
	   'investimentos_valor'            : "category" ,
	   'duracao_residencia'             : "category" ,
       'valor_solicitado'               : "float64",
	   'tem_telefone'                   : "category" ,
	   'duracao_emprego'                : "float64",
       'duracao_credito'                : "float64",
	   'historico_credito'              : "category" ,
	   'financiamento_outros_bens'      : "category" ,
       'proposito_credito'              : "category" ,
	   'tipo_residencia'                : "category" ,
	   'receita_disponivel'             : "float64"
}

label_column = 'deu_default'
label_column_dtype = {'rings': "category"} 

def merge_two_dicts(x, y):
    z = x.copy()   # start with x's keys and values
    z.update(y)    # modifies z with y's keys and values & returns None
    return z

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Sagemaker specific arguments. Defaults are set in the environment variables.
    parser.add_argument('--output-data-dir', type=str)     #, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str)           #, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str)               #, default=os.environ['SM_CHANNEL_TRAIN'])

    args = parser.parse_args()

    # Take the set of files and read them all into a single pandas dataframe
    input_files = [ os.path.join(args.train, file) for file in os.listdir(args.train) ]
    if len(input_files) == 0:
        raise ValueError(('There are no files in {}.\n' +
                          'This usually indicates that the channel ({}) was incorrectly specified,\n' +
                          'the data specification in S3 was incorrectly specified or the role specified\n' +
                          'does not have permission to access the data.').format(args.train, "train"))
    
    raw_data = [ pd.read_csv(
        file, 
        header=None, 
        names=[label_column] + feature_columns_names ,
        dtype=merge_two_dicts(label_column_dtype,feature_columns_dtype, )) for file in input_files ]
    concat_data = pd.concat(raw_data)

    # Labels should not be preprocessed. predict_fn will reinsert the labels after featurizing.
    concat_data.drop(label_column, axis=1, inplace=True)


    numeric_transformer = make_pipeline(
        SimpleImputer(strategy='median'),
        StandardScaler())

    categorical_transformer = make_pipeline(
        SimpleImputer(strategy='constant', fill_value='missing'),
        OneHotEncoder(handle_unknown='ignore'))

    preprocessor = ColumnTransformer(transformers=[
            ("num", numeric_transformer, make_column_selector(dtype_exclude="category")),
            ("cat", categorical_transformer, make_column_selector(dtype_include="category"))])
    
    preprocessor.fit(concat_data)

    joblib.dump(preprocessor, os.path.join(args.model_dir, "model.joblib"))

    print("saved model!")
    
    