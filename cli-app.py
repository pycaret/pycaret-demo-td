# input parameters at command line
import time
import sys
module = sys.argv[1]
dataset = sys.argv[2]
target = sys.argv[3]
exp_name = str(dataset) + '_exp'

# import dataset
from pycaret.datasets import get_data
data = get_data(dataset)

#initialize setup

t0 = time.time()

if module == 'classification':
    from pycaret.classification import setup, compare_models, automl, save_model
    setup(data, target = target, silent = True, html = False, log_experiment=True, experiment_name=exp_name)
    best_model = compare_models()
    model = automl()
    save_model(model, model_name = 'pycaret-clf-best')

elif module == 'regression':
    from pycaret.regression import setup, compare_models, automl, save_model
    setup(data, target = target, silent = True, html = False, log_experiment=True, experiment_name=exp_name)
    best_model = compare_models()
    model = automl()
    save_model(model, model_name = 'pycaret-reg-best')

elif module == 'clustering':
    from pycaret.clustering import setup, create_model, save_model
    setup(data, normalize = True, silent = True, html = False, log_experiment=True, experiment_name=exp_name)
    model = create_model('kmeans')
    save_model(model, model_name = 'pycaret-clu-kmeans')

elif module == 'anomaly':
    from pycaret.anomaly import setup, create_model, save_model
    setup(data, normalize = True, silent = True, html = False, log_experiment=True, experiment_name=exp_name)
    model = create_model('iforest')
    save_model(model, model_name = 'pycaret-clu-iforest')

elif module == 'nlp':
    from pycaret.nlp import setup, create_model, save_model
    setup(data, target = target, html = False, log_experiment=True, experiment_name=exp_name)
    model = create_model('lda', multi_core = True)
    save_model(model, model_name = 'pycaret-nlp-lda')

t1 = time.time()
tt = round(t1-t0,2)

print('Process Successfully Finished in {} seconds'.format(tt))