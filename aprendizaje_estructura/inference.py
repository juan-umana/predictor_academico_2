import pickle

# Read model from PKL file 
filename='C:/Users/jd.umana10/Documents/GitHub/predictor_academico_2/aprendizaje_estructura/model.pkl'
file = open(filename, 'rb')
model_success = pickle.load(file)
file.close()


# Print model 
print(model_success)

# Check_model check for the model structure and the associated CPD and returns True if everything is correct otherwise throws an exception
print(model_success.check_model())

# Infering the posterior probability
from pgmpy.inference import VariableElimination

infer = VariableElimination(model_success)

evidence = {'gdp': 2}

posterior_p = infer.query(["target"], evidence=evidence)
print(posterior_p)
