import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH

cs = CS.ConfigurationSpace()

a = CSH.CategoricalHyperparameter('a', choices=['red', 'green', 'blue','d','e','f'])
b = CSH.CategoricalHyperparameter('b', choices=['red', 'green', 'blue','d','e','f'])
cs.add_hyperparameters([a, b])
print(cs.sample_configuration().get_dictionary())
print(cs.sample_configuration().get_dictionary())
print(cs.sample_configuration().get_dictionary())
print(cs.sample_configuration().get_dictionary())
print(cs.sample_configuration().get_dictionary())
print(cs.sample_configuration().get_dictionary())
print(cs.sample_configuration().get_dictionary())
