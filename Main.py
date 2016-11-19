import genesis as gen
import random as r
import shapes

mutation_model = gen.MutationModel(2)
environment = gen.Environment(1000, 1000)
for i in range(1000):
    environment.create_food(r.random()*1000, r.random()*1000, 100)

body = gen.Body(1000, 100, shapes.Circle(500, 500, 5))
brain = gen.Brain()
legs = gen.Legs()
mouth = gen.Mouth(1, 0)
fission = gen.Fission(mutation_model)

creature = gen.Creature(body, "Genesis")
creature.add_organ(brain)
creature.add_organ(legs)
creature.add_organ(mouth)
creature.add_organ(fission)
environment.queue_creature(creature)

brain._hidden_layer_bias.connect_to_neuron(legs.get_forward_neuron(), 10)
brain._hidden_layer_bias.connect_to_neuron(body._burn_mass_neuron, 10)
brain._hidden_layer_bias.connect_to_neuron(legs.get_turn_clockwise_neuron(), 5)
brain._hidden_layer_bias.connect_to_neuron(mouth._eat_neuron, 1)

environment.start()
