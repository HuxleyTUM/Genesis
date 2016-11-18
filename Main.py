import genesis as gen

mutation_model = gen.MutationModel(2)
environment = gen.Environment(1000, 1000)

body = gen.Body(100, 20)
brain = gen.Brain()
legs = gen.Legs()
mouth = gen.Mouth(0, 10)
fission = gen.Fission(mutation_model)

creature = gen.Creature(body)
creature.add_organ(brain)
creature.add_organ(legs)
creature.add_organ(mouth)
creature.add_organ(fission)
environment.place_creature(creature)
