from sedtrails.particle_tracer import SeedingConfig, ParticleFactory

# Example usage of the ParticleFactory  to create population of particles
config_random = SeedingConfig(
    {
        'population': {
            'particle_type': 'sand',
            'seeding': {
                'strategy': {'random': {'bbox': '1.0,2.0, 3.0,4.0', 'seed': 42}},
                'quantity': 5,
                'release_start': '2025-06-18 13:00:00',
            },
        }
    }
)

particles = ParticleFactory.create_particles(config_random)
print('Created particles:', particles)
