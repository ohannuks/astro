h = 6.62607015e-34  # Planck constant in J*s
c = 2.99792458e8    # Speed of light in m/s
k_B = 1.380649e-23  # Boltzmann constant in J
sigma = 5.670374419e-8  # Stefan-Boltzmann constant in W/m^2/K^4
# Unit conversions
log_nm = jnp.log(1e-9)  # Convert from nanometers to meters
log_solar_radius = jnp.log(6.957e8)  # Solar radius in meters
log_solar_luminosity = jnp.log(3.828)+26*jnp.log(10)  # Solar luminosity in Watts

