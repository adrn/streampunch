# Third-party
import astropy.coordinates as coord
import astropy.units as u
import numpy as np

# gala
import gala.integrate as gi
import gala.dynamics as gd
import gala.potential as gp
from gala.units import galactic

__all__ = ['StreamPuncher']


class StreamPuncher:

    @u.quantity_input(t_today=u.Myr)
    def __init__(self, stream, impact_site, external_potential=None,
                 t_today=0*u.Myr, units=galactic):
        """A class for perturbing pre-generated stream particles with a massive
        perturber object.

        Parameters
        ----------
        stream_w0 : `~gala.dynamics.MockStream`
        impact_site_w0 : `~gala.dynamics.PhaseSpacePosition`
        external_potential : `~gala.potential.PotentialBase` subclass (optional)
        units : `~gala.units.UnitSystem` (optional)

        """

        # Validate input
        if not isinstance(stream, gd.MockStream):
            raise TypeError("Input stream particle phase-space positions must "
                            "be a gala.dynamics.MockStream instance, not a "
                            f"{type(stream)}")
        self.stream = stream
        self._n_stream = self.stream.shape[0]

        if not isinstance(impact_site, gd.PhaseSpacePosition):
            raise TypeError("Input impact site phase-space position must be "
                            "a gala.dynamics.PhaseSpacePosition instance, not "
                            f"a {type(impact_site)}")
        self.impact_site = impact_site

        if external_potential is None:
            external_potential = gp.NullPotential(units)
        else:
            external_potential = external_potential.replace_units(units)
        self.external_potential = external_potential

        self.t_today = t_today
        self.units = units

        # Used to cache things, especially the DirectNBody instance so we don't
        # have to recreate thousands of NullPotential's each time!
        self._cache = {}

    def _get_cyl_rotation(self, site_at_impact_w0):
        """Get the rotation matrix to rotate to cylindrical impact coordinates

        Parameters
        ----------
        site_at_impact_w0 : `~gala.dynamics.PhaseSpacePosition`
            The phase-space coordinates of the impact site at the time of
            impact. The impact "site" is the position of the stream closest to
            the perturber at the impact time, *not* the actual location of the
            perturber at the impact time.
        """
        # Define a coordinate system at the time of impact such that the impact
        # site velocity is aligned with the z axis, and the angular momentum is
        # aligned with the x axis:
        L = site_at_impact_w0.angular_momentum()
        v = site_at_impact_w0.v_xyz

        new_z = v / np.linalg.norm(v, axis=0)
        new_x = L / np.linalg.norm(L, axis=0)
        new_y = -np.cross(new_x, new_z)
        R = np.stack((new_x, new_y, new_z))  # rotation matrix

        return R

    def get_perturber_w0_at_impact(self, site_at_impact_w0, b, psi, vz, vpsi):

        # Get the rotation matrix to rotate from Galactocentric to cylindrical
        # impact coordinates at the impact site along the stream
        R = self._get_cyl_rotation(site_at_impact_w0)

        # Define the position of the perturber at the time of impact in the
        # cylindrical impact coordinates:
        perturber_pos = coord.CylindricalRepresentation(rho=b,
                                                        phi=psi,
                                                        z=0*u.pc)
        # z=0 is fixed by definition: This is the impact site

        # Define the velocity in the cylindrical impact coordinates:
        perturber_vel = coord.CylindricalDifferential(
            d_rho=0*u.km/u.s,  # Fixed by definition: b is closest approach
            d_phi=(vpsi / b).to(u.rad/u.Myr, u.dimensionless_angles()),
            d_z=vz)

        # Transform from the cylindrical impact coordinates to Galactocentric
        perturber_rep = perturber_pos.with_differentials(perturber_vel)
        perturber_rep = perturber_rep.represent_as(
            coord.CartesianRepresentation, coord.CartesianDifferential)
        perturber_rep = perturber_rep.transform(R.T)

        pos = perturber_rep.without_differentials() + site_at_impact_w0.pos
        vel = perturber_rep.differentials['s'] + site_at_impact_w0.vel

        # This should be in Galactocentric Cartesian coordinates now!
        return gd.PhaseSpacePosition(pos, vel)

    @u.quantity_input(b=u.pc, psi=u.degree, vz=u.km/u.s, vpsi=u.km/u.s,
                      tau=u.Myr, dt=u.Myr, impact_dist_buffer=u.kpc)
    def run(self, b, psi, vz, vpsi, tau,
            perturber_potential,
            dt=0.25*u.Myr, coarse_dt_factor=4,
            impact_dist_buffer=5*u.kpc):
        """Run the stream perturbation.

        Parameters
        ----------
        b : quantity_like [length]
            Impact parameter.
        psi : quantity_like [angle]
            Local impact-cylindrical coordinate angle.
        vz : quantity_like [speed]
            Relative z velocity in impact-cylindrical coordinates.
        vpsi : quantity_like [speed]
            Relative psi velocity in impact-cylindrical coordinates.
        tau : quantity_like [time]
            Time of impact.
        perturber_potential : `~gala.potential.PotentialBase` subclass
            The potential/density of the perturber.
        dt : quantity_like [time] (optional)
            Timestep used to integrate orbits during the impact.
        coarse_dt_factor : int (optional)
            After the impact, a coarser timestep is used to integrate the orbits
            of the stream particles to ``t_today``. This coarse timestep is
            defined as a multiple of the finer timestep ``dt``.
        impact_dist_buffer : quantity_like [length] (optional)
            The distance over which the impactor is allowed to interact with the
            stream particles. When the impactor is greater than this distance
            from the impact site, the impactor mass is set to 0.

        Returns
        -------
        TODO:

        """

        if tau >= self.t_today:
            raise ValueError("The encounter time, tau, must be before t_today.")

        if b <= 0:
            raise ValueError("The impact parameter, b, must be > 0.")

        # The timestep values are positive, but we negate them below
        dt = np.abs(dt)  # timestep used to resolve the impact
        coarse_dt = dt * coarse_dt_factor  # timestep used after impact
        # MAGIC NUMBER:
        back_dt = 8 * u.Myr  # used when we just need to rewind/fast-forward

        # Compute the orbit of the impact site. The timestep here is arbitrary
        # and is only used to rewind the impact site to the encounter time
        impact_site_orbit = self.external_potential.integrate_orbit(
            self.impact_site, dt=-back_dt, t1=self.t_today, t2=tau,
            Integrator=gi.DOPRI853Integrator)
        assert np.isclose(impact_site_orbit.t[-1], tau)
        site_at_impact_w0 = impact_site_orbit[-1]

        # Determine how much of a time buffer to use around the impact:
        rel_v = np.sqrt(vz**2 + vpsi**2)
        t_buffer = (impact_dist_buffer / rel_v).to(u.Myr)
        t_buffer = int(t_buffer / dt) * dt  # round to factor of dt

        # Times:
        t1_impact = tau - t_buffer
        t2_impact = tau + t_buffer

        # Phase-space position of the perturber at the time of impact
        perturber_at_impact_w0 = self.get_perturber_w0_at_impact(
            site_at_impact_w0, b, psi, vz, vpsi)

        # Integrate the perturber position to today. Again, the timestep here is
        # arbitrary and is only used to integrate the perturber position to
        perturber_tmp_orbit = self.external_potential.integrate_orbit(
            perturber_at_impact_w0, t1=tau, t2=t1_impact, dt=-back_dt,
            Integrator=gi.DOPRI853Integrator)
        assert np.isclose(perturber_tmp_orbit.t[-1], t1_impact)
        perturber_init_w0 = perturber_tmp_orbit[-1]

        # Check that most of the stream particles were released after the impact
        # TODO:
        # stream_w0

        # Rewind the stream particles to before the impact
        stream_tmp_orbits = self.external_potential.integrate_orbit(
            self.stream, t1=self.t_today, t2=t1_impact, dt=-back_dt,
            Integrator=gi.DOPRI853Integrator)

        E = stream_tmp_orbits[:, 0].energy()
        dE = np.abs((E[1:] - E[0]) / E[0])
        assert np.all(dE < 1e-8)
        assert np.isclose(stream_tmp_orbits.t[-1], t1_impact)
        stream_init_w0 = stream_tmp_orbits[-1]

        # Combine the phase-space positions of the perturber and stream
        # particles at the time before the impact
        all_w0_past = gd.combine((perturber_init_w0, stream_init_w0))

        # Figure out time array - need to do this manually because I think there
        # is a bug in gala DirectNBody:
        t_interaction = np.arange(t1_impact.to_value(u.Myr),
                                  t2_impact.to_value(u.Myr)+1e-6,
                                  dt.to_value(u.Myr)) * u.Myr
        assert np.all(np.diff(t_interaction) > 0)

        # Now that we have the particle positions a little before the time of
        # impact, we forward-simulate as an N-body system with the perturber.
        # First, we set up the NBody object (or load from cache):
        if 'nbody' in self._cache:
            nbody = self._cache['nbody']
            nbody.particle_potentials[0] = perturber_potential
            nbody.w0 = all_w0_past
        else:
            ppots = ([perturber_potential] +
                     [gp.NullPotential(self.units)] * self._n_stream)
            nbody = gd.DirectNBody(all_w0_past, particle_potentials=ppots,
                                   units=self.units,
                                   external_potential=self.external_potential,
                                   save_all=True)
            self._cache['nbody'] = nbody
        orbits_impact = nbody.integrate_orbit(t=t_interaction)

        # Ensure that the final time is t_today:
        t_post = np.arange(orbits_impact.t[-1].to_value(u.Myr),
                           self.t_today.to_value(u.Myr),
                           coarse_dt.to_value(u.Myr)) * u.Myr
        if not np.isclose(t_post[-1], self.t_today):
            t_post = np.append(t_post, self.t_today)

        orbits_post = self.external_potential.integrate_orbit(
            orbits_impact[-1], t=t_post,
            Integrator=gi.DOPRI853Integrator)

        # Downsample the orbits during impact
        orbits_impact = orbits_impact[::coarse_dt_factor]
        orbits_post = orbits_post[1:]

        new_data = coord.concatenate_representations((orbits_impact.data,
                                                      orbits_post.data))
        new_t = np.concatenate((orbits_impact.t, orbits_post.t))
        new_orbit = gd.Orbit(new_data.xyz, new_data.differentials['s'].d_xyz,
                             t=new_t)
        return new_orbit
