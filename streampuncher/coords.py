# Third-party
import numpy as np
import gala.dynamics as gd

__all__ = ['align_stream_with_point']


def align_stream_with_point(stream, w0):
    """Align stream particle positions with the input position.

    The returned coordinates have a z-axis aligned with the angular momentum of
    the input point, and an x-axis aligned with the velocity of the input point.

    Parameters
    ----------
    stream : `~gala.dynamics.MockStream`
    w0 : `~gala.dynamics.PhaseSpacePosition`

    Returns
    -------
    aligned_stream : `~gala.dynamics.PhaseSpacePosition`
    """
    L = w0.angular_momentum()
    v = w0.v_xyz

    new_x = v / np.linalg.norm(v, axis=0)
    new_z = L / np.linalg.norm(L, axis=0)
    new_y = -np.cross(new_x, new_z)
    R = np.stack((new_x, new_y, new_z))  # rotation matrix

    tmp = (stream.pos - w0.pos).transform(R)

    return gd.PhaseSpacePosition(tmp, stream.vel)
