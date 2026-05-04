import potpourri3d as pp3d
import torch


class InletGeodesics(object):
    """Compute the shortest geodesic distances from each vertex to the vessel inlet.

    Args:
        -
    """

    def __call__(self, data):
        solver = pp3d.MeshHeatMethodDistanceSolver(data.pos.numpy(), data.face.t().numpy())

        # Compute the minimum geodesic distances to the inlet
        assert hasattr(data, "inlet_index")

        inlet = data['inlet_index'].numpy()


        geodesics = solver.compute_distance_multisource(inlet)

        # Append the features in single precision
        data.geo = torch.from_numpy(geodesics).float()

        return data

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)
