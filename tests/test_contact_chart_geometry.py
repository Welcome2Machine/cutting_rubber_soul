"""Detect interpolation changes that native polygon CCD cannot certify."""
import unittest
import numpy as np
from tools.qualify_contact_transport import cubic_polygon_check


class ChartGeometryTests(unittest.TestCase):
    def test_cubic_crossing_between_safe_polygon_vertices_is_found(self):
        u=np.arange(5.);nodes=np.column_stack((u,[0.,1.,1.,0.,0.]))
        # Cubic overshoots the P1 flat y=1 interval between vertices 1 and 2.
        box=np.array([[1.,1.05],[2.,1.05],[2.,1.3],[1.,1.3]])
        q=cubic_polygon_check(u,nodes,box)
        self.assertGreaterEqual(q['intersection_count'],2)
        self.assertGreater(q['maximum_same_parameter_p1_deviation_mm'],.05)
        self.assertLess(max(v['residual_mm'] for v in q['intersections']),1e-10)

    def test_straight_disjoint_chart_preserves_polygon_geometry(self):
        u=np.arange(4.);nodes=np.column_stack((u,np.zeros(4)))
        box=np.array([[1.,.1],[2.,.1],[2.,.3],[1.,.3]])
        q=cubic_polygon_check(u,nodes,box)
        self.assertEqual(q['intersection_count'],0)
        self.assertLess(q['maximum_same_parameter_p1_deviation_mm'],1e-14)


if __name__=='__main__':unittest.main()
