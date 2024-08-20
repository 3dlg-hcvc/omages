# import re
# import os
# import sys
# import igl
# import mcubes
# import numpy as np
# from xgutils import nputil, ptutil, sysutil, visutil



# def ptutil_point2index_unittest():
#     points = torch.rand(2000,2) * 2 - 1. #*1.6 - .8
#     rind, grid_points, relative_dist = ptutil.point2index(points, ret_relative=True, grid_dim=4)
#     fig, ax = visutil.newPlot(plotrange=np.array([[-1.,1.],[-1.,1.]]))
#     ax.scatter(points.numpy()[:,0],      points.numpy()[:,1], s=2)
#     ax.scatter(grid_points.numpy()[:,0], grid_points.numpy()[:,1], zorder=4)
#     ax.quiver(grid_points.numpy()[:,0], grid_points.numpy()[:,1], relative_dist.numpy()[:,0], relative_dist.numpy()[:,1], angles='xy', scale_units='xy', scale=1)
#     plt.show()
#     print((grid_points),(points), (relative_dist))


