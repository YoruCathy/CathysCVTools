import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from skimage import measure
from skimage.draw import ellipsoid

#32*32*32 voxel of shepenet
path = 'voxels.npz'
data = np.load(path)
data = data["arr_0"]

for c, i in enumerate(data):


    verts, faces, normals, values = measure.marching_cubes_lewiner(
        i, 0.1, step_size=1, use_classic=True)

    # Display resulting triangular mesh using Matplotlib. This can also be done
    # with mayavi (see skimage.measure.marching_cubes_lewiner docstring).
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces])
    mesh.set_edgecolor('k')
    ax.add_collection3d(mesh)

    ax.set_xlabel("x-axis")
    ax.set_ylabel("y-axis")
    ax.set_zlabel("z-axis")

    ax.set_xlim(0, 36)  
    ax.set_ylim(0, 36)  
    ax.set_zlim(0, 36) 

    plt.tight_layout()
    plt.show()
