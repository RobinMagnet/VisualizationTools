import numpy as np
import meshplot as mp
import matplotlib.colors
from . import utils


def plot(mesh1, cmap=None, points=None, cmap_p=None, rotation=None, colormap='viridis',
         shading=None, shading_p=None, flat=False, pretty=True, point_size=None):

    shading_m = {}
    if pretty:
        shading_m = utils.get_smooth_shading(flat=flat)
    if shading is not None:
        shading_m.update(shading)

    if cmap is None:
        cmap = utils.base_cmap(mesh1.n_vertices, pc=(mesh1.n_faces == 0))
    elif cmap.ndim == 1:  # Scalar quantity
        cmap = utils.get_cmap(cmap, colormap=colormap)

    vertices = utils.rotate(mesh1.vertlist, rotation=rotation)
    faces = mesh1.facelist

    p = mp.plot(vertices, faces, c=cmap, shading=shading_m)

    if points is not None:
        if shading_p is None:
            shading_p = {}
        if point_size is not None:
            shading_p['point_size'] = point_size

        elif 'point_size' not in shading_p.keys():
            mesh_area = mesh1.area
            if mesh_area is not None:
                shading_p['point_size'] = .1*np.sqrt(mesh_area)

        if type(points) is list or np.issubdtype(type(points), np.integer) or (points.ndim == 1):
            points = vertices[points]
            if points.ndim == 1:
                points = points[None, :]
        else:
            points = utils.rotate(points, rotation=rotation)

        if cmap_p is not None and cmap_p.ndim == 1:
            cmap_p = utils.get_cmap(cmap_p, colormap=colormap)

        p.add_points(points, c=cmap_p, shading=shading_p)

    return p


def plot_corr(mesh1, mesh2, p2p, n_points=10, pretty=True, param=[-2, -1, 3], dist_ratio=1, rotation=None,
              rev=False, shading=None, flat=False):

    cmap1 = utils.vert2rgb(utils.rotate(mesh1.vertlist, rotation=rotation),
                           pretty=pretty, param=param)

    shading_m = {}
    if pretty:
        shading_m = utils.get_smooth_shading(flat=flat)
    if shading is not None:
        shading_m.update(shading)

    if rev:
        cmap1 = 1 - cmap1

    if p2p.ndim == 1:
        cmap2 = cmap1[p2p]
    else:
        assert p2p.shape == (mesh2.n_vertices, mesh1.n_vertices), 'Pb with p2p dimension'
        cmap2 = p2p @ cmap1

    if mesh1.vertex_areas is not None:
        cm1 = np.average(mesh1.vertlist, axis=0, weights=mesh1.vertex_areas)
    else:
        cm1 = mesh1.mean(axis=0)

    if mesh2.vertex_areas is not None:
        cm2 = np.average(mesh2.vertlist, axis=0, weights=mesh2.vertex_areas)
    else:
        cm2 = mesh2.mean(axis=0)

    vert1 = mesh1.vertlist - cm1[None,:]
    vert1 = utils.rotate(vert1, rotation=rotation)

    vert2 = mesh2.vertlist - cm2[None,:]
    vert2 = utils.rotate(vert2, rotation=rotation)

    xmin1, xmax1 = vert1[:, 0].min(),  vert1[:, 0].max()
    xmin2 = vert2[:, 0].min()

    # transl_x = xmin1 - xmin2 + xmax1 + (xmax1-xmin1)/dist_ratio
    transl_x = xmax1 - xmin2 + (xmax1-xmin1)/dist_ratio
    transl = np.array([transl_x, 0, 0])

    vert2 += transl[None, :]

    fps2 = mesh2.extract_fps(n_points, geodesic=False)

    p = mp.plot(vert1, mesh1.facelist, c=cmap1, shading=shading_m)
    p.add_mesh(vert2, mesh2.facelist, c=cmap2, shading=shading_m)

    for ind in fps2:
        p.add_lines(vert1[p2p[ind]], vert2[ind],
                    shading={"line_color": matplotlib.colors.to_hex(cmap2[ind])})
