import yaml
import torch
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt
import mcubes
import trimesh
import pickle
import torch
import os
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import Rbf
from numpy.polynomial.polynomial import Polynomial, polyfit
import plotly.tools as tls
import plotly.io as pio
import plotly.graph_objs as go


base_color = np.array(
    [[255,   0,  0, 255],  # Red
    [  0, 255,   0, 255],  # Green
    [  0,   0, 255, 255],  # Blue
    [255, 255,   0, 255],  # Yellow
    [  0, 255, 255, 255],  # Cyan
    [255,   0, 255, 255],  # Magenta
    [255, 165,   0, 255],  # Orange
    [128,   0, 128, 255],  # Purple
    [255, 192, 203, 255],  # Pink
    [128, 128, 128, 255],  # Gray
    [210, 245, 60, 255], # Lime
    [170, 110, 40, 255], # Brown
    [128, 0, 0, 255], # Maroon
    [0, 128, 128, 255], # Teal
    [0, 0, 128, 255], # Navy
    ],
    dtype=np.uint8
)

def load_model(model_class, config_path, pretrained_model_path=None, exclude_prefix='vae_model.'):
    if config_path is None:
        return None
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    model = model_class(**config['model_params'])
    
    if pretrained_model_path is not None:
        state_dict = torch.load(pretrained_model_path, map_location='cpu')['state_dict']
        # modify the key names
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith(exclude_prefix):
                new_state_dict[k[len(exclude_prefix):]] = v
            else:
                new_state_dict[k] = v
        model.load_state_dict(new_state_dict, strict=True)
    
    return model

def load_model_v2(model, pretrained_model_path, exclude_prefix='vae_model.', remove_prefix='vae_model.'):
    state_dict = torch.load(pretrained_model_path, map_location='cpu')['state_dict']
    # modify the key names
    new_state_dict = {}
    for k, v in state_dict.items():
        if remove_prefix is not None:
            if k.startswith(remove_prefix):
                continue
        if k.startswith(exclude_prefix):
            new_state_dict[k[len(exclude_prefix):]] = v
        else:
            new_state_dict[k] = v
    model.load_state_dict(new_state_dict, strict=True)
    
    return model


def draw_cuboid(ax, origin, size, color):
    # Unpack the origin and size
    x, y, z = origin
    dx, dy, dz = size

    # Define vertices of the cuboid
    vertices = np.array([[x, y, z], [x+dx, y, z], [x+dx, y+dy, z], [x, y+dy, z],
                         [x, y, z+dz], [x+dx, y, z+dz], [x+dx, y+dy, z+dz], [x, y+dy, z+dz]])

    # Define the six faces of the cuboid
    faces = [[vertices[j] for j in [0, 1, 2, 3]],
             [vertices[j] for j in [4, 5, 6, 7]], 
             [vertices[j] for j in [0, 3, 7, 4]], 
             [vertices[j] for j in [1, 2, 6, 5]], 
             [vertices[j] for j in [0, 1, 5, 4]], 
             [vertices[j] for j in [2, 3, 7, 6]]]

    # Create a 3D polygon collection and add to axis
    ax.add_collection3d(Poly3DCollection(faces, facecolors='cyan', linewidths=1, edgecolors=color, alpha=0.0))


class Brep:
    def __init__(self) -> None:
        self.faces = []
        self.boundaries = set()

    def add_face(self, face):
        self.faces.append(face)
        face.parent_brep = self
    
    def add_boundary(self, boundary):
        self.boundaries.add(boundary)
    
    def export(self, split_boundary=True):
        boundaries_list = []
        boundaries_instance_list = []
        faces_list = []
        for boundary in self.boundaries:
            curr_b = boundary.export(split_into_groups=split_boundary)
            boundaries_list.extend(curr_b)
            boundaries_instance_list.extend([boundary for i in range(len(curr_b))])
        for face in self.faces:
            face_mesh = face.export()
            boundary_idx = []
            for i in range(len(boundaries_instance_list)):
                if boundaries_instance_list[i] in face.boundaries:
                    boundary_idx.append(i)
            face_mesh['boundaries'] = boundary_idx
            faces_list.append(face_mesh)
        
        return {'boundaries': boundaries_list, 'faces': faces_list}

    def visualize(self, save_path, type='matplotlib'):
        if type == 'matplotlib':
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_xlim(0,64)
            ax.set_ylim(0,64)
            ax.set_zlim(0,64)
            for face in self.faces:
                loops = face.get_close_loop_boundary()
                new_loops = []
                for loop in loops:
                    new_loops.append(loop[::-1])
                poly = Poly3DCollection(new_loops, alpha=.2, facecolors=plt.cm.jet(np.random.rand(1)), edgecolors='k')
                #plt.cm.jet(np.random.rand(1)))
                ax.add_collection3d(poly)
        
            for bd in self.boundaries:
                #for func in bd.parametric_functions:
                #    px, py, pz, t = func
                #    x = px(t)
                #    y = py(t)
                #    z = pz(t)
                #    ax.plot(x, y, z, linewidth=10, c='black')

                corners = bd.vertices[bd.vertices_type == 1]
                ax.scatter(corners[:, 0], corners[:, 1], corners[:, 2], c='red', s=10)
        
            plt.savefig(save_path, dpi=300)
        elif type == 'plotly':
            fig = go.Figure()
            for face in self.faces:
                loops = face.get_close_loop_boundary()
                for loop in loops:
                    # random color
                    color = (np.random.rand(3) * 255).astype(np.uint8)
                    fig.add_trace(go.Mesh3d(
                        x=loop[:, 0], y=loop[:, 1], z=loop[:, 2], 
                        color='rgba({},{},{})'.format(color[0], color[1], color[2]),
                        opacity=0.5
                        )
                    )
                    # add edges
                    edges = go.Scatter3d(
                        x=loop[:, 0], y=loop[:, 1], z=loop[:, 2], mode='lines', line=dict(color='black', width=5)
                    )
                    fig.add_trace(edges)
            for bd in self.boundaries:
                corners = bd.vertices[bd.vertices_type == 1]
                fig.add_trace(go.Scatter3d(
                    x=corners[:, 0], y=corners[:, 1], z=corners[:, 2], mode='markers', marker=dict(size=10, color='red')
                ))
            fig.update_layout(
                scene=dict(
                    xaxis=dict(nticks=10, range=[0, 64]),  # Set x-axis range
                    yaxis=dict(nticks=10, range=[0, 64]),  # Set y-axis range
                    zaxis=dict(nticks=10, range=[0, 64])   # Set z-axis range
                )
            )

            pio.write_html(fig, save_path.replace('.png', '.html'))

    
class Face:
    def __init__(self, id_, mesh) -> None:
        self.id = id_
        self.boundaries = []
        self.mesh = mesh
        # add color
        mesh.visual.vertex_colors = np.ones((len(mesh.vertices), 4)) * base_color[id_ % len(base_color)]
        self.parent_brep = None
        self.parametric_function = None
    
    def add_boundary(self, boundary):
        self.boundaries.append(boundary)
        boundary.parent_faces.append(self)
    
    def compute_parametric_function(self):
        x = self.mesh.vertices[:,0]
        y = self.mesh.vertices[:,1]
        z = self.mesh.vertices[:,2]

        bd_points = []
        for bd in self.boundaries:
            bd_points.append(bd.get_parametric_points())
        bd_points = np.concatenate(bd_points, axis=0)
        x_important = bd_points[:, 0]
        y_important = bd_points[:, 1]
        z_important = bd_points[:, 2]

        x_combined = np.concatenate([x, x_important])
        y_combined = np.concatenate([y, y_important])
        z_combined = np.concatenate([z, z_important])

        weights = np.ones(len(x_combined))
        weights[-len(x_important):] = 1000

        coeff_x = polyfit(x_combined, np.zeros_like(x_combined), 2, w=weights)

    def get_close_loop_boundary(self):
        # get the close loop boundary
        # there might be multiple loops
        edges = []
        for bd in self.boundaries:
            for func in bd.parametric_functions:
                px, py, pz, t = func
                x = px(t)
                y = py(t)
                z = pz(t)
                if z.shape[0] == 0:
                    continue
                edges.append(np.vstack([x, y, z]).T)
        
        # find the closest loop
        loops = []
        if len(edges) == 0:
            return loops
        used_mark = [False for _ in range(len(edges))]
        loop = edges[0]
        used_mark[0] = True
        while True:
            found_new_edge = False
            for i in range(len(edges)):
                if used_mark[i]:
                    continue
                check_edge_pts = edges[i] # N, 3
                if np.linalg.norm(loop[0] - check_edge_pts[0]) < 1e-3:
                    loop = np.vstack([check_edge_pts[::-1], loop])
                    used_mark[i] = True
                    found_new_edge = True
                elif np.linalg.norm(loop[0] - check_edge_pts[-1]) < 1e-3:
                    loop = np.vstack([check_edge_pts, loop])
                    used_mark[i] = True
                    found_new_edge = True
                elif np.linalg.norm(loop[-1] - check_edge_pts[0]) < 1e-3:
                    loop = np.vstack([loop, check_edge_pts])
                    used_mark[i] = True
                    found_new_edge = True
                elif np.linalg.norm(loop[-1] - check_edge_pts[-1]) < 1e-3:
                    loop = np.vstack([loop, check_edge_pts[::-1]])
                    used_mark[i] = True
                    found_new_edge = True
                if found_new_edge:
                    break
            if found_new_edge:
                if np.linalg.norm(loop[0] - loop[-1]) < 1e-3:
                    # check if the trajectory distance is too short
                    if np.linalg.norm(np.diff(loop, axis=0), axis=1).sum() >= 5:
                        loops.append(loop)
                    start_new_loop = False
                    for i in range(len(edges)):
                        if used_mark[i]:
                            continue
                        loop = edges[i]
                        used_mark[i] = True
                        start_new_loop = True
                        break
                    if not start_new_loop:
                        break
            else:
                if np.linalg.norm(loop[0] - loop[-1]) < 0.05:
                    # check if the trajectory distance is too short
                    if np.linalg.norm(np.diff(loop, axis=0), axis=1).sum() >= 5:
                        raise ValueError('Cannot find the next edge')
                    start_new_loop = False
                    for i in range(len(edges)):
                        if used_mark[i]:
                            continue
                        loop = edges[i]
                        used_mark[i] = True
                        start_new_loop = True
                        break
                    if not start_new_loop:
                        break
                else:
                    #raise ValueError('Cannot find the next edge')
                    print('Cannot find the next edge')
                    break

        return loops
                        
    def export(self):
        return {'vertices': self.mesh.vertices, 'triangle': self.mesh.triangles}

    def visualize(self, ax=None):
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_xlim(0,64)
            ax.set_ylim(0,64)
            ax.set_zlim(0,64)
        ax.scatter(self.mesh.vertices[:, 0], self.mesh.vertices[:, 1], self.mesh.vertices[:, 2])
        return ax
        
class Boundary:
    def __init__(self) -> None:
        self.vertices = np.array([[],[],[]]).T
        self.vertices_type = np.array([], dtype=int)
        self.connections = np.array([[],[]], dtype=int).T
        self.groups = None
        self.parent_faces = []
        self.parametric_functions = []

    def add_vertex(self, vertex, v_type=0):
        if len(self.vertices) == 0:
            self.vertices = np.array([vertex])
            self.vertices_type = np.array([v_type])
            return 0
        dist = np.linalg.norm(self.vertices - vertex, axis=1)
        if np.min(dist) > 1e-6:
            self.vertices = np.vstack([self.vertices, vertex])
            self.vertices_type = np.hstack([self.vertices_type, v_type])
            return len(self.vertices) - 1
        else:
            return np.argmin(dist)

    def add_connection(self, connection):
        ### return edge idx
        # check if the edge is already in the list
        connection = sorted(connection)
        if len(self.connections) == 0:
            self.connections = np.array([connection])
            return 0
        dist = np.linalg.norm(self.connections - connection, axis=1)
        if np.min(dist) > 1e-6:
            self.connections = np.vstack([self.connections, connection])
            return len(self.connections) - 1
        else:
            return np.argmin(dist)
    
    def split_into_edges(self):
        v_belongs = [-1 for _ in range(len(self.vertices))]
        groups = []
        for connection_i in range(len(self.connections)):
            v0, v1 = self.connections[connection_i]
            if v_belongs[v0] == -1 and v_belongs[v1] == -1:
                groups.append([v0, v1])
                v_belongs[v0] = len(groups) - 1
                v_belongs[v1] = len(groups) - 1
            elif v_belongs[v0] == -1:
                v_belongs[v0] = v_belongs[v1]
                group = groups[v_belongs[v1]]
                v1_loc = group.index(v1)
                assert v1_loc == 0 or v1_loc == len(group) - 1
                if v1_loc == 0:
                    group.insert(0, v0)
                else:
                    group.append(v0)
            elif v_belongs[v1] == -1:
                v_belongs[v1] = v_belongs[v0]
                group = groups[v_belongs[v0]]
                v0_loc = group.index(v0)
                assert v0_loc == 0 or v0_loc == len(group) - 1
                if v0_loc == 0:
                    group.insert(0, v1)
                else:
                    group.append(v1)
            else:
                if v_belongs[v0] != v_belongs[v1]:
                    group0 = groups[v_belongs[v0]]
                    group1 = groups[v_belongs[v1]]
                    v0_loc = group0.index(v0)
                    v1_loc = group1.index(v1)
                    assert v0_loc == 0 or v0_loc == len(group0) - 1
                    assert v1_loc == 0 or v1_loc == len(group1) - 1
                    if v0_loc == 0:
                        group0[:] = group0[::-1]
                    if v1_loc == len(group1) - 1:
                        group1[:] = group1[::-1]
                    # now v0 is the last element of group0, 
                    # v1 is the first element of group1
                    group0.extend(group1)
                    removed_group_idx = v_belongs[v1]
                    groups.pop(removed_group_idx)
                    for v_idx in group1:
                        v_belongs[v_idx] = v_belongs[v0]
                    for i in range(len(v_belongs)):
                        if v_belongs[i] > removed_group_idx:
                            v_belongs[i] -= 1
        
        # filter some bad groups
        filtered_groups = []
        for g in groups:
            if self.vertices_type[g[0]] != 1 or self.vertices_type[g[-1]] != 1:
                continue
            #if len(g) < 15:
            #    continue
            filtered_groups.append(g)
        groups = filtered_groups
        self.groups = groups

        return len(self.groups)
                    
    def export(self, split_into_groups=True, filename=None):
        groups = []
        if split_into_groups:
            if self.groups is None:
                self.split_into_edges()
            for i in range(len(self.groups)):
                group = self.groups[i]
                vertices = self.vertices[group]
                groups.append(vertices)
        else:
            groups = [self.vertices]
        if filename is not None:
            for i in range(len(groups)):
                group = groups[i]
                pc = trimesh.points.PointCloud(group)
                pc.export(f'{filename}_{i}.obj')
        return groups

    def visualize_vertices(self, filename, split_into_groups=False):
        if split_into_groups:
            assert filename.endswith('.obj') is False
            if self.groups is None:
                self.split_into_edges()
            for i in range(len(self.groups)):
                group = self.groups[i]
                vertices = self.vertices[group]
                pc = trimesh.points.PointCloud(vertices)
                pc.colors = np.ones((len(vertices), 4)) * trimesh.visual.random_color()
                pc.export(f'{filename}_{i}.obj', include_color=True)
            return
        pc = trimesh.points.PointCloud(self.vertices)
        pc.colors = np.ones((len(self.vertices), 4)) * trimesh.visual.random_color()
        pc.export(filename, include_color=True)
    
    def compute_parametric_functions(self):
        # compute the parametric functions for each group
        for group in self.groups:
            out_ = self.compute_parametric_function(group)
            if out_ is not None:
                self.parametric_functions.append(out_)
        
    def compute_parametric_function(self, group):
        # compute the parametric function for the group
        # group: list of vertex idx
        x = self.vertices[group][:,0]
        y = self.vertices[group][:,1]
        z = self.vertices[group][:,2]
        weights = np.ones(len(x))
        weights[0] = weights[-1] = 1000

        t = np.zeros(x.shape[0])
        for i in range(1, t.shape[0]):
            t[i] = t[i-1] + np.linalg.norm(self.vertices[group[i]] - self.vertices[group[i-1]])

        # Fit polynomials to the data
        # check to use deg=1 or 2
        
        last_error = 10000
        for deg in [1, 3]:
            px = Polynomial.fit(t, x, deg=deg, w=weights)
            py = Polynomial.fit(t, y, deg=deg, w=weights)
            pz = Polynomial.fit(t, z, deg=deg, w=weights)
            x_fit, y_fit, z_fit = px(t), py(t), pz(t)
            error = np.mean((x - x_fit)**2 + (y - y_fit)**2 + (z - z_fit)**2)
            if error > last_error / 10 or error < 5:
                break
            else:
                last_error = error

        t = np.linspace(t[0], t[-1], int(t[-1] * 10))
        if t.shape[0] == 0:
            return None

        return [px, py, pz, t]

    def get_parametric_points(self):
        points = []
        for func in self.parametric_functions:
            px, py, pz, t = func
            x = px(t)
            y = py(t)
            z = pz(t)
            points.append(np.vstack([x, y, z]).T)
        return np.concatenate(points, axis=0)
    
def clean_bdf(f_bdf, threshold=0.08):
    valid_idx = []
    for i in range(f_bdf.shape[-1]):
        if np.sum(np.abs(f_bdf[..., i]) < threshold) > 0:
            valid_idx.append(i)
    return f_bdf[..., valid_idx]

def brep_process(v_sdf, f_bdf, save_base):
    # v_sdf: nxnxn
    # f_bdf: nxnxnxm

    if isinstance(v_sdf, torch.Tensor):
        v_sdf = v_sdf.cpu().numpy()
    if isinstance(f_bdf, torch.Tensor):
        f_bdf = f_bdf.cpu().numpy()
    #f_bdf = clean_bdf(f_bdf)

    vertices, triangles = mcubes.marching_cubes(v_sdf, 0)
    triangles = triangles[:, [2, 1, 0]]
    mesh = trimesh.Trimesh(vertices=vertices, faces=triangles, process=False)

    mesh.export(f'{save_base}/mc_mesh_ori.obj')
    mesh.export(f'{save_base}/mc_mesh_ori.glb')

    vertices, triangles = trimesh.remesh.subdivide_loop(vertices, triangles.astype(int), 1)
    mesh = trimesh.Trimesh(vertices=vertices, faces=triangles, process=False)

    mesh.export(f'{save_base}/mc_mesh_sub.obj')
    mesh.export(f'{save_base}/mc_mesh_sub.glb')

    solid_pc = trimesh.points.PointCloud(vertices)
    solid_pc.export(f'{save_base}/mc_vertice.obj', include_color=True)

    all_intersection_points = []

    grid_reso = f_bdf.shape[0]

    filter_v = {} # used for filter the vertices that are not in the mesh. k -> face_id, v -> v_id
    assign_v = {} # used for assign the vertices to the face. k -> v_id, v -> face_id
    compute_number = 0
    while compute_number < 60:
        compute_number += 1
        f_dbf_interpolator = RegularGridInterpolator(
            (np.arange(grid_reso), np.arange(grid_reso), np.arange(grid_reso)), f_bdf, 
            bounds_error=False, fill_value=0)

        interpolated_f_bdf = f_dbf_interpolator(vertices) # v, num_faces
        for k, v in filter_v.items():
            interpolated_f_bdf[v, k] = 100
        for k, v in assign_v.items():
            interpolated_f_bdf[k, v] = 0

        vertices_face_id = interpolated_f_bdf.argmin(-1)
        triangle_face_id = vertices_face_id[triangles] # M, 3

        mesh = trimesh.Trimesh(vertices=vertices, faces=triangles, process=False)
        mesh.visual.vertex_colors = base_color[vertices_face_id % len(base_color)]
        mesh.export(f'{save_base}/{compute_number}.obj', include_color=True)

        # if there is vertices that doesn't belong to any face, set it to one of the neighbor
        # TODO
        invalid_v = np.where((interpolated_f_bdf == 100).all(-1))[0]
        if len(invalid_v) > 0:
            print(f'Find {len(invalid_v)} vertices that doesn\'t belong to any face')
            break

        # check if some faces are not valid
        # rule 1: face should have at least 1 triangle that vertices all belong to the its
        same_face_id_triangle = (triangle_face_id == triangle_face_id[:, 0:1]).all(1)
        mesh_triangle_id = triangle_face_id[same_face_id_triangle][:,0]
        re_compute = False
        for face_id in range(f_bdf.shape[-1]):
            count_ = (mesh_triangle_id == face_id).sum()
            if count_ == 0:
                print(f'Face {face_id} has no valid triangle, remove it')
                f_bdf = np.delete(f_bdf, face_id, axis=-1)
                re_compute = True
                filter_v = {} # recompute the filter
                break
        if re_compute:
            continue
        # rule 2: face mesh should be all connected
        for face_id in range(f_bdf.shape[-1]):
            triangle_belong_face = (triangle_face_id == face_id).any(1) # any / all
            triangle_idx = np.arange(len(triangles))[triangle_belong_face]
            face_mesh = mesh.submesh([triangle_idx], append=True)
            components = face_mesh.split(only_watertight=False)
            if len(components) > 1:
                # mark small faces parts shouldn't belong to the face
                max_v_num = max([len(c.vertices) for c in components])
                for c in components:
                    if len(c.vertices) < max_v_num:
                        marked_v_idx = np.where((vertices == c.vertices[:, None]).all(-1))[1]
                        for v_idx in marked_v_idx:
                            # remove the assign_v
                            if v_idx in assign_v:
                                assign_v.pop(v_idx)
                        if face_id not in filter_v:
                            filter_v[face_id] = marked_v_idx
                        else:
                            filter_v[face_id] = np.concatenate([filter_v[face_id], marked_v_idx])
                re_compute = True
            if re_compute:
                break
        if re_compute:
            continue
        # rule 3: no dangling vertices
        for face_id in range(f_bdf.shape[-1]):
            non_dangling_triangle = (triangle_face_id == face_id).sum(1) >= 2
            triangle_idx = np.arange(len(triangles))[non_dangling_triangle]
            all_v_id = np.unique(triangles[triangle_idx])
            good_v_id = all_v_id[vertices_face_id[all_v_id] == face_id]

            all_v_id = np.where(vertices_face_id == face_id)[0]
            bad_v_id = np.setdiff1d(all_v_id, good_v_id)
            if len(bad_v_id) > 0:
                #print(f'Face {face_id} has dangling vertices, remove them')
                # assign the bad vertices to the neighbor face
                real_bad_v_id = []
                for v_id in bad_v_id:
                    # find the neighbor vertices
                    neighbor_v_ids = np.unique(triangles[np.where(triangles == v_id)[0]])
                    neighbor_face_ids = []
                    for neighbor_v_id in neighbor_v_ids:
                        if vertices_face_id[neighbor_v_id] != face_id:
                            neighbor_face_ids.append(vertices_face_id[neighbor_v_id])
                    if len(neighbor_face_ids) > 0:
                        # random
                        assign_v[v_id] = np.random.choice(neighbor_face_ids)
                    else:
                        print('Strange!!!')
                        real_bad_v_id.append(v_id)
                if len(real_bad_v_id) > 0:
                    real_bad_v_id = np.array(real_bad_v_id)
                    for v_id in real_bad_v_id:
                        # remove the assign_v
                        if v_id in assign_v:
                            assign_v.pop(v_id)
                    if face_id not in filter_v:
                        filter_v[face_id] = real_bad_v_id
                    else:
                        filter_v[face_id] = np.concatenate([filter_v[face_id], real_bad_v_id])
                re_compute = True
            if re_compute:
                break
        if not re_compute:
            break
    
    mesh = trimesh.Trimesh(vertices=vertices, faces=triangles, process=False)
    mesh.visual.vertex_colors = base_color[vertices_face_id % len(base_color)]
    mesh.export(f'{save_base}/mc_mesh_color.obj', include_color=True)
    mesh.export(f'{save_base}/mc_mesh_color.glb')

    brep = Brep()
    for face_idx in range(f_bdf.shape[-1]):
        triangle_belong_face = (triangle_face_id == face_idx).all(1)
        if np.sum(triangle_belong_face) == 0:
            continue
        # get valid triangle idx [True, False, True, True] -> [0, 2, 3]
        triangle_idx = np.arange(len(triangles))[triangle_belong_face]
        face_mesh = mesh.submesh([triangle_idx], append=True)
        components = face_mesh.split(only_watertight=False)
        #if len(components) > 1:
        #    print(f'WARNING: face {face_idx} has {len(components)} components')
        face = Face(face_idx, face_mesh)
        brep.add_face(face)
        face.mesh.export(f'{save_base}/face_{face_idx}.obj', include_color=True)

    faces = brep.faces
    all_face = None
    for face_idx, face in enumerate(faces):
        all_face = face.mesh if all_face is None else all_face + face.mesh
    all_face.export(f'{save_base}/face_all.obj', include_color=True)

    # find the boundary
    boundary_dict = {}
    for tri_idx, tri in enumerate(triangles):
        three_vertice_id = vertices_face_id[tri]
        if len(np.unique(three_vertice_id)) == 1:
            continue
        elif len(np.unique(three_vertice_id)) == 2:
            ids = np.unique(three_vertice_id)
            group_a, group_b = tri[three_vertice_id == ids[0]], tri[three_vertice_id == ids[1]]
            if len(group_a) == 1:
                two_points = group_b
                one_point = group_a
            else:
                two_points = group_a
                one_point = group_b

            # get the intersection points & find two triangles of the intersection points
            point1 = (vertices[two_points[0]] + vertices[one_point[0]]) / 2
            point2 = (vertices[two_points[1]] + vertices[one_point[0]]) / 2

            ids = np.sort(ids)
            if (ids[0], ids[1]) not in boundary_dict:
                boundary_dict[(ids[0], ids[1])] = Boundary()
                boundary = boundary_dict[(ids[0], ids[1])]
            else:
                boundary = boundary_dict[(ids[0], ids[1])]
            idx1 = boundary.add_vertex(point1)
            idx2 = boundary.add_vertex(point2)
            boundary.add_connection([idx1, idx2])
        else:
            center_point = np.mean(vertices[tri], axis=0)
            for i in range(3):
                two_ids = three_vertice_id[[i, (i+1)%3]]
                point = (vertices[tri[i]] + vertices[tri[(i+1)%3]]) / 2

                ids = np.sort(two_ids)
                if (ids[0], ids[1]) not in boundary_dict:
                    boundary_dict[(ids[0], ids[1])] = Boundary()
                    boundary = boundary_dict[(ids[0], ids[1])]
                else:
                    boundary = boundary_dict[(ids[0], ids[1])]
                idx1 = boundary.add_vertex(center_point, v_type=1)
                idx2 = boundary.add_vertex(point)
                boundary.add_connection([idx1, idx2])

    # add boundary to each face and brep
    faces = {}
    for face in brep.faces:
        faces[face.id] = face
    for k, v in boundary_dict.items():
        num_edges = v.split_into_edges() # divide the boundary into groups
        if num_edges == 0:
            print(f'The boundary between {k[0]} and {k[1]} has no valid edges')
            continue
        face_a = faces[k[0]]
        face_b = faces[k[1]]
        face_a.add_boundary(v)
        face_b.add_boundary(v)
        brep.add_boundary(v)
        v.compute_parametric_functions()

    brep.visualize(f'{save_base}/brep.png')
    brep.visualize(f'{save_base}/brep.html', type='plotly')

    # export brep
    brep_data = brep.export()
    with open(f'{save_base}/brep.pkl', 'wb') as f:
        pickle.dump(brep_data, f)

    # export the boundary vertices
    for k, v in boundary_dict.items():
        v.visualize_vertices(f'{save_base}/boundary_{k[0]}_{k[1]}', split_into_groups=True)

    for face_i in range(f_bdf.shape[-1]):
        face = f_bdf[..., face_i]
        points = np.where(face < 0.03)
        points = np.array(points).T
        pointcloud = trimesh.points.PointCloud(points)
        # save
        filename = f'{save_base}/face_only_{face_i}.obj'
        pointcloud.export(filename)


