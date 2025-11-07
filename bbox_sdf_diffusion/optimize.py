import os
os.environ["PYOPENGL_PLATFORM"] = "egl"
import numpy as np
import matplotlib.pyplot as plt
import mcubes
import trimesh
import pyrender
from scipy.interpolate import RegularGridInterpolator
from numpy.polynomial.polynomial import Polynomial, polyfit
import pymeshlab
from PIL import Image
import argparse
import pickle


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

def distance2targets(vertices, targets):
    # vertices: (N, 3)
    # targets: (M, 3)
    dist_matrix = np.linalg.norm(vertices[None] - targets[:, None], axis=-1) # (M, N)
    dist_min = np.min(dist_matrix, axis=0) # (N,)
    return dist_min
        
class Final_mesh:
    def __init__(self, vertices, vertices_face_id, triangles, boundary_info) -> None:
        self.vertices = vertices
        self.vertices_face_id = vertices_face_id
        self.triangles = triangles
        assert len(vertices) == len(vertices_face_id)
        self.boundary_info = boundary_info

        self.original_vertices = vertices.copy()
        self.original_vertices_face_id = vertices_face_id.copy()
        self.original_triangles = triangles.copy()
    
    def get_trimesh(self):
        mesh = trimesh.Trimesh(vertices=self.vertices, faces=self.triangles, process=False)
        return mesh
    
    def get_original_trimesh(self):
        mesh = trimesh.Trimesh(vertices=self.original_vertices, faces=self.original_triangles, process=False)
        return mesh

    def get_all_face_ids(self):
        face_ids = np.unique(self.vertices_face_id)
        face_ids = np.sort(face_ids[face_ids >= 0])
        return face_ids
    
    def get_face_mesh_by_id(self, face_id):
        triangle_face_id = self.vertices_face_id[self.triangles]
        triangle_belong_face = (triangle_face_id == face_id).any(1)
        triangle_idx = np.arange(len(self.triangles))[triangle_belong_face]
        all_mesh = self.get_trimesh()
        mesh = all_mesh.submesh([triangle_idx], append=True)
        return mesh
    
    def get_face_mesh_and_mapping_by_id(self, face_id):
        # return a trimesh and a mapping from 
        # the face mesh vertices to the original vertices
        # also return the boundary information
        triangle_face_id = self.vertices_face_id[self.triangles]
        triangle_belong_face = (triangle_face_id == face_id).any(1)
        triangle_idx = np.arange(len(self.triangles))[triangle_belong_face]
        mesh = self.get_trimesh()
        face_mesh = mesh.submesh([triangle_idx], append=True)
        new_idx, order = np.unique(face_mesh.faces, return_index=True)
        ori_idx = mesh.faces[triangle_idx].flatten()[order]
        # next, we find the boundary index
        boundary_idx = face_mesh.faces[triangle_face_id[triangle_idx]<0]
        boundary_idx = np.unique(boundary_idx)

        # next, T junction index
        t_idx = boundary_idx[self.vertices_face_id[ori_idx[boundary_idx]]==-2]
        t_idx = np.unique(t_idx)

        return face_mesh, ori_idx, boundary_idx, t_idx
        
    def get_boundary_vertices_idx_by_face_ids(self, face_id1, face_id2):
        triangle_face_id = self.vertices_face_id[self.triangles]
        all_bd_triangles_idx = np.where(triangle_face_id < 0)[0]
        all_bd_triangles_face_id = triangle_face_id[all_bd_triangles_idx]
        all_bd_triangles = self.triangles[all_bd_triangles_idx]
        
        face1_bd_tri = np.where((all_bd_triangles_face_id==face_id1).any(1))[0]
        face1_bd_vertices_idx = all_bd_triangles[face1_bd_tri][all_bd_triangles_face_id[face1_bd_tri] < 0]
        face2_bd_tri = np.where((all_bd_triangles_face_id==face_id2).any(1))[0]
        face2_bd_vertices_idx = all_bd_triangles[face2_bd_tri][all_bd_triangles_face_id[face2_bd_tri] < 0]
        face_12_vertices_idx = np.intersect1d(face1_bd_vertices_idx, face2_bd_vertices_idx) 

        return face_12_vertices_idx
    
    def get_order_and_split_boundary_vertices_idx_by_face_ids(self, face_id1, face_id2):
        bd_vertices_idx = self.get_boundary_vertices_idx_by_face_ids(face_id1, face_id2)
        all_bd_triangles = self.triangles[(self.vertices_face_id[self.triangles]<0).any(1)]

        # find all connections
        connections = {}
        for idx in bd_vertices_idx:
            connections[idx] = []
        for i in range(len(bd_vertices_idx)):
            idx = bd_vertices_idx[i]
            if len(connections[idx]) == 2:
                continue
            idx_belongs = all_bd_triangles[(all_bd_triangles == idx).any(1)]
            for check_idx in bd_vertices_idx[i+1:]:
                if check_idx in connections[idx]:
                    continue
                if (idx_belongs == check_idx).sum() > 0:
                    connections[idx].append(check_idx)
                    connections[check_idx].append(idx)

        # connections length:1 T-junc
        all_t_junctions = []
        for idx in connections:
            if len(connections[idx]) == 1:
                all_t_junctions.append(idx)
        groups = []
        unused_t_juncs = all_t_junctions.copy()
        while len(unused_t_juncs) != 0:
            start_t_junc = unused_t_juncs[0]
            group = [start_t_junc]
            unused_t_juncs = unused_t_juncs[1:]
            group.append(connections[start_t_junc][0])
            while True:
                current_idx = group[-1]
                neighbor_idxs = connections[current_idx]
                if len(neighbor_idxs) == 0:
                    raise ValueError('No neighbor idx')
                elif len(neighbor_idxs) == 1:
                    unused_t_juncs.remove(current_idx)
                    break
                elif len(neighbor_idxs) == 2:
                    if neighbor_idxs[0] not in group:
                        group.append(neighbor_idxs[0])
                    elif neighbor_idxs[1] not in group:
                        group.append(neighbor_idxs[1])
                    else:
                        raise ValueError('Loop')
            groups.append(group)
        
        return groups
    
    def reduce_boundary_vertices(self, percent=0.5):
        # percent: 0.9 means keep 90% of the vertices
        # reduce the boundary vertices
        merge_pairs = []
        face_ids = self.get_all_face_ids()
        for face_i in range(len(face_ids)):
            face0 = face_ids[face_i]
            for face_j in range(face_i+1, len(face_ids)):
                face1 = face_ids[face_j]
                groups = self.get_order_and_split_boundary_vertices_idx_by_face_ids(face0, face1)
                for group in groups:
                    merge_number = int((len(group) - 3) * (1-percent))
                    merge_number = max(merge_number, 0)
                    # uniformly select the vertices
                    if merge_number == 0:
                        continue
                    step = int((len(group) - 3) / merge_number)
                    for i in range(1, len(group)-1, step):
                        if i+1 >= len(group)-1:
                            break
                        merge_pairs.append([group[i], group[i+1]])
        if len(merge_pairs) == 0:
            return

        # merge the vertices
        vertices = self.vertices.copy()
        vertices_face_id = self.vertices_face_id.copy()
        index_map = {}
        merged_set = set()
        for merge_pair in merge_pairs:
            i, j = merge_pair
            if i in merged_set or j in merged_set:
                continue
            new_pos = vertices[i] * 0.5 + vertices[j] * 0.5
            new_index = len(vertices)
            vertices = np.vstack([vertices, new_pos])
            vertices_face_id = np.hstack([vertices_face_id, -1])
            index_map[i] = new_index
            index_map[j] = new_index
            merged_set.add(i)
            merged_set.add(j)
        
        # update the triangles
        new_triangles = []
        for tri in self.triangles:
            new_tri = [index_map.get(v, v) for v in tri]
            if len(set(new_tri)) == 3: # skip degenerate triangles
                new_triangles.append(new_tri)
        new_triangles = np.array(new_triangles, dtype=int)

        # remove unused vertices
        unused_vertices_idx = list(merged_set)
        unused_vertices_idx.sort(reverse=True)
        for v_idx in unused_vertices_idx:
            assert (new_triangles == v_idx).sum() == 0
            new_triangles[new_triangles > v_idx] -= 1
        new_vertices = np.delete(vertices, unused_vertices_idx, axis=0)
        new_vertices_face_id = np.delete(vertices_face_id, unused_vertices_idx, axis=0)
        
        self.vertices = new_vertices
        self.vertices_face_id = new_vertices_face_id
        self.triangles = new_triangles

    def get_boundary_vertices(self):
        # return the boundary vertices
        return self.vertices[self.vertices_face_id < 0], self.vertices_face_id[self.vertices_face_id < 0]

    def smoothing(self, iterations=10):
        all_face_ids = self.get_all_face_ids()
        for round_i in range(iterations):
            # 1. smooth each face
            boundary_dict = {}
            for face_id in all_face_ids:
                face_mesh, ori_idx, bd_idx, t_idx = self.get_face_mesh_and_mapping_by_id(face_id)
                bd2t_dist = distance2targets(face_mesh.vertices[bd_idx], face_mesh.vertices[t_idx])
                weights = 1 - bd2t_dist / np.max(bd2t_dist)
                for t in t_idx:
                    weights[bd_idx==t] = 1 # keep t-junc unchanged
                ms = pymeshlab.MeshSet()
                ms.add_mesh(
                    pymeshlab.Mesh(
                        vertex_matrix=face_mesh.vertices.copy(), 
                        face_matrix=face_mesh.faces)
                )
                ms.apply_filter("apply_coord_laplacian_smoothing", 
                        stepsmoothnum=1, boundary=True, 
                        cotangentweight=False)
                new_vertices = np.array(ms.current_mesh().vertex_matrix())
                new_vertices[bd_idx] = face_mesh.vertices[bd_idx] * weights[:,None] + new_vertices[bd_idx] * (1 - weights[:,None])

                new_vertices = new_vertices * 0.2 + face_mesh.vertices * 0.8

                self.vertices[ori_idx] = new_vertices # update the original vertices
                # store the boundary, later average
                for i in range(len(bd_idx)):
                    ori_bd_idx = ori_idx[bd_idx[i]]
                    if ori_bd_idx not in boundary_dict:
                        boundary_dict[ori_bd_idx] = [new_vertices[bd_idx[i]]]
                    else:
                        boundary_dict[ori_bd_idx].append(new_vertices[bd_idx[i]])
            # 2. average boundary vertices
            for ori_bd_idx in boundary_dict:
                assert len(boundary_dict[ori_bd_idx]) == 2 or len(boundary_dict[ori_bd_idx]) == 3
                self.vertices[ori_bd_idx] = np.mean(boundary_dict[ori_bd_idx], axis=0)

    def get_boundary_vertices_idx_by_face_id(self, face_id):
        triangle_face_id = self.vertices_face_id[self.triangles]
        valid_triangles = self.triangles[(triangle_face_id == face_id).any(1)]
        bd_idx = valid_triangles[self.vertices_face_id[valid_triangles] < 0]
        bd_idx = np.unique(bd_idx)
        return bd_idx

    def remeshing_(self, mesh):
        ms = pymeshlab.MeshSet()
        ms.add_mesh(
            pymeshlab.Mesh(
                vertex_matrix=mesh.vertices.copy(), 
                face_matrix=mesh.faces)
        )
        boundaries = mesh.outline()
        entities = boundaries.entities
        
        ms.compute_selection_from_mesh_border()
        ms.apply_selection_inverse(invfaces=True)
        ms.apply_filter('meshing_decimation_quadric_edge_collapse',
                        targetperc=0.5, preserveboundary=True, boundaryweight=100000,
                        selected=True)
        remeshed_vertices = np.array(ms.current_mesh().vertex_matrix())
        remeshed_faces = np.array(ms.current_mesh().face_matrix())

        return remeshed_vertices, remeshed_faces

    def remeshing(self):
        all_face_ids = self.get_all_face_ids()
        remeshed_results = {}
        for face_id in all_face_ids:
            face_bd_idx = self.get_boundary_vertices_idx_by_face_id(face_id)
            face_mesh = self.get_face_mesh_by_id(face_id)
            remeshed_vertices, remeshed_faces = self.remeshing_(face_mesh)
            # search for the boundary vertice idx
            dist_matrix = np.linalg.norm(remeshed_vertices[None] - self.vertices[face_bd_idx][:, None], axis=-1) # (N, M)
            remeshed_bd_idx = np.argmin(dist_matrix, axis=1) # (N,)
            assert len(remeshed_bd_idx) == len(np.unique(remeshed_bd_idx))
            remeshed_results[face_id] = (remeshed_vertices, remeshed_faces, remeshed_bd_idx, face_bd_idx)
        
        # now we need to rebuild the mesh
        final_vertices = np.empty((0, 3))
        final_faces = np.empty((0, 3), dtype=int)
        final_vertices_face_id = np.empty((0,), dtype=int)
        bd_map = {}
        final_unused_vertices_idx = []
        for face_id in remeshed_results:
            remeshed_vertices, remeshed_faces, remeshed_bd_idx, face_bd_idx = remeshed_results[face_id]
            remeshed_vertices_face_id = np.full(len(remeshed_vertices), face_id, dtype=int)
            for bd_i in range(len(remeshed_bd_idx)):
                remeshed_vertices_face_id[remeshed_bd_idx[bd_i]] = self.vertices_face_id[face_bd_idx[bd_i]]
            remeshed_faces = remeshed_faces.copy()
            for bd_i in range(len(remeshed_bd_idx)):
                original_bd_idx = face_bd_idx[bd_i]
                if original_bd_idx in bd_map:
                    curr_idx = remeshed_bd_idx[bd_i]
                    remeshed_faces[remeshed_faces == curr_idx] = bd_map[original_bd_idx] - len(final_vertices)
                    final_unused_vertices_idx.append(curr_idx+len(final_vertices))
                else:
                    bd_map[original_bd_idx] = remeshed_bd_idx[bd_i] + len(final_vertices)
            final_faces = np.vstack([final_faces, remeshed_faces + len(final_vertices)])
            final_vertices = np.vstack([final_vertices, remeshed_vertices])
            final_vertices_face_id = np.hstack([final_vertices_face_id, remeshed_vertices_face_id])
        
        # now we need to remove the unused vertices
        final_unused_vertices_idx.sort(reverse=True)
        for unused_idx in final_unused_vertices_idx:
            assert (final_faces == unused_idx).sum() == 0
            final_faces[final_faces > unused_idx] -= 1
        final_vertices = np.delete(final_vertices, final_unused_vertices_idx, axis=0)
        final_vertices_face_id = np.delete(final_vertices_face_id, final_unused_vertices_idx, axis=0)

        self.vertices = final_vertices
        self.triangles = final_faces
        self.vertices_face_id = final_vertices_face_id

    def export_faces(self, rootdir):
        face_ids = self.get_all_face_ids()
        os.makedirs(rootdir, exist_ok=True)
        for face_id in face_ids:
            mesh = self.get_face_mesh_by_id(face_id)
            mesh.visual.vertex_colors = base_color[face_id % len(base_color)]
            mesh.export(os.path.join(rootdir, f'face_{face_id}.obj'), include_color=True)

    def export(self, filename, is_colored=False):
        mesh = self.get_trimesh()
        if is_colored:
            mesh.visual.vertex_colors = base_color[self.vertices_face_id % len(base_color)]
            mesh.visual.vertex_colors[self.vertices_face_id < 0] = [0, 0, 0, 255]
        mesh.export(filename, include_color=True)
    
    def export_to_wireframe(self, filename):
        import matplotlib.pyplot as plt
        elev, azim, dist = 30, -60, 5

        fig = plt.figure(figsize=(12,12))
        ax = fig.add_subplot(111, projection='3d')
        ax.view_init(elev=elev, azim=azim)
        ax.dist = dist
        for bd_info in self.boundary_info.values():
            for func in bd_info.parametric_functions:
                px, py, pz, t = func
                x, y, z = px(t), py(t), pz(t)
                ax.plot(x, y, z, color='black', linewidth=5)
                # draw end points
                ax.scatter([x[0], x[-1]], [y[0], y[-1]], [z[0], z[-1]], color='red', s=24)
        ax.set_box_aspect([1, 1, 1])  # aspect ratio is 1:1:1
        ax.set_xlim([0, 64])
        ax.set_ylim([0, 64])
        ax.set_zlim([0, 64])
        ax.axis('off')
        plt.savefig(filename, dpi=100, bbox_inches='tight')
        plt.close()
    
    def render_mesh_with_camera(self, elev=30, azim=-60, dist=10):
        scene = pyrender.Scene()
        #mesh = self.get_trimesh()
        mesh = self.get_original_trimesh()
        if mesh.vertices.shape[0]!=0 and mesh.vertices.max() > 10:
            mesh.vertices = (mesh.vertices - 32) / 64 * 2
        mesh.vertices[:, [0,1,2]] = mesh.vertices[:, [1,2,0]]
        mesh = pyrender.Mesh.from_trimesh(mesh)
        scene.add(mesh)

        azim = np.radians(azim)
        elev = np.radians(elev)
    
        x = dist * np.sin(azim) * np.cos(elev)
        y = dist * np.sin(elev)
        z = dist * np.cos(azim) * np.cos(elev)
        # Set up the camera with given parameters

        camera_z_axis = -np.array([x, y, z]) / np.linalg.norm([x, y, z])
        camera_x_axis = np.array([np.cos(azim), 0, -np.sin(azim)])
        camera_y_axis = -np.cross(camera_z_axis, camera_x_axis)

        camera2world = np.eye(4)
        camera2world[:3, :3] = np.vstack([camera_x_axis, camera_y_axis, camera_z_axis]).T
        camera2world[:3, 3] = [x, y, z]
        world2camera = np.linalg.inv(camera2world)
        world2camera[2,:] = -world2camera[2,:]
        camera2world = np.linalg.inv(world2camera)
        cam_pose = camera2world

        camera = pyrender.PerspectiveCamera(yfov=np.pi / 9.0, aspectRatio=1.0)
        scene.add(camera, pose=cam_pose)

        # Set up the light
        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2.0)
        scene.add(light, pose=cam_pose)

        # Render the scene
        r = pyrender.OffscreenRenderer(viewport_width=1024, viewport_height=1024)
        color, depth = r.render(scene)

        return color

    def export_to_mesh_image(self, filename):
        img = self.render_mesh_with_camera(elev=30, azim=-60, dist=5)
        img = Image.fromarray(img)
        img.save(filename)
    
    def export_to_mesh_rotation_images(self, rootdir):
        os.makedirs(rootdir, exist_ok=True)
        for azim in range(-60, 300, 10):
            img = self.render_mesh_with_camera(elev=30, azim=azim, dist=5)
            img = Image.fromarray(img)
            img.save(os.path.join(rootdir, f'rot_{azim+60:03d}.png'))

    def optimize(self, iterations=10, save_root='./'):
        # 1. push boundary, 2. smooth, 3. remesh, 4. reduce boundary vertices
        for iter_i in range(iterations):
            print(f'Iter {iter_i}/{iterations}')
            save_dir = os.path.join(save_root, f'iter_{iter_i}')
            os.makedirs(save_dir, exist_ok=True)
            try:
                for _ in range(5):
                    self.optimize_boundary(factor=0.1) # the higher the factor, the more aggressive
                    self.smoothing(iterations=15)
                self.export(os.path.join(save_dir, 'opt_smooth.obj'), is_colored=True)
            except Exception as e:
                print('Smoothing failed: ', e)
            try:
                self.remeshing()
                self.export(os.path.join(save_dir, 'opt_remesh.obj'), is_colored=True)
            except Exception as e:
                print('Remeshing failed:', e)
            previous_vertices = self.vertices.copy()
            previous_triangles = self.triangles.copy()
            previous_vertices_face_id = self.vertices_face_id.copy()
            self.reduce_boundary_vertices(percent=0.5)
            self.export_faces(save_dir)
            if self.get_trimesh().is_watertight:
                self.export(os.path.join(save_dir, 'opt_reduce.obj'))
            else:
                self.vertices = previous_vertices
                self.triangles = previous_triangles
                self.vertices_face_id = previous_vertices_face_id
                print('Reduce failed, revert to previous state')
                break
            self.export(os.path.join(save_dir, 'opt_final.obj'), is_colored=True)
        
    def optimize_boundary(self, factor=0.1):
        for k, bd_info in self.boundary_info.items():
            groups = self.get_order_and_split_boundary_vertices_idx_by_face_ids(k[0], k[1])
            assert len(groups) == len(bd_info.parametric_functions)
            # match the groups with the parametric functions
            matched_groups = []
            used_parametric_functions = []
            for group_i in range(len(groups)):
                group = groups[group_i]
                for param_i in range(len(bd_info.parametric_functions)):
                    if param_i in used_parametric_functions:
                        continue
                    parametric_function = bd_info.parametric_functions[param_i]
                    px, py, pz, t = parametric_function
                    x, y, z = px(t), py(t), pz(t)
                    # check if the start and end points are close
                    if (np.linalg.norm(self.vertices[group[0]] - np.array([x[0], y[0], z[0]])) < 1e-2 and \
                        np.linalg.norm(self.vertices[group[-1]] - np.array([x[-1], y[-1], z[-1]])) < 1e-2) or \
                            (np.linalg.norm(self.vertices[group[0]] - np.array([x[-1], y[-1], z[-1]])) < 1e-2 and \
                             np.linalg.norm(self.vertices[group[-1]] - np.array([x[0], y[0], z[0]])) < 1e-2):
                        matched_groups.append((group, parametric_function))
                        used_parametric_functions.append(param_i)
                        break
            assert len(matched_groups) == len(groups)
            # now we have the matched groups and parametric functions
            # for each group, we want to push the vertices to the parametric function
            for group, parametric_function in matched_groups:
                px, py, pz, t = parametric_function
                x, y, z = px(t), py(t), pz(t)
                target_points = np.array([x, y, z]).T
                for v_idx in group:
                    # find the closest point on the parametric function
                    dist_matrix = np.linalg.norm(self.vertices[v_idx] - target_points, axis=1)
                    closest_idx = np.argmin(dist_matrix)
                    closest_point = target_points[closest_idx]
                    # update the vertex position
                    self.vertices[v_idx] = self.vertices[v_idx] * (1 - factor) + closest_point * factor

    def export_fusion_format(self, save_name):
        with open(save_name, 'w') as f:
            for v in self.vertices:
                v_ = v / 64 * 2 - 1
                f.write(f'v {v_[0]} {v_[1]} {v_[2]}\n')
            
            all_face_ids = self.get_all_face_ids()
            for face_id in all_face_ids:
                f.write(f'g face {face_id}\n')
                valid_triangle = (self.vertices_face_id[self.triangles] == face_id).any(1)
                for tri in self.triangles[valid_triangle]:
                    t = tri + 1
                    f.write(f'f {t[0]} {t[1]} {t[2]}\n')


class Boundary:
    def __init__(self):
        self.vertices = np.array([[],[],[]]).T
        self.vertices_type = np.array([], dtype=int)
        self.connections = np.array([[],[]], dtype=int).T
        self.groups = None
        self.parametric_functions = []
    
    def add_vertex(self, vertex, v_type=-1):
        if len(self.vertices) == 0:
            self.vertices = np.array([vertex])
            self.vertices_type = np.array([v_type])
            return 0
        dist = np.linalg.norm(self.vertices - vertex, axis=1)
        if np.min(dist) > 1e-7:
            self.vertices = np.vstack([self.vertices, vertex])
            self.vertices_type = np.hstack([self.vertices_type, v_type])
            return len(self.vertices) - 1
        else:
            self.vertices_type[np.argmin(dist)] = v_type
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
    
    def compute_parametric_functions(self):
        for group in self.groups:
            out_ = self.compute_parametric_function(group)
            if out_ is not None:
                self.parametric_functions.append(out_)

    def compute_parametric_function(self, group):
        # compute the parametric function for a group of vertices
        # group: list of vertex indices
        x = self.vertices[group][:, 0]
        y = self.vertices[group][:, 1]
        z = self.vertices[group][:, 2]
        weights = np.ones(len(x))
        weights[0] = weights[-1] = 1000

        t = np.zeros(x.shape[0])
        for i in range(1, t.shape[0]):
            t[i] = t[i-1] + np.linalg.norm(self.vertices[group[i]] - self.vertices[group[i-1]])
        
        # Fit polynomials to the data
        # check to use deg=1 or 3

        last_error = 10000
        for deg in [1, 3]:
            px = Polynomial.fit(t, x, deg, w=weights)
            py = Polynomial.fit(t, y, deg, w=weights)
            pz = Polynomial.fit(t, z, deg, w=weights)
            x_fit, y_fit, z_fit = px(t), py(t), pz(t)
            error = np.mean((x-x_fit)**2 + (y-y_fit)**2 + (z-z_fit)**2)
            if error > last_error / 10 or error < 1:
                break
            else:
                last_error = error
        
        t = np.linspace(t[0], t[-1], int(t[-1] * 20))
        if t.shape[0] == 0:
            return None
        
        return [px, py, pz, t]

    def post_process(self):
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
            if self.vertices_type[g[0]] != -2 or self.vertices_type[g[-1]] != -2:
                continue
            filtered_groups.append(g)
        self.groups = filtered_groups

        self.compute_parametric_functions()
        
        return len(self.groups)
    
    def export_parametric_points(self, save_base):
        for i in range(len(self.parametric_functions)):
            px, py, pz, t = self.parametric_functions[i]
            x = px(t)
            y = py(t)
            z = pz(t)
            points = np.vstack([x, y, z]).T
            pc = trimesh.points.PointCloud(points)
            pc.export(f'{save_base}_{i}.obj')
    
    def export_boundary(self, save_base):
        for i in range(len(self.groups)):
            group = self.groups[i]
            points = self.vertices[group]
            pc = trimesh.points.PointCloud(points)
            pc.export(f'{save_base}_{i}.obj')


def clean_bdf(f_bdf, threshold=0.08):
    valid_idx = []
    for i in range(f_bdf.shape[-1]):
        if np.sum(np.abs(f_bdf[..., i]) < threshold) > 0:
            valid_idx.append(i)
    return f_bdf[..., valid_idx]

def add_bd_vertices_return_idx(vertices, vertices_face_id, new_vertice, face_id):
    dist = np.linalg.norm(vertices - new_vertice, axis=1) + 10 * (vertices_face_id >= 0) # avoid collapse with real vertices
    if np.min(dist) > 1e-6:
        vertices = np.vstack([vertices, new_vertice])
        vertices_face_id = np.hstack([vertices_face_id, face_id])
        return vertices, vertices_face_id, len(vertices) - 1
    else:
        idx = np.argmin(dist)
        assert vertices_face_id[idx] < 0
        return vertices, vertices_face_id, idx
    

def distance_to_line_and_project_point(p, a, b):
    if p is None:
        return -1,-1,-1
    # p is the point to check
    # a, b is the line
    ap = p - a
    ab = b - a
    distance = np.linalg.norm(np.cross(ap, ab)) / np.linalg.norm(ab)
    
    # project point to the line
    t = np.dot(ap, ab) / np.dot(ab, ab)
    proj = a + t * ab
    return distance, proj, t

def line_interpolation(loc_a, loc_b, da, db):
    # da is the 2d distance
    t = (db[1]-db[0]) / ((da[0] - db[0]) - (da[1] - db[1]))

    w_a = 1/min(da)
    w_b = 1/min(db)
    # clip t to [0, 1]
    if t > 1 or t < 0:
        print('Warning: t out of range', t, loc_a, loc_b, da, db)
    t = max(0.1, min(0.9, t))
    assert t >= 0 and t <= 1
    return loc_a * t + loc_b * (1 - t)

def is_point_in_triangle(p, a, b, c):
    if p is None:
        return False
    # p is the point to check
    v0 = c - a
    v1 = b - a
    v2 = p - a

    dot00 = np.dot(v0, v0)
    dot01 = np.dot(v0, v1)
    dot02 = np.dot(v0, v2)
    dot11 = np.dot(v1, v1)
    dot12 = np.dot(v1, v2)

    denom = dot00 * dot11 - dot01 * dot01
    if abs(denom) < 1e-6:
        return False
    
    u = (dot11 * dot02 - dot01 * dot12) / denom
    v = (dot00 * dot12 - dot01 * dot02) / denom

    return u >= 0 and v >= 0 and u + v < 1

def project_point_to_plane(P, A, N):
    N = N / np.linalg.norm(N)
    d = np.dot(P-A, N)
    return P - d * N

def line_intersection_2d(p1, p2, p3, p4):
    A1 = p2[1] - p1[1]
    B1 = p1[0] - p2[0]
    C1 = A1*p1[0] + B1*p1[1]

    A2 = p4[1] - p3[1]
    B2 = p3[0] - p4[0]
    C2 = A2*p3[0] + B2*p3[1]

    det = A1*B2 - A2*B1
    if abs(det) < 1e-6:
        return None # parallel
    x = (B2*C1 - B1*C2) / det
    y = (A1*C2 - A2*C1) / det
    return np.array([x, y])

def intersection_of_projected_lines(x0, x1, y0, y1, a, b, c):
    N = np.cross(b-a, c-a)

    # plane equation
    # N0*(x-a0) + N1*(y-a1) + N2*(z-a2) = 0

    x0_proj = project_point_to_plane(x0, a, N)
    x1_proj = project_point_to_plane(x1, a, N)
    y0_proj = project_point_to_plane(y0, a, N)
    y1_proj = project_point_to_plane(y1, a, N)

    # Define plane coordinate system (U, V)
    U = (b - a) / np.linalg.norm(b - a)
    V = np.cross(N, U) / np.linalg.norm(np.cross(N, U))

    # convert to 2d coordinate
    x0_2d = np.array([np.dot(x0_proj - a, U), np.dot(x0_proj - a, V)])
    x1_2d = np.array([np.dot(x1_proj - a, U), np.dot(x1_proj - a, V)])
    y0_2d = np.array([np.dot(y0_proj - a, U), np.dot(y0_proj - a, V)])
    y1_2d = np.array([np.dot(y1_proj - a, U), np.dot(y1_proj - a, V)])

    # find the intersection of the two lines
    inter_2d = line_intersection_2d(x0_2d, x1_2d, y0_2d, y1_2d)
    if inter_2d is None:
        return None

    # convert back to 3d coordinate
    inter_3d = a + inter_2d[0] * U + inter_2d[1] * V
    return inter_3d

def center_interpolation(loc_x, loc_y, loc_z, dx, dy, dz):
    # dx is the 3d distance
    # t*dx[0] + p*dy[0] + (1-t-p)*dz[0] = t*dx[1] + p*dy[1] + (1-t-p)*dz[1]
    # t*dx[0] + p*dy[0] + (1-t-p)*dz[0] = t*dx[2] + p*dy[2] + (1-t-p)*dz[2]
    # t=?, p=?
    A = dx[0]-dx[1]-dz[0]+dz[1]
    B = dy[0]-dz[0]-dy[1]+dz[1]
    C = dz[1]-dz[0]
    D = dx[0]-dx[2]-dz[0]+dz[2]
    E = dy[0]-dz[0]-dy[2]+dz[2]
    F = dz[2]-dz[0]
    t = (B*F-C*E) / (B*D-A*E)
    p = (C*D-A*F) / (B*D-A*E)
    #assert t >= 0 and t <= 1

    center_point = t*loc_x + p*loc_y + (1-t-p)*loc_z
    return center_point

def find_intersection_between_line_and_surface(x,y,a,b,c):
    N = np.cross(b-a, c-a)
    N = N / np.linalg.norm(N)
    # plane equation
    # N0*(x-a0) + N1*(y-a1) + N2*(z-a2) = 0
    d = y - x
    denom = np.dot(d, N)
    if abs(denom) < 1e-6:
        return None
    t = np.dot(a-x, N) / denom
    return x + t * d

def get_vertices_idx(vertices, quaries):
    # vertices: nx3, quaries: mx3
    dist_matrix = np.linalg.norm(vertices[None] - quaries[:, None], axis=-1) # m, n
    return np.argmin(dist_matrix, axis=-1)

def brep_process(v_sdf, f_bdf, save_base, save_rot_images=False, opt_iter=3):
    # v_sdf: nxnxn
    # f_bdf: nxnxnxm

    #f_bdf = clean_bdf(f_bdf)

    vertices, triangles = mcubes.marching_cubes(v_sdf, 0)
    triangles = triangles[:, [2, 1, 0]]
    mesh = trimesh.Trimesh(vertices=vertices, faces=triangles, process=False)

    mesh.export(f'{save_base}/mc_mesh_ori.obj')
    triangles = triangles.astype(int)

    if vertices.shape[0] != 0:
        vertices, triangles = trimesh.remesh.subdivide_loop(vertices, triangles, 1)
    mesh = trimesh.Trimesh(vertices=vertices, faces=triangles, process=False)
    mesh.export(f'{save_base}/mc_mesh_sub.obj')

    grid_reso = f_bdf.shape[0]

    filter_v = {} # used for filter the vertices that are not in the mesh. k -> face_id, v -> v_id
    assign_v = {} # used for assign the vertices to the face. k -> v_id, v -> face_id
    compute_number = 0
    while compute_number < 10:
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
        if vertices_face_id.shape[0] != 0:
            mesh.visual.vertex_colors = base_color[vertices_face_id % len(base_color)]
        #mesh.export(f'{save_base}/{compute_number}.obj', include_color=True)

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
            else:
                strict_belong_face = (triangle_face_id == face_id).all(1)
                strict_vertices_idx = np.unique(triangles[strict_belong_face])
                soft_vertices_idx = np.unique(triangles[triangle_belong_face])
                # get vertices idx in soft but not in strict
                diff_vertices_idx = np.setdiff1d(soft_vertices_idx, strict_vertices_idx)
                for v_idx in diff_vertices_idx:
                    if vertices_face_id[v_idx] == face_id:
                        if face_id not in filter_v:
                            filter_v[face_id] = np.array([v_idx])
                        else:
                            filter_v[face_id] = np.concatenate([filter_v[face_id], [v_idx]])
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
    
    if compute_number == 10:
        print('Cannot find good solution')
        with open(f'{save_base}/warning.txt', 'w') as f:
            f.write('Cannot find good solution')

    mesh = trimesh.Trimesh(vertices=vertices, faces=triangles, process=False)
    if vertices_face_id.shape[0] != 0:
        mesh.visual.vertex_colors = base_color[vertices_face_id % len(base_color)]
    mesh.export(f'{save_base}/mc_mesh_color.obj', include_color=True)

    # start find the boundaries then project
    boundary_dict = {}
    new_triangles = {'2': [], '3': []}
    # 1. find the boundary vertices
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
            # interpolate the intersection points
            one_point_dists = interpolated_f_bdf[one_point[0]][ids]
            two_points_dists = interpolated_f_bdf[two_points][:, ids]

            # find the intersection between two lines
            # point1:
            loc_a, loc_b = vertices[one_point[0]], vertices[two_points[0]]
            da, db = one_point_dists, two_points_dists[0]
            point1 = line_interpolation(loc_a, loc_b, da, db)
            # point2:
            loc_a, loc_b = vertices[one_point[0]], vertices[two_points[1]]
            da, db = one_point_dists, two_points_dists[1]
            point2 = line_interpolation(loc_a, loc_b, da, db)

            ids = np.sort(ids)
            if (ids[0], ids[1]) not in boundary_dict:
                boundary_dict[(ids[0], ids[1])] = Boundary()
                bd = boundary_dict[(ids[0], ids[1])]
            else:
                bd = boundary_dict[(ids[0], ids[1])]
            idx1 = bd.add_vertex(point1, -1)
            idx2 = bd.add_vertex(point2, -1)
            if idx1 != idx2:
                bd.add_connection((idx1, idx2))

            new_triangles['2'].append(
                {
                    'old_tri_idx': tri_idx,
                    'one_point': one_point[0],
                    'two_points': two_points,
                    'new_points': [point1, point2],
                }
            )

        elif len(np.unique(three_vertice_id)) == 3:
            center_point = np.mean(vertices[tri], axis=0)

            # three points
            # point1
            id_ = three_vertice_id[[0,1]]
            point1_dist = interpolated_f_bdf[tri[0]][id_]
            point2_dist = interpolated_f_bdf[tri[1]][id_]
            point1_loc = vertices[tri[0]]
            point2_loc = vertices[tri[1]]
            point1 = line_interpolation(point1_loc, point2_loc, point1_dist, point2_dist)
            sorted_id = np.sort(id_)
            if (sorted_id[0], sorted_id[1]) not in boundary_dict:
                boundary_dict[(sorted_id[0], sorted_id[1])] = Boundary()
                bd = boundary_dict[(sorted_id[0], sorted_id[1])]
            else:
                bd = boundary_dict[(sorted_id[0], sorted_id[1])]
            idx1 = bd.add_vertex(point1, -1)
            idx2 = bd.add_vertex(center_point, -2)
            if idx1 != idx2:
                bd.add_connection((idx1, idx2))
        
            # point2
            id_ = three_vertice_id[[1,2]]
            point1_dist = interpolated_f_bdf[tri[1]][id_]
            point2_dist = interpolated_f_bdf[tri[2]][id_]
            point1_loc = vertices[tri[1]]
            point2_loc = vertices[tri[2]]
            point2 = line_interpolation(point1_loc, point2_loc, point1_dist, point2_dist)
            sorted_id = np.sort(id_)
            if (sorted_id[0], sorted_id[1]) not in boundary_dict:
                boundary_dict[(sorted_id[0], sorted_id[1])] = Boundary()
                bd = boundary_dict[(sorted_id[0], sorted_id[1])]
            else:
                bd = boundary_dict[(sorted_id[0], sorted_id[1])]
            idx1 = bd.add_vertex(point2, -1)
            idx2 = bd.add_vertex(center_point, -2)
            if idx1 != idx2:
                bd.add_connection((idx1, idx2))

            # point3
            id_ = three_vertice_id[[0,2]]
            point1_dist = interpolated_f_bdf[tri[0]][id_]
            point2_dist = interpolated_f_bdf[tri[2]][id_]
            point1_loc = vertices[tri[0]]
            point2_loc = vertices[tri[2]]
            point3 = line_interpolation(point1_loc, point2_loc, point1_dist, point2_dist)
            sorted_id = np.sort(id_)
            if (sorted_id[0], sorted_id[1]) not in boundary_dict:
                boundary_dict[(sorted_id[0], sorted_id[1])] = Boundary()
                bd = boundary_dict[(sorted_id[0], sorted_id[1])]
            else:
                bd = boundary_dict[(sorted_id[0], sorted_id[1])]
            idx1 = bd.add_vertex(point3, -1)
            idx2 = bd.add_vertex(center_point, -2)
            if idx1 != idx2:
                bd.add_connection((idx1, idx2))

            new_triangles['3'].append(
                {
                    'old_tri_idx': tri_idx,
                    'center_point': center_point,
                    'points': tri,
                    'new_points': [point1, point2, point3],
                }
            )
    
    # establish the boundary
    for data in new_triangles['2']:
        old_tri_idx = data['old_tri_idx']
        one_point_id = data['one_point']
        two_point_ids = data['two_points']
        new_points = data['new_points']
        center_point = (new_points[0] + new_points[1]) / 2

        vertices, vertices_face_id, new_idx1 = add_bd_vertices_return_idx(vertices, vertices_face_id, new_points[0], -1)
        vertices, vertices_face_id, new_idx2 = add_bd_vertices_return_idx(vertices, vertices_face_id, new_points[1], -1)
        vertices, vertices_face_id, center_idx = add_bd_vertices_return_idx(vertices, vertices_face_id, center_point, -1)

        # add triangles
        orientation = np.cross(
            vertices[triangles[old_tri_idx][0]] - vertices[triangles[old_tri_idx][1]],
            vertices[triangles[old_tri_idx][2]] - vertices[triangles[old_tri_idx][1]]
        )

        new_tris = []
        new_tris.append([center_idx, two_point_ids[0], two_point_ids[1]])
        new_tris.append([center_idx, two_point_ids[0], new_idx1])
        new_tris.append([center_idx, two_point_ids[1], new_idx2])
        new_tris.append([center_idx, new_idx1, one_point_id])
        new_tris.append([center_idx, new_idx2, one_point_id])

        # adjust the orientation
        for new_tri in new_tris:
            new_tri_orientation = np.cross(
                vertices[new_tri[0]] - vertices[new_tri[1]],
                vertices[new_tri[2]] - vertices[new_tri[1]]
            )
            if np.dot(new_tri_orientation, orientation) < 0:
                new_tri[0], new_tri[1] = new_tri[1], new_tri[0]
        new_tris = np.array(new_tris)
        triangles = np.vstack([triangles, new_tris])
    
    for data in new_triangles['3']:
        old_tri_idx = data['old_tri_idx']
        center_point = data['center_point']
        point_ids = data['points'] 
        new_points = data['new_points'] # 1: [0, 1], 2: [1, 2], 3: [0, 2]
        
        vertices, vertices_face_id, new_idx1 = add_bd_vertices_return_idx(vertices, vertices_face_id, new_points[0], -1)
        vertices, vertices_face_id, new_idx2 = add_bd_vertices_return_idx(vertices, vertices_face_id, new_points[1], -1)
        vertices, vertices_face_id, new_idx3 = add_bd_vertices_return_idx(vertices, vertices_face_id, new_points[2], -1)
        vertices, vertices_face_id, center_idx = add_bd_vertices_return_idx(vertices, vertices_face_id, center_point, -2)

        # add triangles
        orientation = np.cross(
            vertices[triangles[old_tri_idx][0]] - vertices[triangles[old_tri_idx][1]],
            vertices[triangles[old_tri_idx][2]] - vertices[triangles[old_tri_idx][1]]
        )
        new_tris = []
        new_tris.append([center_idx, point_ids[0], new_idx1])
        new_tris.append([center_idx, point_ids[1], new_idx1])
        new_tris.append([center_idx, point_ids[1], new_idx2])
        new_tris.append([center_idx, point_ids[2], new_idx2])
        new_tris.append([center_idx, point_ids[2], new_idx3])
        new_tris.append([center_idx, point_ids[0], new_idx3])

        # adjust the orientation
        for new_tri in new_tris:
            new_tri_orientation = np.cross(
                vertices[new_tri[0]] - vertices[new_tri[1]],
                vertices[new_tri[2]] - vertices[new_tri[1]]
            )
            if np.dot(new_tri_orientation, orientation) < 0:
                new_tri[0], new_tri[1] = new_tri[1], new_tri[0]
        new_tris = np.array(new_tris)
        triangles = np.vstack([triangles, new_tris])

    # remove the old triangles
    delted_tri_id1 = [data['old_tri_idx'] for data in new_triangles['2']]
    delted_tri_id2 = [data['old_tri_idx'] for data in new_triangles['3']]
    delted_tri_id = delted_tri_id1 + delted_tri_id2
    delted_tri_id = sorted(delted_tri_id, reverse=True)
    for tri_id in delted_tri_id:
        triangles = np.delete(triangles, tri_id, axis=0)

    mesh = trimesh.Trimesh(vertices=vertices, faces=triangles, process=False)
    mesh.export(f'{save_base}/mc_mesh_final.obj', include_color=True)

    # 2. find the boundary sequence
    for k, bd in boundary_dict.items():
        num_edges = bd.post_process()
        if num_edges == 0:
            print(f'The boundary between {k[0]} and {k[1]} has no valid edges')
            continue
        #bd.export_parametric_points(f'{save_base}/boundary_param_{k[0]}_{k[1]}')
        #bd.export_boundary(f'{save_base}/boundary_{k[0]}_{k[1]}')
    
    ### remove number of bd points and parametric it
    final_mesh = Final_mesh(
        vertices=vertices, triangles=triangles, 
        vertices_face_id=vertices_face_id, boundary_info=boundary_dict)
    final_mesh.optimize(iterations=opt_iter, save_root=f'{save_base}/opt')
    final_mesh.export_faces(f'{save_base}')
    final_mesh.export_fusion_format(f'{save_base}/final.obj')
    final_mesh.export_to_wireframe(f'{save_base}/wireframe.png')
    final_mesh.export_to_mesh_image(f'{save_base}/mesh.png')
    if save_rot_images:
        final_mesh.export_to_mesh_rotation_images(f'{save_base}/mesh_view')
    
    # store final_mesh as pkl
    with open(f'{save_base}/final_mesh.pkl', 'wb') as f:
        pickle.dump(final_mesh, f)


if __name__ == '__main__':
    save_path='temp_output2'
    #data = np.load(f'bbox_sdf_diffusion/output_deepcad/001/data.npy', allow_pickle=True).tolist()
    data = np.load(f'Data/processed/val/00200657/solid_0.npz', allow_pickle=True)
    solid_voxel = data['v_sdf']
    face_voxels = data['f_udf']
    save_path = os.path.join(save_path, 'processed')
    os.makedirs(save_path, exist_ok=True)
    brep_process(solid_voxel, face_voxels, save_path, save_rot_images=True, opt_iter=0)