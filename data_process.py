import os
from tqdm import tqdm
import open3d as o3d
from occwl.io import load_step, save_stl
from OCC.Extend.TopologyUtils import TopologyExplorer, WireExplorer
from OCC.Core.ShapeFix import ShapeFix_Face, ShapeFix_Wire, ShapeFix_Edge
from OCC.Core.ShapeAnalysis import ShapeAnalysis_Wire
import numpy as np
import glob
import argparse
import trimesh
import mcubes

def sample_points_on_mesh(mesh, num_points):
    points = mesh.sample_points_uniformly(number_of_points=num_points)
    points = np.asarray(points.points)
    return points

def fix_wires(face, debug=False):
    top_exp = TopologyExplorer(face)
    for wire in top_exp.wires():
        if debug:
            wire_checker = ShapeAnalysis_Wire(wire, face, 0.01)
            print(f"Check order 3d {wire_checker.CheckOrder()}")
            print(f"Check 3d gaps {wire_checker.CheckGaps3d()}")
            print(f"Check closed {wire_checker.CheckClosed()}")
            print(f"Check connected {wire_checker.CheckConnected()}")
        wire_fixer = ShapeFix_Wire(wire, face, 0.01)
        assert wire_fixer.IsReady()
        ok = wire_fixer.Perform()

def fix_face(face):
    fixer = ShapeFix_Face(face)
    fixer.SetPrecision(0.01)
    fixer.SetMaxTolerance(0.1)
    ok = fixer.Perform()
    # assert ok
    fixer.FixOrientation()
    face = fixer.Face()
    return face

def add_pcurves_to_edges(face):
    edge_fixer = ShapeFix_Edge()
    top_exp = TopologyExplorer(face)
    for wire in top_exp.wires():
        wire_exp = WireExplorer(wire)
        for edge in wire_exp.ordered_edges():
            edge_fixer.FixAddPCurve(edge, face, False, 0.001)

def split_solid(solid):
    '''
    Split closed faces on the solid into halve
    '''
    solid = solid.split_all_closed_faces(num_splits=0)
    solid = solid.split_all_closed_edges(num_splits=0)
    split_faces = []
    for face in solid.faces():
        face_occ = face.topods_shape()
        # Do some fixing
        fix_wires(face_occ)
        add_pcurves_to_edges(face_occ)
        fix_wires(face_occ)
        face_occ = fix_face(face_occ)
        split_faces.append(face_occ)
    return split_faces, solid

def scale_to_unit_cube(mesh, centroid, max_size):
    mesh.translate(-centroid)
    mesh.scale(2 / max_size, center=(0, 0, 0))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='Data/deepcad_subset/val')
    parser.add_argument('--save_root', type=str, default='Data/process/val')
    parser.add_argument('--group_num', type=int, default=1)
    parser.add_argument('--group_id', type=int, default=0)
    parser.add_argument('--max_faces', type=int, default=40)
    parser.add_argument('--save_npz', action='store_true', default=False)
    parser.add_argument('--check_size', action='store_true', default=False)
    args = parser.parse_args()

    voxel_resolution = 64

    data_ids = os.listdir(args.data_path)
    data_ids.sort()
    data_ids = data_ids[args.group_id::args.group_num]

    save_root = args.save_root
    for data_id in tqdm(data_ids):
        try:
            data_dir = os.path.join(args.data_path, data_id)
            # Load the step file
            save_dir = os.path.join(save_root, data_id)
            step_file = glob.glob(os.path.join(data_dir, '*.step'))[0]
            #if os.path.exists(os.path.join(save_dir, 'finished.txt')):
            #    continue
            solids = load_step(step_file)
            solids = [solids[0],]
            for solid_i, solid in enumerate(solids):
                faces_halve, solid_halve = split_solid(solid)
                if os.path.exists(os.path.join(save_dir, f'solid_{solid_i}.stl')) is False:
                    os.makedirs(save_dir, exist_ok=True)
                    save_stl(solid_halve.topods_shape(), 
                             os.path.join(save_dir, f'solid_{solid_i}.stl'))
                solid_mesh = o3d.io.read_triangle_mesh(os.path.join(save_dir, f'solid_{solid_i}.stl'))
                solid_mesh.remove_duplicated_vertices()
                solid_mesh.remove_duplicated_triangles()
                # check if the mesh is watertight
                if solid_mesh.is_watertight() is False:
                    break

                # normalise the mesh to [-1, 1]
                bbox = solid_mesh.get_axis_aligned_bounding_box()
                centroid = bbox.get_center()
                
                if args.check_size:
                    extent = bbox.get_extent()
                    if extent.max() > extent.min() * 30:
                        continue

                max_size = np.max(bbox.get_extent()) * 1.2
                scale_to_unit_cube(solid_mesh, centroid, max_size)
                
                # save normalized mesh
                save_path = os.path.join(save_dir, f'solid_{solid_i}_norm.obj')
                o3d.io.write_triangle_mesh(save_path, solid_mesh)

                faces = []
                check_flag = True
                if args.save_npz:
                    for face_i, face_halve in enumerate(faces_halve):
                        if os.path.exists(os.path.join(save_dir, f'face_{solid_i}_{face_i}.stl')) is False:
                            os.makedirs(save_dir, exist_ok=True)
                            save_stl(face_halve, 
                                     os.path.join(save_dir, f'face_{solid_i}_{face_i}.stl'))
                        face_mesh = o3d.io.read_triangle_mesh(os.path.join(save_dir, f'face_{solid_i}_{face_i}.stl'))
                        face_mesh.remove_duplicated_vertices()   
                        face_mesh.remove_duplicated_triangles()
                        scale_to_unit_cube(face_mesh, centroid, max_size)
                        face_bbox = face_mesh.get_axis_aligned_bounding_box()
                        face_centroid = face_bbox.get_center()
                        face_size = face_bbox.get_extent()

                        faces.append(face_mesh)
                        if args.check_size:
                            face_size_max = face_size.max()
                            try:
                                face_size_non_zero_min = face_size[face_size > 0.005].min()
                            except:
                                check_flag = False
                            if face_size_non_zero_min < 0.05:
                                check_flag = False
                            if face_size_max > face_size_non_zero_min * 20:
                                check_flag = False
                    if check_flag is False:
                        continue
                
                    if len(faces) > args.max_faces:
                        continue

                if args.save_npz:
                    # make solid sdf
                    mesh = o3d.t.geometry.TriangleMesh.from_legacy(solid_mesh)
                    scene = o3d.t.geometry.RaycastingScene()
                    _ = scene.add_triangles(mesh)
                    points = np.meshgrid(np.arange(voxel_resolution), np.arange(voxel_resolution), np.arange(voxel_resolution))
                    points = np.stack(points, axis=-1) # (64, 64, 64, 3)
                    points = points.reshape(-1, 3).astype(np.float32)
                    points = points / (voxel_resolution-1) * 2 - 1 # [-1, 1]
                    signed_distance = scene.compute_signed_distance(points)
                    voxel = signed_distance.numpy().reshape(voxel_resolution, voxel_resolution, voxel_resolution)
                    voxel = np.clip(voxel, -0.1, 0.1)

                    # save bounded distance field
                    distance_all = []
                    for face_mesh in faces:
                        scene = o3d.t.geometry.RaycastingScene()
                        scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(face_mesh))
                        unsigned_distance = scene.compute_distance(points).numpy()
                        unsigned_distance = unsigned_distance.reshape(voxel_resolution, voxel_resolution, voxel_resolution)
                        unsigned_distance = np.clip(unsigned_distance, -0.1, 0.1)
                        distance_all.append(unsigned_distance)

                    data = {}
                    data['v_sdf'] = voxel
                    data['f_udf'] = np.stack(distance_all, axis=-1)

                    # check if marching cube is good enough
                    #vertices, triangles = mcubes.marching_cubes(voxel, 0)
                    #check_mesh = o3d.geometry.TriangleMesh()
                    #check_mesh.vertices = o3d.utility.Vector3dVector(vertices)
                    #check_mesh.triangles = o3d.utility.Vector3iVector(triangles)
                    #check_mesh.remove_duplicated_vertices()
                    #check_mesh.remove_duplicated_triangles()
                    #vertices = np.asarray(check_mesh.vertices)
                    #triangles = np.asarray(check_mesh.triangles)
                    #check_mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)
                    #tt = check_mesh.split()
                    #if len(tt) > 1:
                    #    continue
                    np.savez_compressed(os.path.join(save_dir, f'solid_{solid_i}.npz'), **data)
                
            # label finished
            with open(os.path.join(save_dir, 'finished.txt'), 'w') as f:
                f.write('done')
        except:
            continue
            
            

