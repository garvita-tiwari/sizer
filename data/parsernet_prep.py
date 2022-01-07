"""Data for ParserNet
enighbours, vertices id etc..."""

import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
import tensorflow as tf
import ipdb
from psbody.mesh import Mesh, MeshViewer
import networkx
from networkx.algorithms.approximation import clique
import ipdb
import matplotlib.pyplot as plt
from psbody.mesh import Mesh
import time

def find_nb(init_node, graph, num_n, neighbour_list = []):
    nodes_to_be_visited = [init_node]
    nodes_visited = []
    while(len(neighbour_list) < num_n):
        tmp_list = [n  for i in range(len(nodes_to_be_visited)) for n in graph[nodes_to_be_visited[i]]]

        #update visited nodes
        nodes_visited = nodes_visited + nodes_to_be_visited

        #update neighbour list

        neighbour_list = list(set(nodes_to_be_visited + neighbour_list +tmp_list))

        #update nodes to be visited in next loop
        #visit all the nodes in
        nodes_to_be_visited = np.setdiff1d(neighbour_list, nodes_visited).tolist()
    return  neighbour_list

def get_lres_temp(gar_class='g1', garment='UpperClothes'):

    if gar_class == 'g1':
        sub = 10001
        scan = 1937

    if gar_class == 'g3':
        sub = 10001
        scan = 1953
    if gar_class == 'g5':
        sub = 10015
        scan = 2620

    if gar_class == 'g4':
        sub = 10005
        scan = 2086
    if gar_class == 'g7':
        sub = 10011
        scan = 2286
    import sys
    sys.path.append('/BS/garvita/work/code/registration_ver2-master')
    sys.path.extend(['/BS/garvita/work/code/RVH', '/BS/RVH/work/frankengeist'])
    from core.geometry.geometry import get_submesh
    multi_data = pkl.load(open('/BS/garvita2/static00/ClothingSize_registrations2/{}/{}/multimesh_neutral_init/{}.pkl'.format(sub, scan,garment)))
    m1 = Mesh(filename='/BS/garvita2/static00/ClothingSize_registrations2/{}/{}/multimesh_neutral_init/{}.ply'.format(sub, scan,garment))
    #m1 = Mesh(filename='/BS/garvita2/static00/ClothingSize_registrations/10015/2616/multimesh_neutral/Pants.ply')
    v,faces_body, _,_ =  get_submesh(multi_data['vh'], multi_data['fh'], multi_data['vert_indices'])
    m3 = Mesh(v=v, f=faces_body)
    garment = 'Body'
    m3.write_obj('/BS/garvita2/static00/ClothSize_data/gcn_assets/real_{}_hres_{}.obj'.format(gar_class, garment))

    lres_vert = np.array([vid for vid in multi_data['vert_indices'] if vid < 6890])
    np.save('/BS/garvita2/static00/ClothSize_data/gcn_assets/real_{}_lres_vertid_{}.npy'.format(gar_class,garment), lres_vert)
    np.save('/BS/garvita2/static00/ClothSize_data/gcn_assets/real_{}_hres_vertid_{}.npy'.format(gar_class,garment), multi_data['vert_indices'])
    #save sample lres mesh
    v, faces_body, _, _ = get_submesh(multi_data['v'], multi_data['f'], lres_vert)
    m3 = Mesh(v=v, f=faces_body)
    m3.write_obj('/BS/garvita2/static00/ClothSize_data/gcn_assets/real_{}_lres_{}.obj'.format(gar_class,garment))

    if garment == 'Pants':
        #store hres data as lres
        multi_data = pkl.load(open(
            '/BS/garvita2/static00/ClothingSize_registrations2/{}/{}/multimesh_neutral_init/{}.pkl'.format(sub, scan,
                                                                                                           garment)))

        np.save('/BS/garvita2/static00/ClothSize_data/gcn_assets/real_{}_lres_vertid_{}.npy'.format(gar_class, garment),
                multi_data['vert_indices'])

        # m1 = Mesh(
        #     filename='/BS/garvita2/static00/ClothingSize_registrations2/{}/{}/multimesh_neutral_init/{}.ply'.format(sub,
        #                                                                                                             scan,
        #                                                                                                             garment))
        # # m1 = Mesh(filename='/BS/garvita2/static00/ClothingSize_registrations/10015/2616/multimesh_neutral/Pants.ply')
        # v, faces_body, _, _ = get_submesh(multi_data['vh'], multi_data['fh'], multi_data['vert_indices'])
        m3 = Mesh(v=v, f=faces_body)
        m1.write_obj('/BS/garvita2/static00/ClothSize_data/gcn_assets/real_{}_lres_{}.obj'.format(gar_class, garment))



def get_neighbor(garment='real_g5', garment_layer='UpperClothes', res='lres',  num_neigh = 100):
    """ vertices id of neighbors for every vertex in the garment template"""

    G = networkx.Graph()
    #load whole SMPL mesh

    #if garment_layer == 'UpperClothes' or garment_layer  == 'Body':
    m = Mesh(filename='/BS/garvita/work/dataset/smpl_mesh.obj')
    # if garment_layer == 'Pants':
    #     m = Mesh(filename='/BS/garvita/work/dataset/smpl_mesh_apose.obj')
    if res == 'hres':
        m = Mesh(filename='/BS/garvita/work/dataset/smpl_mesh_apose.obj')

    #m2 =Mesh(filename='/BS/garvita2/static00/ClothSize_data/gcn_assets/{}_{}_{}.obj'.format(garment, res, garment_layer))
    #read garment_label
    #upper_file = pkl.load(open('/BS/garvita2/static00/ClothingSize_registrations/10015/2620/multimesh_neutral/{}.pkl'.format(garment)))

    vert_difile = '/BS/garvita4/static00/sizer_final/mesh_utils/{}/{}_{}_id.npy'.format(garment, garment_layer, res)

        #read upper and lowers verts and remove it from body, do this for high res
    upper_vertid = np.load(vert_difile)

    G.add_nodes_from(range(len(m.v)))
    edges = [[vid[(i + 1) % 3], vid[i % 3]] for vid in m.f for i in range(3)]
    G.add_edges_from(edges)
    # networkx.draw_networkx(G, with_labels=False)
    # plt.axis("off")
    # plt.show()
    # ipdb.set_trace()
    H = networkx.path_graph(5)
    G.add_edges_from(H.edges())
    G.add_edges_from([(0, 2)])
    fin2 = []
    #get neighbour
    net_neighbours = {}
    net_neighbours_smpl = {}
    net_neighbours_gar = {}



    for j,i in enumerate(upper_vertid):
        #find neighbours for vertices belonging to garment mesh only
            fin_list = find_nb(i, G, num_neigh, fin2)
            net_neighbours['{}_{}'.format(i, j)] =fin_list[:num_neigh]

            #todo: hack, make sure the vertex is present in this list
            if i not in fin_list[:num_neigh]:
                fin_list[num_neigh -1] = i
            net_neighbours_smpl[i] =fin_list[:num_neigh]
            net_neighbours_gar[j] =fin_list[:num_neigh]
            print("neighbor {} done".format(i))
            label = np.zeros(m.v.shape[0])
            label[fin_list] = 1

            #replace_vert
            #final_mesh.v[i] = m2.v[j]
            m.set_vertex_colors_from_weights(label)
            #Mesh(v=m.v[fin_list]).write_obj('/BS/garvita4/static00/sizer_final/mesh_utils/tmp/{}.obj'.format(j))
            # mv  = MeshViewer()
            # mv.set_static_meshes([m])
            # mv.set_background_color(np.array([1., 1., 1.]))
            # mv.save_snapshot("/BS/garvita/work/dataset/gcn/heuristics/100N_{}/{}.png".format(garment, i))
            # time.sleep(1)
            # mv.close()
    # with open("/BS/garvita2/static00/ClothSize_data/gcn_assets/{}_neighborheuristics_{}_{}_{}_both2.pkl".format(garment, res, garment_layer, num_neigh), 'wb') as f:
    #    pkl.dump(net_neighbours, f)
    #
    # with open("/BS/garvita2/static00/ClothSize_data/gcn_assets/{}_neighborheuristics_{}_{}_{}_smpl2.pkl".format(garment,  res,garment_layer,num_neigh), 'wb') as f:
    #    pkl.dump(net_neighbours_smpl, f)
    # with open("/BS/garvita2/static00/ClothSize_data/gcn_assets/{}_neighborheuristics_{}_{}_{}_gar2.pkl".format(garment, res,garment_layer, num_neigh), 'wb') as f:
    #    pkl.dump(net_neighbours_gar, f)

    ordered_neighbour = []
    for i in range(len(upper_vertid)):
        ordered_neighbour.append(net_neighbours_gar[i])
    #
    # with open("/BS/garvita2/static00/ClothSize_data/gcn_assets/{}_neighborheuristics_{}_{}_{}_gar_order.pkl".format(
    #         garment, res,garment_layer, num_neigh), 'wb') as f:
    #     pkl.dump(np.array(ordered_neighbour), f)

    np.save("/BS/garvita4/static00/sizer_final/mesh_utils/{}/{}_{}_{}_gar_order.npy".format(
            garment, garment_layer,res, num_neigh), np.array(ordered_neighbour))

def temp_files(gar_class='g1'):

    import os
    import sys

    sys.path.append('/BS/garvita/work/code/cloth_static/TailorNet')

    from models.smpl4garment import SMPL4Garment
    sys.path.append('/BS/garvita/work/code/registration_ver2-master')
    sys.path.extend(['/BS/garvita/work/code/RVH', '/BS/RVH/work/frankengeist'])
    from core.geometry.geometry import get_submesh
    if gar_class == 'g1' or  gar_class == 'g2' or gar_class == 'g3' or gar_class == 'g4':
        upper = 'shirt'
        lower = 'pant'
    if gar_class == 'g5' or  gar_class == 'g6':
        upper = 't-shirt'
        lower = 'short-pant'

    if gar_class == 'g7':
        upper = 't-shirt'
        lower = 'short-pant'
        upper_old = 'Vest'

    root_out = '/BS/garvita4/static00/sizer_final/mesh_utils/{}'.format(gar_class)
    if not os.path.exists(root_out):
        os.makedirs(root_out )
    smpl = SMPL4Garment('male')
    _, upper_m = smpl.run(garment_class=upper)
    _, lower_m = smpl.run(garment_class=lower)


    upper_m.write_obj('{}/UpperClothes_hres.obj'.format(root_out))
    lower_m.write_obj('{}/Pants_hres.obj'.format(root_out))
    #
    #
    # template_file = '/BS/garvita2/work/template_files/allTemplate_withBoundaries_symm.pkl'
    # with open(template_file, 'rb') as f:
    #     template_data = pkl.load(f, encoding="latin1")
    #
    # #load hres smpl
    smpl_hres = Mesh(filename='/BS/cloth3d/static00/nasa_data/smpl_sdf/meshes/000/000000.obj')
    upper_id = smpl.class_info[upper]['vert_indices']
    lower_id =smpl.class_info[lower]['vert_indices']
    body_vert = range(len(smpl_hres.v))

    body_vert2 = [i for i in body_vert if i not in upper_id]
    body_vert2 = [i for i in body_vert2 if i not in lower_id]
    body_vert = np.array(body_vert2)
    v, faces_body, _, _ = get_submesh(smpl_hres.v, smpl_hres.f, body_vert)
    body_hres = Mesh(v=v,f=faces_body)
    body_hres.write_obj('{}/Body_hres.obj'.format(root_out))

    #get vertices of body

    #create lres for upper clothing
    lres_vert_upper = np.array([vid for vid in upper_id if vid < 6890])
    lres_vert_lower = np.array([vid for vid in lower_id if vid < 6890])
    lres_body_vert = np.array([vid for vid in body_vert if vid < 6890])

    #load lres and create the templates
    smpl_lres = '/BS/garvita/work/dataset/smpl_mesh.obj'
    m1 = Mesh(filename=smpl_lres)
    v, faces_body, _, _ = get_submesh(m1.v, m1.f, lres_vert_upper)
    upper_lres = Mesh(v=v,f=faces_body)
    v, faces_body, _, _ = get_submesh(m1.v, m1.f, lres_vert_lower)
    lower_lres = Mesh(v=v,f=faces_body)
    v, faces_body, _, _ = get_submesh(m1.v, m1.f, lres_body_vert)
    body_lres = Mesh(v=v,f=faces_body)
    upper_lres.write_obj('{}/UpperClothes_lres.obj'.format(root_out))
    lower_lres.write_obj('{}/Pants_lres.obj'.format(root_out))
    body_lres.write_obj('{}/Body_lres.obj'.format(root_out))
    np.save('{}/UpperClothes_hres_id.npy'.format(root_out), upper_id)
    np.save('{}/Pants_hres_id.npy'.format(root_out), lower_id)
    np.save('{}/Body_hres_id.npy'.format(root_out), body_vert)

    np.save('{}/UpperClothes_lres_id.npy'.format(root_out), lres_vert_upper)
    np.save('{}/Pants_lres_id.npy'.format(root_out), lres_vert_lower)
    np.save('{}/Body_lres_id.npy'.format(root_out), lres_body_vert)



if __name__ == "__main__":
    # temp_files("g1")
    # temp_files("g2")
    # temp_files("g3")
    # temp_files("g4")
    # temp_files("g5")
    # temp_files("g6")
    # temp_files("g7")
    # 
    get_neighbor('g1', 'Body', 'hres', 100)
    get_neighbor('g2', 'Body', 'hres', 100)
    get_neighbor('g3', 'Body', 'hres', 100)
    get_neighbor('g4', 'Body', 'hres', 100)
    get_neighbor('g5', 'Body', 'hres', 100)
    get_neighbor('g6', 'Body', 'hres', 100)
    get_neighbor('g7', 'Body', 'hres', 100)

    get_neighbor('g1', 'Body', 'lres', 100)
    get_neighbor('g2', 'Body', 'lres', 100)
    get_neighbor('g3', 'Body', 'lres', 100)
    get_neighbor('g4', 'Body', 'lres', 100)
    get_neighbor('g5', 'Body', 'lres', 100)
    get_neighbor('g6', 'Body', 'lres', 100)
    get_neighbor('g7', 'Body', 'lres', 100)
    
    get_neighbor('g1', 'Body', 'hres', 50)
    get_neighbor('g2', 'Body', 'hres', 50)
    get_neighbor('g3', 'Body', 'hres', 50)
    get_neighbor('g4', 'Body', 'hres', 50)
    get_neighbor('g5', 'Body', 'hres', 50)
    get_neighbor('g6', 'Body', 'hres', 50)
    get_neighbor('g7', 'Body', 'hres', 50)
    
    get_neighbor('g1', 'Body', 'lres', 50)
    get_neighbor('g2', 'Body', 'lres', 50)
    get_neighbor('g3', 'Body', 'lres', 50)
    get_neighbor('g4', 'Body', 'lres', 50)
    get_neighbor('g5', 'Body', 'lres', 50)
    get_neighbor('g6', 'Body', 'lres', 50)
    get_neighbor('g7', 'Body', 'lres', 50)
    
    get_neighbor('g1', 'Body', 'hres', 20)
    get_neighbor('g2', 'Body', 'hres', 20)
    get_neighbor('g3', 'Body', 'hres', 20)
    get_neighbor('g4', 'Body', 'hres', 20)
    get_neighbor('g5', 'Body', 'hres', 20)
    get_neighbor('g6', 'Body', 'hres', 20)
    get_neighbor('g7', 'Body', 'hres', 20)

    get_neighbor('g1', 'Body', 'lres', 20)
    get_neighbor('g2', 'Body', 'lres', 20)
    get_neighbor('g3', 'Body', 'lres', 20)
    get_neighbor('g4', 'Body', 'lres', 20)
    get_neighbor('g5', 'Body', 'lres', 20)
    get_neighbor('g6', 'Body', 'lres', 20)
    get_neighbor('g7', 'Body', 'lres', 20)
    ipdb.set_trace()
    get_neighbor('g1', 'UpperClothes', 'hres', 100)
    get_neighbor('g1', 'Pants', 'hres', 100)
    get_neighbor('g2', 'UpperClothes', 'hres', 100)
    get_neighbor('g2', 'Pants', 'hres', 100)
    get_neighbor('g3', 'UpperClothes', 'hres', 100)
    get_neighbor('g3', 'Pants', 'hres', 100)
    get_neighbor('g4', 'UpperClothes', 'hres', 100)
    get_neighbor('g4', 'Pants', 'hres', 100)
    get_neighbor('g5', 'UpperClothes', 'hres', 100)
    get_neighbor('g5', 'Pants', 'hres', 100)
    get_neighbor('g6', 'UpperClothes', 'hres', 100)
    get_neighbor('g6', 'Pants', 'hres', 100)
    get_neighbor('g7', 'UpperClothes', 'hres', 100)
    get_neighbor('g7', 'Pants', 'hres', 100)


    get_neighbor('g1', 'UpperClothes', 'hres', 50)
    get_neighbor('g1', 'Pants', 'hres', 50)
    get_neighbor('g2', 'UpperClothes', 'hres', 50)
    get_neighbor('g2', 'Pants', 'hres', 50)
    get_neighbor('g3', 'UpperClothes', 'hres', 50)
    get_neighbor('g3', 'Pants', 'hres', 50)
    get_neighbor('g4', 'UpperClothes', 'hres', 50)
    get_neighbor('g4', 'Pants', 'hres', 50)
    get_neighbor('g5', 'UpperClothes', 'hres', 50)
    get_neighbor('g5', 'Pants', 'hres', 50)
    get_neighbor('g6', 'UpperClothes', 'hres', 50)
    get_neighbor('g6', 'Pants', 'hres', 50)
    get_neighbor('g7', 'UpperClothes', 'hres', 50)
    get_neighbor('g7', 'Pants', 'hres', 50)




    get_neighbor('g1', 'UpperClothes', 'hres', 20)
    get_neighbor('g1', 'Pants', 'hres', 20)
    get_neighbor('g2', 'UpperClothes', 'hres', 20)
    get_neighbor('g2', 'Pants', 'hres', 20)
    get_neighbor('g3', 'UpperClothes', 'hres', 20)
    get_neighbor('g3', 'Pants', 'hres', 20)
    get_neighbor('g4', 'UpperClothes', 'hres', 20)
    get_neighbor('g4', 'Pants', 'hres', 20)
    get_neighbor('g5', 'UpperClothes', 'hres', 20)
    get_neighbor('g5', 'Pants', 'hres', 20)
    get_neighbor('g6', 'UpperClothes', 'hres', 20)
    get_neighbor('g6', 'Pants', 'hres', 20)
    get_neighbor('g7', 'UpperClothes', 'hres', 20)
    get_neighbor('g7', 'Pants', 'hres', 20)



    get_neighbor('g1', 'UpperClothes', 'lres', 100)
    get_neighbor('g1', 'Pants', 'lres', 100)
    get_neighbor('g2', 'UpperClothes', 'lres', 100)
    get_neighbor('g2', 'Pants', 'lres', 100)
    get_neighbor('g3', 'UpperClothes', 'lres', 100)
    get_neighbor('g3', 'Pants', 'lres', 100)
    get_neighbor('g4', 'UpperClothes', 'lres', 100)
    get_neighbor('g4', 'Pants', 'lres', 100)
    get_neighbor('g5', 'UpperClothes', 'lres', 100)
    get_neighbor('g5', 'Pants', 'lres', 100)
    get_neighbor('g6', 'UpperClothes', 'lres', 100)
    get_neighbor('g6', 'Pants', 'lres', 100)
    get_neighbor('g7', 'UpperClothes', 'lres', 100)
    get_neighbor('g7', 'Pants', 'lres', 100)


    get_neighbor('g1', 'UpperClothes', 'lres', 50)
    get_neighbor('g1', 'Pants', 'lres', 50)
    get_neighbor('g2', 'UpperClothes', 'lres', 50)
    get_neighbor('g2', 'Pants', 'lres', 50)
    get_neighbor('g3', 'UpperClothes', 'lres', 50)
    get_neighbor('g3', 'Pants', 'lres', 50)
    get_neighbor('g4', 'UpperClothes', 'lres', 50)
    get_neighbor('g4', 'Pants', 'lres', 50)
    get_neighbor('g5', 'UpperClothes', 'lres', 50)
    get_neighbor('g5', 'Pants', 'lres', 50)
    get_neighbor('g6', 'UpperClothes', 'lres', 50)
    get_neighbor('g6', 'Pants', 'lres', 50)
    get_neighbor('g7', 'UpperClothes', 'lres', 50)
    get_neighbor('g7', 'Pants', 'lres', 50)




    get_neighbor('g1', 'UpperClothes', 'lres', 20)
    get_neighbor('g1', 'Pants', 'lres', 20)
    get_neighbor('g2', 'UpperClothes', 'lres', 20)
    get_neighbor('g2', 'Pants', 'lres', 20)
    get_neighbor('g3', 'UpperClothes', 'lres', 20)
    get_neighbor('g3', 'Pants', 'lres', 20)
    get_neighbor('g4', 'UpperClothes', 'lres', 20)
    get_neighbor('g4', 'Pants', 'lres', 20)
    get_neighbor('g5', 'UpperClothes', 'lres', 20)
    get_neighbor('g5', 'Pants', 'lres', 20)
    get_neighbor('g6', 'UpperClothes', 'lres', 20)
    get_neighbor('g6', 'Pants', 'lres', 20)
    get_neighbor('g7', 'UpperClothes', 'lres', 20)
    get_neighbor('g7', 'Pants', 'lres', 20)


