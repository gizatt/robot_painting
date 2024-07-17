import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
import skfmm
import open3d
import networkx as nx


def get_brush_stroke_masked_sd(im: np.ndarray, threshold: float, brush_width: int):
    """
    Input should be a binary image. an SDF to the interior.

    We'll threshold it, compute an SDF, and return
    an image that masks the SDF to just those pixels
    near a brush width of a boundary or at a local min. These
    pixels correspond to points on a path that a brush should
    take to draw part of the image.
    """
    # Differentiate the inside / outside region
    phi = np.int64(im >= threshold)

    # The array will go from - 1 to 0. Add 0.5(arbitrary) so there 's a 0 contour.
    phi = np.where(phi, 0, -1) + 0.5
    sd = skfmm.distance(phi, dx=1)

    boundary_pixels = np.isclose(sd, -brush_width/2, atol=0.5)
    gx, gy = np.gradient(sd)
    grad_norm = gx * gx + gy * gy
    min_pixels = (
        np.isclose(grad_norm, 0.0, atol=0.5) * (sd >= -brush_width/2) * (sd < 0.0)
    )
    boundary_to_draw = np.logical_or(min_pixels, boundary_pixels)
    # Pack the signed distance into those values. Flip sign so they'r epositive
    return boundary_to_draw * -sd

def generate_best_run_from_start(graph, component, start_node):
    """
    Run Dijkstra's to find the minimum-cost path to all nodes in the component,
    where the new cost at a node is computed in a special way based on the potential
    predecessor.

    Then tries to find the best balance of longest and straightest path.
    """
    min_cost_at_nodes = {node: np.inf for node in component}
    min_cost_at_nodes[start_node] = 0.0
    momentum_at_nodes = {start_node: np.zeros(2)}
    predecessor_at_nodes = {start_node: None}
    path_lengths_at_nodes = {start_node: 0}
    unexpanded_nodes = component.copy()  # Shallow list copy, don't need full deepcopy
    adjacency_dict = dict(graph.adjacency())
    graph_pos_dict = nx.get_node_attributes(graph, "pos")

    print("Starting")
    while len(unexpanded_nodes) > 0:
        # Grab the lowest-cost unexpanded node.
        update_node = min(
            [
                (node, value)
                for (node, value) in min_cost_at_nodes.items()
                if node in unexpanded_nodes
            ],
            key=lambda x: x[1],
        )[0]
        unexpanded_nodes.remove(update_node)

        # Update min cost of its neighbors.
        neighbors = [n for n in adjacency_dict[update_node] if n in unexpanded_nodes]
        if len(neighbors) == 0:
            continue

        this_pos = graph_pos_dict[update_node]
        if update_node not in momentum_at_nodes:
            print(f"Execution error -- bad update node {update_node}, {len(unexpanded_nodes)} nodes left, {len(component)} total.")
            break
        this_momentum = momentum_at_nodes[update_node]
        this_cost = min_cost_at_nodes[update_node]
        this_path_length = path_lengths_at_nodes[update_node]
        vecs = [graph_pos_dict[neighbor] - this_pos for neighbor in neighbors]
        vecs = [vec / np.linalg.norm(vec) for vec in vecs]
        if np.linalg.norm(this_momentum) > 0:
            angles = np.arccos([np.sum(vec * this_momentum) for vec in vecs])
        else:
            angles = np.zeros(len(vecs))

        for neighbor, angle, vec in zip(neighbors, angles, vecs, strict=True):
            new_cost = this_cost + angle
            if new_cost < min_cost_at_nodes[neighbor]:
                min_cost_at_nodes[neighbor] = new_cost
                momentum_at_nodes[neighbor] = vec
                predecessor_at_nodes[neighbor] = update_node
                path_lengths_at_nodes[neighbor] = this_path_length + 1

    # Now determine the best path, as a combination of straightness and
    # path length/
    def sorting_function(x):
        node, path_length = x
        cost = min_cost_at_nodes[node]
        return cost - path_length * 0.1

    # Find node that terminates longest run
    best_run = []
    current_node = min(path_lengths_at_nodes.items(), key=sorting_function)[0]
    while current_node is not None:
        best_run.append(current_node)
        current_node = predecessor_at_nodes[current_node]

    return best_run[::-1]


def generate_best_runs_for_component(graph, component):
    # Start runs at all nodes with degree 1, or 1 random node with degree 2
    # if no nodes have degree 1.
    degrees = graph.degree(component)
    min_degree = min([degree for _, degree in degrees])
    start_nodes = [node for node, degree in degrees if degree == min_degree]
    runs = [
        generate_best_run_from_start(graph, component, start_node)
        for start_node in start_nodes
    ]
    return runs


def convert_masked_sd_to_strokes(masked_sd: np.ndarray, point_spacing: int = 5):
    nonzeros = np.c_[np.nonzero(masked_sd)]
    # Use Open3D for downsampling. Pack the SD into the z dim so it gets carried through.
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(
        np.c_[nonzeros, -masked_sd[nonzeros[:, 0], nonzeros[:, 1]]]
    )
    pcd = pcd.voxel_down_sample(voxel_size=point_spacing)
    downsampled_pts = np.asarray(pcd.points)

    # Reform a new pointcloud to get nearest neighbors. We only want NN in xy, so remove the z component.
    pcd_for_nn = open3d.geometry.PointCloud()
    pcd_for_nn.points = open3d.utility.Vector3dVector(
        downsampled_pts * np.array([1.0, 1.0, 0])
    )
    pcd_tree = open3d.geometry.KDTreeFlann(pcd_for_nn)

    # Form a graph indicating local connectivity of points.
    graph = nx.Graph()
    for index, (x, y, s) in enumerate(downsampled_pts):
        graph.add_node(index, pos=np.array([x, y]), sd=s)
        [_, neighbor_indices, _] = pcd_tree.search_radius_vector_3d(
            (x, y, 0.0), point_spacing * 2
        )
        for neighbor_index in neighbor_indices:
            if neighbor_index != index:
                graph.add_edge(index, neighbor_index)

    # Generate runs through the graph, which will start at endpoints with low degree and run as
    # long/straight as possible.
    connected_components = list(nx.connected_components(graph))
    runs = []
    for component in connected_components:
        runs += generate_best_runs_for_component(graph, component)

    # Finally, convert each run into a 3xN array of [x, y, size].
    graph_pos_dict = nx.get_node_attributes(graph, "pos")
    graph_sd_dict = nx.get_node_attributes(graph, "sd")

    runs_arrays = []
    for run in runs:
        xys = np.array([graph_pos_dict[n] for n in run]).T
        sds = np.array([[graph_sd_dict[n] for n in run]])
        runs_arrays.append(np.r_[xys, sds])
    return runs_arrays


if __name__ == "__main__":
    img_bgr = cv2.imread("eye.png")
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) 
    
    
    brush_width = 10.0

    start_time = time.time()
    masked_sd = get_brush_stroke_masked_sd(img_gray, threshold=0.5, brush_width=10.0)
    sd_time = time.time()
    strokes = convert_masked_sd_to_strokes(masked_sd, point_spacing=5)
    strokes_time = time.time()

    print(f"Took %f sec for signed distance, %f sec for stroke gen." % (sd_time-start_time, strokes_time-sd_time))

    plt.figure()
    plt.subplot(3, 1, 1)
    plt.imshow(img_gray)
    plt.subplot(3, 1, 2)
    plt.imshow(masked_sd)
    plt.subplot(3, 1, 3)
    
    print([x.shape[1] for x in strokes])
    plt.imshow(img_bgr)
    cmap = plt.get_cmap("jet")
    for k, stroke in enumerate(strokes):
      plt.plot(stroke[1, :], stroke[0, :], c=cmap(float(k)/len(strokes)), linewidth=brush_width, alpha=0.5)
    plt.show()
    