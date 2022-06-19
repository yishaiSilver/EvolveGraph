import numpy as np
import torch
import tqdm

def _get_npy_path(feat, mode):
	return f"./edges_{mode}_springs5.npy"

for mode in ["test", "train", "valid"]:
	edge_feat = np.load(_get_npy_path('edges', mode), allow_pickle=False)
	edges = torch.from_numpy(np.array(edge_feat, dtype=np.float32))

	print(edges.shape)

	num_atoms = edges.shape[2]
	off_diag_idx = np.ravel_multi_index(
			np.where(np.ones((num_atoms, num_atoms)) - np.eye(num_atoms)),
			[num_atoms, num_atoms])
	
	# num_timesteps = edges.shape[1]
	num_timesteps = 49
	num_sims = edges.shape[0]
	out_edges = np.zeros((num_sims, num_timesteps, num_atoms * (num_atoms - 1)))
	
	for s in tqdm.tqdm(range(num_sims)):
		for t in range(num_timesteps):
			for index, idx in enumerate(off_diag_idx):
				i = idx // num_atoms
				j = idx % num_atoms

				out_edges[s, t, index] = edges[s, i, j]
		

	# print(out_edges.shape)
	np.save(_get_npy_path('edges_new', mode), out_edges)