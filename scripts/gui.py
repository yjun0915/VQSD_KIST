import yaml

configs = []

dims = [3, 4, 5]
lambda_vals = [1, 0.87, 0.8]
overlaps = [0.25, 0.5, 0.75]
optimizers = ['COBYLA', 'Nelder-Mead']
tols = [0.01, 0.001]
for optimizer in optimizers:
    for tol in tols:
        for overlap in overlaps:
            for dim in dims:
                configs.append({'dim': dim,
                                'overlap': overlap,
                                'lambda_val': lambda_vals[dim-3],
                                'optimizer': optimizer,
                                'tol': tol})

with open('../config/recipe.yaml', 'w') as f:
    yaml.dump({"settings": configs}, f)