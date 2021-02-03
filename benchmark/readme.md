# Demo with single conditions

Key Configurations
```Julia
ns = 100;  # number of nodes / species
tfinal = 20.0;
ntotal = 40;  # number of samples for each perturbation

u0 = zeros(ns);
μ = rand(ns);

p_gold = gen_network(ns; weight_params=(0.0, 1.0), sparsity=0.9);

# pay attentions to this one, we can discuss if we need this one (encourage sparcity)
p_gold = sign.(p_gold) .* clamp.(abs.(p_gold), 0.1, Inf);
```

Put `\mu` outside of the function could reduce the number of variables to half and thus accelrate the ODE solver.

## Results

Loss and gradient norm

![loss](./figs/loss_grad.png)

prediction
![pred](./figs/pred.png)
