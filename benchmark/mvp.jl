using OrdinaryDiffEq, Flux, Optim, Random, Plots
using DiffEqSensitivity
using Zygote
using ForwardDiff
using LinearAlgebra, Statistics
using ProgressBars, Printf
using Flux.Optimise: update!, ExpDecay
using Flux.Losses: mae
using Distributions
using StatsBase
using BSON: @save, @load

Random.seed!(1234);

# Arguments
is_restart = true;
n_epoch = 10000;
n_plot = 10;  # frequency of callback
alg = Tsit5();

opt = ADAMW(5.f-3, (0.9, 0.999), 1.f-4);
ns = 100;  # number of nodes / species
tfinal = 40.0;
ntotal = 40;  # number of samples for each perturbation
batch_size = 40;

function gen_network(m; weight_params=(0., 1.), sparsity=0.)
    w = rand(Normal(weight_params[1], weight_params[2]), (m, m))
    p = [sparsity, 1 - sparsity]
    w .*= sample([0, 1], weights(p), (m, m), replace=true)
    α = abs.(rand(Normal(weight_params[1], weight_params[2]), (m)))
    return hcat(α, w)
end

Random.seed!(1234);
u0 = zeros(ns);
μ = rand(ns);
p_gold = gen_network(ns; weight_params=(0.0, 1.0), sparsity=0.9);

# pay attentions to this one, we can discuss if we need this one (encourage sparcity)
# p_gold = sign.(p_gold) .* clamp.(abs.(p_gold), 0.1, Inf);
p_gold = clamp.(abs.(p_gold), 0, 1.0);

p = gen_network(ns; weight_params=(0.0, 0.1), sparsity=0);

function cellbox!(du, u, p, t)
    du .= tanh.(view(p, :, 2:ns + 1) * u - μ) - view(p, :, 1) .* u
end

tspan = (0, tfinal);
ts = 0:tspan[2] / ntotal:tspan[2];
ts = ts[2:end];
prob = ODEProblem(cellbox!, u0, tspan, saveat=ts);

# synthesize data
sol = solve(prob, alg, u0=u0, p=p_gold);
ode_data = Array(sol);
yscale = maximum(ode_data, dims=2);

function predict_neuralode(u0, p, sample=ntotal)
    _prob = remake(prob, u0=u0, p=p, tspan=[0, ts[sample]])
    pred = Array(solve(_prob, alg, saveat=ts[1:sample], sensalg=QuadratureAdjoint()))
    return pred
end
predict_neuralode(u0, p);
using BenchmarkTools

function loss_neuralode(p, sample=ntotal)
    pred = predict_neuralode(u0, p, sample)
    loss = mae(ode_data[:, 1:sample], pred)
    return loss
end
loss_neuralode(p)

# Zygote.gradient(x -> loss_neuralode(x), p)
# @benchmark Zygote.gradient(x -> loss_neuralode(x), p)
# @benchmark ForwardDiff.gradient(x -> loss_neuralode(x), p)

l_loss_train = []
l_loss_val = []
l_grad = []
iter = 1
cb = function (p, loss_train, loss_val, g_norm)
    global l_loss_train, l_loss_val, l_grad, iter
    push!(l_loss_train, loss_train)
    push!(l_loss_val, loss_val)
    push!(l_grad, g_norm)

    if iter % n_plot == 0
        # println("\np_gold")
        # show(stdout, "text/plain", round.(p_gold, digits=2))
        # println("\np_learned")
        # show(stdout, "text/plain", round.(p, digits=2))
        pred = predict_neuralode(u0, p)
        l_plt = []
        for i in 1:minimum([10, ns])
            plt = scatter(ts, ode_data[i,:], label="data");
            plot!(plt, ts, pred[i,:], label="pred");
            ylabel!(plt, "x$i")
            xlabel!(plt, "Time")
            push!(l_plt, plt)
        end
        plt_all = plot(l_plt..., legend=false, size=(1000, 1000));
        png(plt_all, string("figs/pred"))

        plt_loss = plot(l_loss_train, yscale=:log10, label="Training")
        # plot!(plt_loss, l_loss_val, yscale=:log10, label="Validation")
        plt_grad = plot(l_grad, yscale=:log10, label="grad_norm")
        xlabel!(plt_loss, "Epoch")
        xlabel!(plt_grad, "Epoch")
        ylabel!(plt_loss, "Loss")
        ylabel!(plt_grad, "Gradient Norm")
        # ylims!(plt_loss, (-Inf, 1e0))
        plt_all = plot([plt_loss, plt_grad]..., legend=:top)
        png(plt_all, "figs/loss_grad")

        @save "./checkpoint/mymodel.bson" p opt l_loss_train l_loss_val iter;
    end
    iter += 1;
end

if is_restart
    @load "./checkpoint/mymodel.bson" p opt l_loss_train l_loss_val iter;
    iter += 1;
    # opt = ADAMW(5.f-4, (0.9, 0.999), 1.f-4);
end

epochs = ProgressBar(iter:n_epoch);
for epoch in epochs
    global p
    sample = rand(batch_size:ntotal)   # STEER paper
    loss = loss_neuralode(p)
    # grad = ForwardDiff.gradient(x -> loss_neuralode(x), p)
    grad = Zygote.gradient(x -> loss_neuralode(x, sample), p)[1]
    grad_norm = norm(grad, 2)
    update!(opt, p, grad)
    set_description(epochs, string(@sprintf("Loss train %.2e gnorm %.1e lr %.1e", loss, grad_norm, opt[1].eta)))
    cb(p, loss, loss, grad_norm)
end