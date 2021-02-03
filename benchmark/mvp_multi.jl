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
is_restart = false;
n_epoch = 10000;
n_plot = 20;  # frequency of callback
alg = Tsit5();

# opt = ADAMW(5.f-3, (0.9, 0.999), 1.f-4);
opt = Flux.Optimiser(ExpDecay(5e-3, 0.2, 300 * n_exp_train, 1e-4),
                     ADAMW(0.005, (0.9, 0.999), 1.f-6));
ns = 100;  # number of nodes / species
tfinal = 20.0;
ntotal = 20;  # number of samples for each perturbation
batch_size = 8;  # STEER

n_exp_train = 20;
n_exp_val = 10;
n_exp = n_exp_train + n_exp_val;
noise = 0.01;
grad_max = 1.e2;

function gen_network(m; weight_params=(0., 1.), sparsity=0.)
    w = rand(Uniform(weight_params[1], weight_params[2]), (m, m))
    p = [sparsity, 1 - sparsity]
    w .*= sample([0, 1], weights(p), (m, m), replace=true)
    α = abs.(rand(Uniform(weight_params[1], weight_params[2]), (m)))
    return hcat(α, w)
end

function show_network(p)
    println("p_gold")
    show(stdout, "text/plain", round.(p_gold, digits=2))
    println("\np_learned")
    show(stdout, "text/plain", round.(p, digits=2))
end

Random.seed!(1234);
u0 = zeros(ns);
p_gold = gen_network(ns; weight_params=(0.0, 1.0), sparsity=0.9);
p = gen_network(ns; weight_params=(0.0, 0.01), sparsity=0);

# show_network(p)

function cellbox!(du, u, p, t)
    du .= tanh.(view(p, :, 2:ns + 1) * u - μ) - view(p, :, 1) .* u
end

tspan = (0, tfinal);
ts = 0:tspan[2] / ntotal:tspan[2];
ts = ts[2:end];
prob = ODEProblem(cellbox!, u0, tspan, saveat=ts);

function max_min(ode_data)
    return maximum(ode_data, dims=2) .- minimum(ode_data, dims=2)
end

# Generate data sets
μ_list = rand(n_exp, ns);
ode_data_list = zeros(Float64, (n_exp, ns, ntotal));
yscale_list = [];
for i = 1:n_exp
    global μ = μ_list[i, 1:ns]
    prob = ODEProblem(cellbox!, u0, tspan, saveat=ts)
    ode_data = Array(solve(prob, alg, u0=u0, p=p_gold))
    
    ode_data += randn(size(ode_data)) .* ode_data .* noise
    ode_data_list[i, :, :] = ode_data
    
    push!(yscale_list, max_min(ode_data))
end
yscale = maximum(hcat(yscale_list...), dims=2);

function predict_neuralode(u0, p, i_exp=1, batch=ntotal)
    global μ = μ_list[i_exp, 1:ns]
    _prob = remake(prob, p=p, tspan=[0, ts[batch]])
    pred = Array(solve(_prob, alg, saveat=ts[1:batch], sensalg=QuadratureAdjoint()))
    return pred
end
predict_neuralode(u0, p, 1)

function loss_neuralode(p, i_exp=1, batch=ntotal)
    pred = predict_neuralode(u0, p, i_exp, batch)
    loss = mae(@views(ode_data_list[i_exp, :, 1:batch]), pred)
    return loss
end
loss_neuralode(p, 1)

cbi = function (p, i_exp)
    ode_data = ode_data_list[i_exp, :, :]
    pred = predict_neuralode(u0, p, i_exp)
    l_plt = []
    for i in 1:minimum([10, ns])
        plt = scatter(ts, ode_data[i,:], label="data");
        plot!(plt, ts, pred[i,:], label="pred");
        ylabel!(plt, "x$i")
        xlabel!(plt, "Time")
        push!(l_plt, plt)
    end
    plt_all = plot(l_plt..., legend=false, size=(1000, 1000));
    png(plt_all, string("figs/i_exp_", i_exp))
    return false
end

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
        
        l_exp = randperm(n_exp)[1:1]
        println("update plot for ", l_exp)
        for i_exp in l_exp
            cbi(p, i_exp)
        end

        plt_loss = plot(l_loss_train, yscale=:log10, label="train")
        plot!(plt_loss, l_loss_val, yscale=:log10, label="val")
        plt_grad = plot(l_grad, yscale=:log10, label="grad_norm")
        xlabel!(plt_loss, "Epoch")
        xlabel!(plt_grad, "Epoch")
        ylabel!(plt_loss, "Loss")
        ylabel!(plt_grad, "Gradient Norm")
        # ylims!(plt_loss, (-Inf, 1))
        plt_all = plot([plt_loss, plt_grad]..., legend=:top)
        png(plt_all, "figs/loss_grad")

        @save "./checkpoint/mymodel.bson" p opt l_loss_train l_loss_val l_grad iter;
    end
    iter += 1;
end

if is_restart
    @load "./checkpoint/mymodel.bson" p opt l_loss_train l_loss_val l_grad iter;
    iter += 1;
    # opt = ADAMW(1.f-4, (0.9, 0.999), 1.f-6);
end

epochs = ProgressBar(iter:n_epoch);
loss_epoch = zeros(Float32, n_exp);
grad_norm = zeros(Float32, n_exp_train);
for epoch in epochs
    global p
    for i_exp in randperm(n_exp_train)
        batch = rand(batch_size:ntotal)
        grad = Zygote.gradient(x -> loss_neuralode(x, i_exp, batch), p)[1]
        grad_norm[i_exp] = norm(grad, 2)
        if grad_norm[i_exp] > grad_max
            grad = grad ./ grad_norm[i_exp] .* grad_max
        end
        update!(opt, p, grad)
    end
    for i_exp in 1:n_exp
        loss_epoch[i_exp] = loss_neuralode(p, i_exp)
    end
    loss_train = mean(loss_epoch[1:n_exp_train]);
    loss_val = mean(loss_epoch[n_exp_train + 1:end]);
    g_norm = mean(grad_norm)
    set_description(epochs, string(@sprintf("Loss train %.4e lr %.1e", loss_train, opt[1].eta)))
    cb(p, loss_train, loss_val, g_norm);
end

for i_exp in 1:n_exp
    cbi(p, i_exp)
end

# show_network(p)