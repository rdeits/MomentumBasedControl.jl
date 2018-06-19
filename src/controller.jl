mutable struct MomentumBasedController{N, O<:MOI.AbstractOptimizer, S<:MechanismState}
    state::S
    result::DynamicsResult{Float64, Float64}
    centroidalframe::CartesianFrame3D
    momentum_matrix::MomentumMatrix{Matrix{Float64}}
    contactconfigurations::Dict{RigidBody{Float64}, ContactConfiguration{N}}
    qpmodel::SimpleQP.Model{Float64, O}
    v̇::Vector{Variable}
    objective::SimpleQP.LazyExpression # buffer to incrementally build the objective function
    initialized::Bool

    function MomentumBasedController{N}(mechanism::Mechanism{Float64}, optimizer::O) where {N, O<:MOI.AbstractOptimizer}
        state = MechanismState(mechanism)
        result = DynamicsResult(mechanism)
        worldframe = root_frame(mechanism)
        centroidalframe = CartesianFrame3D("centroidal")
        nv = num_velocities(state)
        momentum_matrix = MomentumMatrix(worldframe, zeros(3, nv), zeros(3, nv))
        rootframe = root_frame(mechanism)
        contactconfigurations = Dict{RigidBody{Float64}, ContactConfiguration{N}}()
        qpmodel = SimpleQP.Model(optimizer)
        v̇ = [Variable(qpmodel) for _ = 1 : nv]
        objective = SimpleQP.LazyExpression(identity, zero(QuadraticFunction{Float64}))
        new{N, O, typeof(state)}(state, result, centroidalframe, momentum_matrix, contactconfigurations, qpmodel, v̇, objective, false)
    end
end

centroidal_frame(controller::MomentumBasedController) = controller.centroidalframe

function (controller::MomentumBasedController)(τ::AbstractVector, t::Number, x::Union{<:Vector, <:MechanismState})
    if !controller.initialized
        initialize!(controller)
        controller.initialized = true
    end

    qpmodel = controller.qpmodel
    state = controller.state
    result = controller.result
    contactconfigurations = controller.contactconfigurations
    contactwrenches = result.contactwrenches

    copyto!(state, x)
    solve!(qpmodel)
    @assert terminationstatus(qpmodel) == MOI.Success
    @assert primalstatus(qpmodel) == MOI.FeasiblePoint
    result.v̇ .= value.(qpmodel, controller.v̇)
    for body in keys(controller.contactconfigurations)

    end
    # com = center_of_mass(state)
    # centroidal_to_world = Transform3D(controller.centroidalframe, com.frame, com.v)
    # map!(@closure(wrench -> transform(wrench, centroidal_to_world)), values(contactwrenches), values(contactwrenches)) # TODO: desirable?
    inverse_dynamics!(τ, result.jointwrenches, result.accelerations, state, result.v̇)#, contactwrenches)
    τ
end

function addtask!(controller::MomentumBasedController, task::AbstractMotionTask)
    model = controller.qpmodel
    state = controller.state
    v̇ = controller.v̇
    @constraint(model, task_error(task, model, state, v̇) == zeros(dimension(task)))
    nothing
end

function addtask!(controller::MomentumBasedController, task::AbstractMotionTask, weight::Number)
    e = add_task_error_slack_variables!(controller, task)
    controller.objective = @expression controller.objective + weight * (e ⋅ e)
    e
end

function addtask!(controller::MomentumBasedController, task::AbstractMotionTask, weight::AbstractMatrix)
    e = add_task_error_slack_variables!(controller, task)
    controller.objective = @expression controller.objective + e' * weight * e
    e
end

function add_task_error_slack_variables!(controller::MomentumBasedController, task::AbstractMotionTask)
    model = controller.qpmodel
    state = controller.state
    v̇ = controller.v̇
    e = [Variable(model) for _ = 1 : dimension(task)]
    @constraint(model, e == task_error(task, model, state, v̇))
    e
end

function regularize!(controller::MomentumBasedController, joint::Joint, weight)
    task = JointAccelerationTask(joint)
    setdesired!(task, zeros(num_velocities(joint)))
    addtask!(controller, task, weight)
    task
end

function addcontact!(controller::MomentumBasedController{N}, body::RigidBody{Float64}, point::ContactPoint) where N
    model = controller.qpmodel
    ρ = SVector(ntuple(_ -> Variable(model)), Val(N))
    config = ContactConfiguration(point, ρ)
    body_configs = get!(Vector{ContactConfiguration{N}}, controller.contactconfigurations, body)
    push!(body_configs, config)
    @constraint(model, ρ >= zeros(N))
    weight = Parameter{Float64}(() -> config.weight, model)
    controller.objective = @expression controller.objective + weight * (ρ ⋅ ρ)


    # TODO: maxnormalforce
    # TODO: objective term
    config
end

@noinline function initialize!(controller::MomentumBasedController)
    setobjective!(controller)
    mechanism = controller.state.mechanism
    if any(isfloating, out_joints(root_body(mechanism), mechanism))
        add_wrench_balance_constraint!(controller)
    end
end

function setobjective!(controller::MomentumBasedController)
    @objective(controller.qpmodel, Minimize, controller.objective)
end

function add_wrench_balance_constraint!(controller::MomentumBasedController{N}) where N
    # TODO: repeated computation of A if there's a MomentumRateTask
    qpmodel = controller.qpmodel
    state = controller.state
    mechanism = state.mechanism
    fg = mass(mechanism) * mechanism.gravitational_acceleration
    v̇ = controller.v̇
    A = Parameter(A -> momentum_matrix!(A, state), controller.momentum_matrix, qpmodel)
    Ȧv = Parameter{Wrench{Float64}}(() -> momentum_rate_bias(state), qpmodel)
    Wg = Parameter{Wrench{Float64}}(() -> Wrench(center_of_mass(state) × fg, fg), qpmodel)

    torque = @expression angular(Wg)
    force = @expression linear(Wg)
    for body_configs in values(controller.contactconfigurations)
        for config in body_configs
            ρ = config.ρ
            basis = wrenchbasis(config, state, qpmodel)
            torque = @expression torque + angular(basis) * ρ
            force = @expression force + angular(basis) * ρ
        end
    end

    @constraint(qpmodel, angular(A) * v̇ + angular(Ȧv) == torque)
    @constraint(qpmodel, linear(A) * v̇ + linear(Ȧv) == force)

    nothing
end
