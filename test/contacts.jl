@testset "ContactInfo" begin
    frame = CartesianFrame3D()
    point = Point3D(frame, 1., 2., 3.)
    normal = FreeVector3D(frame, normalize(SVector(1., 3., 2.)))
    μ = 0.5
    contactinfo = ContactInfo(point, normal, μ)

    for N = 1 : 2 : 10
        basis = MBC.wrenchbasis(contactinfo, Val(N))
        for i = 1 : N
            ρ = zeros(N)
            ρ[i] = 1
            wrench = Wrench(basis, ρ)
            @test wrench.frame == frame
            force = FreeVector3D(frame, linear(wrench))
            torque = FreeVector3D(frame, angular(wrench))
            @test torque ≈ point × force atol = 1e-12
            normalforce = force ⋅ normal
            @test normalforce > 0
            @test norm(force - normalforce * normal) <= μ * normalforce + 1e-12
        end
    end
    let contactinfo = contactinfo
        @test_noalloc MBC.wrenchbasis(contactinfo, Val(4))
    end
end

@testset "ContactSettings" begin
    frame = CartesianFrame3D()
    point = Point3D(frame, 1., 2., 3.)
    normal = FreeVector3D(frame, normalize(SVector(1., 3., 2.)))
    μ = 0.5
    contactinfo = ContactInfo(point, normal, μ)
    settings = ContactSettings{4}(contactinfo)
    @test !MBC.isenabled(settings)
    settings.maxnormalforce = 1.0
    @test MBC.isenabled(settings)
    basis1 = MBC.wrenchbasis(settings, eye(Transform3D{Float64}, frame))
    basis2 = MBC.wrenchbasis(contactinfo, Val(4))
    @test angular(basis1) == angular(basis2)
    @test linear(basis1) == linear(basis2)
end

@testset "Wrench expression allocations" begin
    model = MockModel()

    frame = CartesianFrame3D()
    point = Point3D(frame, 1., 2., 3.)
    normal = FreeVector3D(frame, normalize(SVector(1., 3., 2.)))
    μ = 0.5
    contactinfo = ContactInfo(point, normal, μ)
    settings = ContactSettings{4}(contactinfo)

    ρ = map(Variable, 1 : MBC.num_basis_vectors(settings))
    tf = eye(Transform3D{Float64}, frame)
    wrenchbasis = Parameter(@closure(() -> MBC.wrenchbasis(settings, tf)), model)

    torque = @expression angular(wrenchbasis) * ρ
    force = @expression linear(wrenchbasis) * ρ

    @test_noalloc (setdirty!(model); torque())
    @test_noalloc (setdirty!(model); force())
end
