using LinearAlgebra

const X = ComplexF64[0   1; 1 0]
const Y = ComplexF64[0 -im;im 0]
const Z = ComplexF64[1   0;0 -1]

Z1 = [1, 0]
Z2 = [0, 1]
X1 = 1 / sqrt(2) * [1, 1]
X2 = 1 / sqrt(2) * [1, -1]
Y1 = 1 / sqrt(2) * [1, im]
Y2 = 1 / sqrt(2) * [1, -im]

M0 = 1/3 * Z1 * Z1'
M1 = 1/3 * Z2 * Z2'
M2 = 1/3 * X1 * X1'
M3 = 1/3 * X2 * X2'
M4 = 1/3 * Y1 * Y1'
M5 = 1/3 * Y2 * Y2'

T = ComplexF64[
    1     1/2   1/2   1;
    1/2   1     1/2   1;
    1/2   1/2   1     1;
    1     1     1     6;]

M = zeros(ComplexF64, 2, 2, 4)
M[:, :, 1] = 1/3 * [1 0;0 0]
M[:, :, 2] = 1/6 * [1 1;1 1]
M[:, :, 3] = 1/6 * [1 -im;im 1]
M[:, :, 4] = 1/3 * ([0 0;0 1] + 0.5 * [1 -1;-1 1] + 0.5 * [1 im;-im 1])

function overlap_matrix(M)
    T = zeros(eltype(M), size(M, 3), size(M, 3))

    for i in 1:size(M, 3)
        for j in 1:size(M, 3)
            T[i, j] = tr(M[:, :, i] * M[:, :, j])
        end
    end
    T
end

overlap_matrix(M) * 9

T = inv(overlap_matrix(M))

samples = rand(1:4, 3, 20)


function contract_TM_op(T, M, op)
    out = zeros(2, 2, 4)
    for i in 1:size(T, 1)
        for j in 1:size(T, 2)
            out[:, :, i] = T[i, j] * M[:, :, j]
        end
    end
    out
end

P = rand(1<<4)

function rho(samples)
    ρ = zeros(1<<size(samples, 1), 1<<size(samples, 1))

    for k in 1:size(samples, 2)
        ρ += kronsum(tm[:, :, i] for i in samples[:, k])
    end

    ρ
end
