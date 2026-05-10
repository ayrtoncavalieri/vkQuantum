#include "VulkanCompute.h"
#include <iostream>

int main(int argc, const char* argv[]) {
    try {
        VulkanCompute vc("shaders/comp.spv");

        // 1. List available GPUs
        const auto gpus = vc.enumerateGPUs();
        std::cout << "Available GPUs:\n";
        for (size_t i = 0; i < gpus.size(); ++i)
            std::cout << "  [" << i << "] " << gpus[i] << "\n";

        // 2. Select one
        vc.selectGPU(0);
        std::cout << "\nUsing: " << gpus[0] << "\n\n";

        // 3. Build some 2×2 complex matrices
        //    A = | 1+0i  0+1i |     B = | 2+0i  0+0i |
        //        | 0-1i  1+0i |         | 0+0i  2+0i |
        ComplexMatrix A(2, 2), B(2, 2);
        A.at(0, 0) = { 1.f,  0.f }; A.at(0, 1) = { 0.f,  1.f };
        A.at(1, 0) = { 0.f, -1.f }; A.at(1, 1) = { 1.f,  0.f };

        B.at(0, 0) = { 2.f,  0.f }; B.at(0, 1) = { 0.f,  0.f };
        B.at(1, 0) = { 0.f,  0.f }; B.at(1, 1) = { 2.f,  0.f };

        // 4. Matrix multiply
        ComplexMatrix C = vc.multiply(A, B);
        std::cout << "A × B:\n";
        for (uint32_t r = 0; r < C.rows; ++r) {
            for (uint32_t c = 0; c < C.cols; ++c)
                std::cout << "  (" << C.at(r, c).real() << ", " << C.at(r, c).imag() << "i) ";
            std::cout << "\n";
        }

        // 5. Kronecker product
        ComplexMatrix K = vc.kronecker(A, B);
        std::cout << "\nA ⊗ B (" << K.rows << "×" << K.cols << "):\n";
        for (uint32_t r = 0; r < K.rows; ++r) {
            for (uint32_t c = 0; c < K.cols; ++c)
                std::cout << "  (" << K.at(r, c).real() << ", " << K.at(r, c).imag() << "i) ";
            std::cout << "\n";
        }

        // 6. Matrix addition
        ComplexMatrix S = vc.add(A, B);
        std::cout << "\nA + B (" << S.rows << "+" << S.cols << "):\n";
        for (uint32_t r = 0; r < S.rows; ++r) {
            for (uint32_t c = 0; c < S.cols; ++c)
                std::cout << "  (" << S.at(r, c).real() << ", " << S.at(r, c).imag() << "i) ";
            std::cout << "\n";
        }

        // 7. Scalar multiplication A * (0.5 + 0.5i)
        std::complex<float> scalar = { 0.5f, 0.5f }; // 0.5 + 0.5i
        ComplexMatrix M = vc.multiplyByScalar(scalar, A);
        std::cout << "\nA * (0.5 + 0.5i) (" << M.rows << "+" << M.cols << "):\n";
        for (uint32_t r = 0; r < M.rows; ++r) {
            for (uint32_t c = 0; c < M.cols; ++c)
                std::cout << "  (" << M.at(r, c).real() << ", " << M.at(r, c).imag() << "i) ";
            std::cout << "\n";
        }
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
