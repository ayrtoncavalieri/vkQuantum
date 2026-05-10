#include "VulkanCompute.h"
#include <iostream>

using namespace std;

int main(int argc, const char* argv[]) {
    size_t gpuSelection;
    try {
        VulkanCompute vc("shaders/comp.spv");

        // 1. List available GPUs
        const auto gpus = vc.enumerateGPUs();
        cout << "Available GPUs:\n";
        for (size_t i = 0; i < gpus.size(); ++i)
            cout << "  [" << i << "] " << gpus[i] << "\n";

        // 2. Select one
		cout << "\nSelect GPU index: ";
		cin >> gpuSelection;
        cout << "\nUsing: " << gpus[gpuSelection] << "\n\n";
        vc.selectGPU(gpuSelection);

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
        cout << "A × B:\n";
        for (uint32_t r = 0; r < C.rows; ++r) {
            for (uint32_t c = 0; c < C.cols; ++c)
                cout << "  (" << C.at(r, c).real() << ", " << C.at(r, c).imag() << "i) ";
            cout << "\n";
        }

        // 5. Kronecker product
        ComplexMatrix K = vc.kronecker(A, B);
        cout << "\nA ⊗ B (" << K.rows << "×" << K.cols << "):\n";
        for (uint32_t r = 0; r < K.rows; ++r) {
            for (uint32_t c = 0; c < K.cols; ++c)
                cout << "  (" << K.at(r, c).real() << ", " << K.at(r, c).imag() << "i) ";
            cout << "\n";
        }

        // 6. Matrix addition
        ComplexMatrix S = vc.add(A, B);
        cout << "\nA + B (" << S.rows << "+" << S.cols << "):\n";
        for (uint32_t r = 0; r < S.rows; ++r) {
            for (uint32_t c = 0; c < S.cols; ++c)
                cout << "  (" << S.at(r, c).real() << ", " << S.at(r, c).imag() << "i) ";
            cout << "\n";
        }

        // 7. Scalar multiplication A * (0.5 + 0.5i)
        complex<float> scalar = { 0.5f, 0.5f }; // 0.5 + 0.5i
        ComplexMatrix M = vc.multiplyByScalar(scalar, A);
        cout << "\nA * (0.5 + 0.5i) (" << M.rows << "+" << M.cols << "):\n";
        for (uint32_t r = 0; r < M.rows; ++r) {
            for (uint32_t c = 0; c < M.cols; ++c)
                cout << "  (" << M.at(r, c).real() << ", " << M.at(r, c).imag() << "i) ";
            cout << "\n";
        }
    }
    catch (const std::exception& e) {
        cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
