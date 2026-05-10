#ifndef VK_COMP
#define VK_COMP

#include <vulkan/vulkan.h>

#include <complex>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// ComplexMatrix
// Row-major matrix of single-precision complex numbers.
// Matches the shader's vec2 layout: x = real, y = imaginary.
// ---------------------------------------------------------------------------
struct ComplexMatrix {
    std::vector<std::complex<float>> data;
    uint32_t rows = 0;
    uint32_t cols = 0;

    ComplexMatrix() = default;

    ComplexMatrix(uint32_t rows, uint32_t cols)
        : rows(rows), cols(cols), data(rows* cols, { 0.f, 0.f }) {
    }

    std::complex<float>& at(uint32_t r, uint32_t c) { return data[r * cols + c]; }
    const std::complex<float>& at(uint32_t r, uint32_t c) const { return data[r * cols + c]; }

    size_t byteSize() const { return data.size() * sizeof(std::complex<float>); }
};

// ---------------------------------------------------------------------------
// VulkanCompute
//
// Wraps all Vulkan state needed to dispatch multi.comp.
//
// Typical usage:
//
//   VulkanCompute vc("multi.comp.spv");
//
//   auto gpus = vc.enumerateGPUs();         // list available devices
//   vc.selectGPU(0);                        // pick one
//
//   ComplexMatrix C = vc.multiply(A, B);    // C = A × B
//   ComplexMatrix K = vc.kronecker(A, B);   // K = A ⊗ B
//
// ---------------------------------------------------------------------------
class VulkanCompute {
public:
    // spirvPath: path to the SPIR-V binary compiled from multi.comp.
    // The shader MUST be compiled with layout(local_size_x_id=0,
    // local_size_y_id=1, local_size_z_id=2) so the workgroup size can be
    // tuned at pipeline-creation time via specialization constants.
    explicit VulkanCompute(const std::string& spirvPath = "shaders/comp.spv");
    ~VulkanCompute();

    VulkanCompute(const VulkanCompute&) = delete;
    VulkanCompute& operator=(const VulkanCompute&) = delete;
    VulkanCompute(VulkanCompute&&) = delete;
    VulkanCompute& operator=(VulkanCompute&&) = delete;

    // Returns the name of every Vulkan-capable device on the system.
    std::vector<std::string> enumerateGPUs() const;

    // Selects which GPU to use.  Recreates all device-level objects if a
    // different GPU was previously selected.  Must be called at least once
    // before multiply() or kronecker().
    void selectGPU(uint32_t index);

    // Returns the index of the currently selected GPU, or UINT32_MAX if none.
    uint32_t selectedGPU() const { return m_selectedGPUIndex; }

    // Complex matrix multiplication: C = A × B
    // Requires A.cols == B.rows.
    ComplexMatrix multiply(const ComplexMatrix& A, const ComplexMatrix& B);

    // Kronecker (tensor) product: C = A ⊗ B
    ComplexMatrix kronecker(const ComplexMatrix& A, const ComplexMatrix& B);

    // Complex matrix addition: C = A + B
    // Requires A and B to have the same dimensions.
    ComplexMatrix add(const ComplexMatrix& A, const ComplexMatrix& B);

    // Complex matrix scalar multiplication: C = s * A
    // Requires s and A to be compatible.
    ComplexMatrix multiplyByScalar(const std::complex<float>& s, const ComplexMatrix& A);

private:
    // -----------------------------------------------------------------------
    // Push-constant block — must mirror the shader layout exactly.
    // -----------------------------------------------------------------------
    struct PushConstants {
        int32_t operation;  // 0 = Kronecker, anything else = matrix multiply
        int32_t linesA;
        int32_t columnsA;
        int32_t linesB;
        int32_t columnsB;
    };

    // -----------------------------------------------------------------------
    // Owned buffer + memory pair
    // -----------------------------------------------------------------------
    struct Buffer {
        VkBuffer       handle = VK_NULL_HANDLE;
        VkDeviceMemory memory = VK_NULL_HANDLE;
        VkDeviceSize   size = 0;
    };

    // -----------------------------------------------------------------------
    // Vulkan handles
    // -----------------------------------------------------------------------
    std::string           m_spirvPath;
    uint32_t              m_selectedGPUIndex = UINT32_MAX;

    VkInstance            m_instance = VK_NULL_HANDLE;
    VkPhysicalDevice      m_physicalDevice = VK_NULL_HANDLE;
    VkDevice              m_device = VK_NULL_HANDLE;
    VkQueue               m_computeQueue = VK_NULL_HANDLE;
    VkCommandPool         m_commandPool = VK_NULL_HANDLE;
    VkDescriptorSetLayout m_descSetLayout = VK_NULL_HANDLE;
    VkDescriptorPool      m_descPool = VK_NULL_HANDLE;
    VkDescriptorSet       m_descSet = VK_NULL_HANDLE;
    VkPipelineLayout      m_pipelineLayout = VK_NULL_HANDLE;
    VkPipeline            m_pipeline = VK_NULL_HANDLE;
    VkShaderModule        m_shaderModule = VK_NULL_HANDLE;

    uint32_t              m_computeQueueFamily = UINT32_MAX;
    uint32_t              m_localSize = 32; // derived from device limits

    std::vector<VkPhysicalDevice> m_physicalDevices;

    // -----------------------------------------------------------------------
    // Initialization helpers
    // -----------------------------------------------------------------------
    void createInstance();
    void createDevice(uint32_t gpuIndex);
    void createCommandPool();
    void createDescriptorSetLayout();
    void createDescriptorPool();
    void createPipelineLayout();
    void createPipeline();
    void destroyDevice();   // tears down all device-level Vulkan objects

    // -----------------------------------------------------------------------
    // Core dispatch
    // -----------------------------------------------------------------------
    ComplexMatrix runCompute(const PushConstants& pc,
        const ComplexMatrix& A,
        const ComplexMatrix& B,
        uint32_t resultRows,
        uint32_t resultCols);

    // -----------------------------------------------------------------------
    // Buffer helpers
    // -----------------------------------------------------------------------
    Buffer   createBuffer(VkDeviceSize          size,
        VkBufferUsageFlags     usage,
        VkMemoryPropertyFlags  properties);
    void     destroyBuffer(Buffer& buf);
    uint32_t findMemoryType(uint32_t typeBits, VkMemoryPropertyFlags properties);

    // -----------------------------------------------------------------------
    // Command-buffer helpers (submit-and-forget pattern)
    // -----------------------------------------------------------------------
    VkCommandBuffer beginCommands();
    void            submitAndWait(VkCommandBuffer cmd);

    // -----------------------------------------------------------------------
    // Misc
    // -----------------------------------------------------------------------
    static std::vector<uint32_t> loadSpirv(const std::string& path);
    static void                  vkCheck(VkResult result, const char* context);
};


#endif // !VK_COMP

