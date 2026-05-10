#include "VulkanCompute.h"
#include <algorithm>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>

// =============================================================================
// Static helpers
// =============================================================================

void VulkanCompute::vkCheck(VkResult result, const char* context) {
    if (result != VK_SUCCESS)
        throw std::runtime_error(std::string(context) +
            " failed (VkResult=" + std::to_string(result) + ")");
}

std::vector<uint32_t> VulkanCompute::loadSpirv(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f.is_open())
        throw std::runtime_error("Cannot open SPIR-V file: " + path);

    const size_t size = static_cast<size_t>(f.tellg());
    if (size == 0 || size % 4 != 0)
        throw std::runtime_error("SPIR-V file has invalid size: " + path);

    std::vector<uint32_t> code(size / 4);
    f.seekg(0);
    f.read(reinterpret_cast<char*>(code.data()), static_cast<std::streamsize>(size));
    return code;
}

// =============================================================================
// Construction / destruction
// =============================================================================

VulkanCompute::VulkanCompute(const std::string& spirvPath)
    : m_spirvPath(spirvPath)
{
    createInstance();
}

VulkanCompute::~VulkanCompute() {
    destroyDevice();
    if (m_instance != VK_NULL_HANDLE)
        vkDestroyInstance(m_instance, nullptr);
}

// =============================================================================
// Instance creation + GPU enumeration
// =============================================================================

void VulkanCompute::createInstance() {
    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "VulkanCompute";
    appInfo.apiVersion = VK_API_VERSION_1_1;

    VkInstanceCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ci.pApplicationInfo = &appInfo;

    vkCheck(vkCreateInstance(&ci, nullptr, &m_instance), "vkCreateInstance");

    uint32_t count = 0;
    vkEnumeratePhysicalDevices(m_instance, &count, nullptr);
    if (count == 0)
        throw std::runtime_error("No Vulkan-capable devices found on this system");

    m_physicalDevices.resize(count);
    vkEnumeratePhysicalDevices(m_instance, &count, m_physicalDevices.data());
}

std::vector<std::string> VulkanCompute::enumerateGPUs() const {
    std::vector<std::string> names;
    names.reserve(m_physicalDevices.size());
    for (VkPhysicalDevice pd : m_physicalDevices) {
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(pd, &props);
        names.emplace_back(props.deviceName);
    }
    return names;
}

// =============================================================================
// GPU selection — tears down any previous device, then rebuilds everything
// =============================================================================

void VulkanCompute::selectGPU(uint32_t index) {
    if (index >= m_physicalDevices.size())
        throw std::out_of_range("GPU index " + std::to_string(index) +
            " is out of range (found " +
            std::to_string(m_physicalDevices.size()) + " device(s))");

    if (m_selectedGPUIndex == index)
        return; // already configured for this GPU

    destroyDevice();

    createDevice(index);
    createCommandPool();
    createDescriptorSetLayout();
    createDescriptorPool();
    createPipelineLayout();
    createPipeline();

    m_selectedGPUIndex = index;
}

// =============================================================================
// Logical device
// =============================================================================

void VulkanCompute::createDevice(uint32_t gpuIndex) {
    m_physicalDevice = m_physicalDevices[gpuIndex];

    // Derive the largest square workgroup that fits within device limits.
    // e.g. maxComputeWorkGroupInvocations=1024 → localSize=32 (32×32=1024)
    //      maxComputeWorkGroupInvocations=512  → localSize=16
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(m_physicalDevice, &props);
    const uint32_t maxInv = props.limits.maxComputeWorkGroupInvocations;
    m_localSize = 1;
    while ((m_localSize * 2u) * (m_localSize * 2u) <= maxInv)
        m_localSize *= 2u;

    // Find the first queue family that supports compute.
    uint32_t qfCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(m_physicalDevice, &qfCount, nullptr);
    std::vector<VkQueueFamilyProperties> qfs(qfCount);
    vkGetPhysicalDeviceQueueFamilyProperties(m_physicalDevice, &qfCount, qfs.data());

    m_computeQueueFamily = UINT32_MAX;
    for (uint32_t i = 0; i < qfCount; ++i) {
        if (qfs[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            m_computeQueueFamily = i;
            break;
        }
    }
    if (m_computeQueueFamily == UINT32_MAX)
        throw std::runtime_error("Selected GPU has no compute queue family");

    const float priority = 1.f;
    VkDeviceQueueCreateInfo qci{};
    qci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    qci.queueFamilyIndex = m_computeQueueFamily;
    qci.queueCount = 1;
    qci.pQueuePriorities = &priority;

    VkDeviceCreateInfo dci{};
    dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;

    vkCheck(vkCreateDevice(m_physicalDevice, &dci, nullptr, &m_device), "vkCreateDevice");
    vkGetDeviceQueue(m_device, m_computeQueueFamily, 0, &m_computeQueue);
}

void VulkanCompute::destroyDevice() {
    if (m_device == VK_NULL_HANDLE)
        return;

    vkDeviceWaitIdle(m_device);

    if (m_pipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(m_device, m_pipeline, nullptr);
        m_pipeline = VK_NULL_HANDLE;
    }
    if (m_pipelineLayout != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(m_device, m_pipelineLayout, nullptr);
        m_pipelineLayout = VK_NULL_HANDLE;
    }
    if (m_shaderModule != VK_NULL_HANDLE) {
        vkDestroyShaderModule(m_device, m_shaderModule, nullptr);
        m_shaderModule = VK_NULL_HANDLE;
    }
    // Destroying the pool implicitly frees all descriptor sets allocated from it.
    if (m_descPool != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(m_device, m_descPool, nullptr);
        m_descPool = VK_NULL_HANDLE;
        m_descSet = VK_NULL_HANDLE;
    }
    if (m_descSetLayout != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(m_device, m_descSetLayout, nullptr);
        m_descSetLayout = VK_NULL_HANDLE;
    }
    if (m_commandPool != VK_NULL_HANDLE) {
        vkDestroyCommandPool(m_device, m_commandPool, nullptr);
        m_commandPool = VK_NULL_HANDLE;
    }

    vkDestroyDevice(m_device, nullptr);
    m_device = VK_NULL_HANDLE;
    m_physicalDevice = VK_NULL_HANDLE;
    m_computeQueueFamily = UINT32_MAX;
    m_selectedGPUIndex = UINT32_MAX;
}

// =============================================================================
// Command pool
// =============================================================================

void VulkanCompute::createCommandPool() {
    VkCommandPoolCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    ci.queueFamilyIndex = m_computeQueueFamily;
    // RESET_COMMAND_BUFFER_BIT lets us reuse the pool across multiple dispatches.
    ci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    vkCheck(vkCreateCommandPool(m_device, &ci, nullptr, &m_commandPool),
        "vkCreateCommandPool");
}

// =============================================================================
// Descriptor set layout — 3 storage buffers at bindings 0, 1, 2
// =============================================================================

void VulkanCompute::createDescriptorSetLayout() {
    VkDescriptorSetLayoutBinding bindings[3]{};
    for (uint32_t i = 0; i < 3; ++i) {
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }

    VkDescriptorSetLayoutCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    ci.bindingCount = 3;
    ci.pBindings = bindings;

    vkCheck(vkCreateDescriptorSetLayout(m_device, &ci, nullptr, &m_descSetLayout),
        "vkCreateDescriptorSetLayout");
}

// =============================================================================
// Descriptor pool — one persistent set that is updated per dispatch
// =============================================================================

void VulkanCompute::createDescriptorPool() {
    VkDescriptorPoolSize poolSize{};
    poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSize.descriptorCount = 3; // A, B, C

    VkDescriptorPoolCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    ci.maxSets = 1;
    ci.poolSizeCount = 1;
    ci.pPoolSizes = &poolSize;

    vkCheck(vkCreateDescriptorPool(m_device, &ci, nullptr, &m_descPool),
        "vkCreateDescriptorPool");

    VkDescriptorSetAllocateInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    ai.descriptorPool = m_descPool;
    ai.descriptorSetCount = 1;
    ai.pSetLayouts = &m_descSetLayout;

    vkCheck(vkAllocateDescriptorSets(m_device, &ai, &m_descSet),
        "vkAllocateDescriptorSets");
}

// =============================================================================
// Pipeline layout — one descriptor set + push-constant block
// =============================================================================

void VulkanCompute::createPipelineLayout() {
    VkPushConstantRange pcRange{};
    pcRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pcRange.offset = 0;
    pcRange.size = sizeof(PushConstants);

    VkPipelineLayoutCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    ci.setLayoutCount = 1;
    ci.pSetLayouts = &m_descSetLayout;
    ci.pushConstantRangeCount = 1;
    ci.pPushConstantRanges = &pcRange;

    vkCheck(vkCreatePipelineLayout(m_device, &ci, nullptr, &m_pipelineLayout),
        "vkCreatePipelineLayout");
}

// =============================================================================
// Compute pipeline
//
// The shader must declare its workgroup size via specialization constants:
//   layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;
//
// This lets us specialize to the optimal tile size for the selected GPU
// at pipeline-creation time without recompiling the GLSL source.
// =============================================================================

void VulkanCompute::createPipeline() {
    const auto code = loadSpirv(m_spirvPath);

    VkShaderModuleCreateInfo smci{};
    smci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    smci.codeSize = code.size() * sizeof(uint32_t);
    smci.pCode = code.data();

    vkCheck(vkCreateShaderModule(m_device, &smci, nullptr, &m_shaderModule),
        "vkCreateShaderModule");

    // Specialization constants: local_size_x (id=0), local_size_y (id=1), local_size_z (id=2)
    const uint32_t specData[3] = { m_localSize, m_localSize, 1u };

    VkSpecializationMapEntry specEntries[3];
    for (uint32_t i = 0; i < 3; ++i) {
        specEntries[i].constantID = i;
        specEntries[i].offset = i * static_cast<uint32_t>(sizeof(uint32_t));
        specEntries[i].size = sizeof(uint32_t);
    }

    VkSpecializationInfo specInfo{};
    specInfo.mapEntryCount = 3;
    specInfo.pMapEntries = specEntries;
    specInfo.dataSize = sizeof(specData);
    specInfo.pData = specData;

    VkPipelineShaderStageCreateInfo stageCI{};
    stageCI.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stageCI.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stageCI.module = m_shaderModule;
    stageCI.pName = "main";
    stageCI.pSpecializationInfo = &specInfo;

    VkComputePipelineCreateInfo pipelineCI{};
    pipelineCI.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineCI.stage = stageCI;
    pipelineCI.layout = m_pipelineLayout;

    vkCheck(vkCreateComputePipelines(m_device, VK_NULL_HANDLE, 1, &pipelineCI,
        nullptr, &m_pipeline),
        "vkCreateComputePipelines");
}

// =============================================================================
// Public compute methods
// =============================================================================

ComplexMatrix VulkanCompute::multiply(const ComplexMatrix& A, const ComplexMatrix& B) {
    if (m_device == VK_NULL_HANDLE)
        throw std::runtime_error("No GPU selected — call selectGPU() first");
    if (A.cols != B.rows)
        throw std::invalid_argument(
            "multiply: A.cols (" + std::to_string(A.cols) +
            ") must equal B.rows (" + std::to_string(B.rows) + ")");

    PushConstants pc{};
    pc.operation = 1;
    pc.linesA = static_cast<int32_t>(A.rows);
    pc.columnsA = static_cast<int32_t>(A.cols);
    pc.linesB = static_cast<int32_t>(B.rows);
    pc.columnsB = static_cast<int32_t>(B.cols);

    return runCompute(pc, A, B, A.rows, B.cols);
}

ComplexMatrix VulkanCompute::kronecker(const ComplexMatrix& A, const ComplexMatrix& B) {
    if (m_device == VK_NULL_HANDLE)
        throw std::runtime_error("No GPU selected — call selectGPU() first");

    PushConstants pc{};
    pc.operation = 0;
    pc.linesA = static_cast<int32_t>(A.rows);
    pc.columnsA = static_cast<int32_t>(A.cols);
    pc.linesB = static_cast<int32_t>(B.rows);
    pc.columnsB = static_cast<int32_t>(B.cols);

    return runCompute(pc, A, B,
        A.rows * B.rows,
        A.cols * B.cols);
}

ComplexMatrix VulkanCompute::add(const ComplexMatrix& A, const ComplexMatrix& B) {
    if (m_device == VK_NULL_HANDLE)
        throw std::runtime_error("No GPU selected — call selectGPU() first");
    if (A.rows != B.rows || A.cols != B.cols)
        throw std::invalid_argument(
            "add: A and B must have the same dimensions");

    PushConstants pc{};
    pc.operation = 2;
    pc.linesA = static_cast<int32_t>(A.rows);
    pc.columnsA = static_cast<int32_t>(A.cols);
    pc.linesB = static_cast<int32_t>(B.rows);
    pc.columnsB = static_cast<int32_t>(B.cols);

    return runCompute(pc, A, B, A.rows, A.cols);
}

ComplexMatrix VulkanCompute::multiplyByScalar(const std::complex<float>& s, const ComplexMatrix& A) {
    if (m_device == VK_NULL_HANDLE)
        throw std::runtime_error("No GPU selected — call selectGPU() first");

    PushConstants pc{};
    pc.operation = 3;
    pc.linesA = static_cast<int32_t>(A.rows);
    pc.columnsA = static_cast<int32_t>(A.cols);
    pc.linesB = 0; // unused
    pc.columnsB = 0; // unused

    // We can reuse the B buffer to pass the scalar, since the shader only reads B for multiply and ignores it for scalar multiplication.
    ComplexMatrix scalarMatrix(1, 1);
    scalarMatrix.at(0, 0) = s;
    return runCompute(pc, A, scalarMatrix, A.rows, A.cols);
}

// =============================================================================
// Core dispatch
// =============================================================================

ComplexMatrix VulkanCompute::runCompute(const PushConstants& pc,
    const ComplexMatrix& A,
    const ComplexMatrix& B,
    uint32_t resultRows,
    uint32_t resultCols)
{
    const VkDeviceSize sizeA = A.byteSize();
    const VkDeviceSize sizeB = B.byteSize();
    const VkDeviceSize sizeC = resultRows * resultCols * sizeof(std::complex<float>);

    // Host-visible + coherent memory gives us direct CPU↔GPU access without
    // explicit flushes.  For large, repeated workloads you would want
    // device-local buffers with a separate staging buffer, but for a compute
    // utility this is correct and simple.
    constexpr VkMemoryPropertyFlags kHostProps =
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    constexpr VkBufferUsageFlags kStorageUsage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;

    Buffer bufA = createBuffer(sizeA, kStorageUsage, kHostProps);
    Buffer bufB = createBuffer(sizeB, kStorageUsage, kHostProps);
    Buffer bufC = createBuffer(sizeC, kStorageUsage, kHostProps);

    // --- Upload A and B ---
    auto upload = [&](Buffer& buf, const void* src, VkDeviceSize size) {
        void* mapped = nullptr;
        vkMapMemory(m_device, buf.memory, 0, size, 0, &mapped);
        std::memcpy(mapped, src, static_cast<size_t>(size));
        vkUnmapMemory(m_device, buf.memory);
        };
    upload(bufA, A.data.data(), sizeA);
    upload(bufB, B.data.data(), sizeB);

    // --- Point the persistent descriptor set at the new buffers ---
    VkDescriptorBufferInfo dbiA{ bufA.handle, 0, sizeA };
    VkDescriptorBufferInfo dbiB{ bufB.handle, 0, sizeB };
    VkDescriptorBufferInfo dbiC{ bufC.handle, 0, sizeC };

    const VkDescriptorBufferInfo* dbiPtrs[3] = { &dbiA, &dbiB, &dbiC };

    VkWriteDescriptorSet writes[3]{};
    for (uint32_t i = 0; i < 3; ++i) {
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = m_descSet;
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = dbiPtrs[i];
    }
    vkUpdateDescriptorSets(m_device, 3, writes, 0, nullptr);

    // --- Record command buffer ---
    VkCommandBuffer cmd = beginCommands();

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipeline);

    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
        m_pipelineLayout, 0, 1, &m_descSet, 0, nullptr);

    vkCmdPushConstants(cmd, m_pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT,
        0, sizeof(PushConstants), &pc);

    // Compute dispatch dimensions:
    //   Kronecker: the shader iterates over A — one thread per A element.
    //   Multiply:  one thread per output element of C.
    // gl_GlobalInvocationID.x → i (rows), .y → j (columns).
    uint32_t groupsX, groupsY;
    if (pc.operation == 0) {
        // Kronecker: cover A's dimensions
        groupsX = (static_cast<uint32_t>(pc.linesA) + m_localSize - 1) / m_localSize;
        groupsY = (static_cast<uint32_t>(pc.columnsA) + m_localSize - 1) / m_localSize;
    }
    else {
        // Multiply: cover C's dimensions (resultRows × resultCols)
        groupsX = (resultRows + m_localSize - 1) / m_localSize;
        groupsY = (resultCols + m_localSize - 1) / m_localSize;
    }
    vkCmdDispatch(cmd, groupsX, groupsY, 1);

    submitAndWait(cmd); // blocks until the GPU is done

    // --- Download result ---
    ComplexMatrix C(resultRows, resultCols);
    void* mapped = nullptr;
    vkMapMemory(m_device, bufC.memory, 0, sizeC, 0, &mapped);
    std::memcpy(C.data.data(), mapped, static_cast<size_t>(sizeC));
    vkUnmapMemory(m_device, bufC.memory);

    destroyBuffer(bufA);
    destroyBuffer(bufB);
    destroyBuffer(bufC);

    return C;
}

// =============================================================================
// Buffer helpers
// =============================================================================

VulkanCompute::Buffer VulkanCompute::createBuffer(VkDeviceSize          size,
    VkBufferUsageFlags    usage,
    VkMemoryPropertyFlags properties)
{
    Buffer buf;
    buf.size = size;

    VkBufferCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    ci.size = size;
    ci.usage = usage;
    ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    vkCheck(vkCreateBuffer(m_device, &ci, nullptr, &buf.handle), "vkCreateBuffer");

    VkMemoryRequirements memReqs;
    vkGetBufferMemoryRequirements(m_device, buf.handle, &memReqs);

    VkMemoryAllocateInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    ai.allocationSize = memReqs.size;
    ai.memoryTypeIndex = findMemoryType(memReqs.memoryTypeBits, properties);

    vkCheck(vkAllocateMemory(m_device, &ai, nullptr, &buf.memory), "vkAllocateMemory");
    vkCheck(vkBindBufferMemory(m_device, buf.handle, buf.memory, 0),
        "vkBindBufferMemory");

    return buf;
}

void VulkanCompute::destroyBuffer(Buffer& buf) {
    if (buf.handle != VK_NULL_HANDLE) {
        vkDestroyBuffer(m_device, buf.handle, nullptr);
        buf.handle = VK_NULL_HANDLE;
    }
    if (buf.memory != VK_NULL_HANDLE) {
        vkFreeMemory(m_device, buf.memory, nullptr);
        buf.memory = VK_NULL_HANDLE;
    }
}

uint32_t VulkanCompute::findMemoryType(uint32_t typeBits, VkMemoryPropertyFlags props) {
    VkPhysicalDeviceMemoryProperties memProps;
    vkGetPhysicalDeviceMemoryProperties(m_physicalDevice, &memProps);

    for (uint32_t i = 0; i < memProps.memoryTypeCount; ++i) {
        if ((typeBits & (1u << i)) &&
            (memProps.memoryTypes[i].propertyFlags & props) == props)
            return i;
    }
    throw std::runtime_error("findMemoryType: no suitable memory type found");
}

// =============================================================================
// Command-buffer helpers
// =============================================================================

VkCommandBuffer VulkanCompute::beginCommands() {
    VkCommandBufferAllocateInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    ai.commandPool = m_commandPool;
    ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;

    VkCommandBuffer cmd = VK_NULL_HANDLE;
    vkCheck(vkAllocateCommandBuffers(m_device, &ai, &cmd), "vkAllocateCommandBuffers");

    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    vkCheck(vkBeginCommandBuffer(cmd, &bi), "vkBeginCommandBuffer");
    return cmd;
}

void VulkanCompute::submitAndWait(VkCommandBuffer cmd) {
    vkCheck(vkEndCommandBuffer(cmd), "vkEndCommandBuffer");

    // Use a fence so we can block on CPU until the GPU finishes.
    VkFenceCreateInfo fci{};
    fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;

    VkFence fence = VK_NULL_HANDLE;
    vkCheck(vkCreateFence(m_device, &fci, nullptr, &fence), "vkCreateFence");

    VkSubmitInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd;

    vkCheck(vkQueueSubmit(m_computeQueue, 1, &si, fence), "vkQueueSubmit");
    vkCheck(vkWaitForFences(m_device, 1, &fence, VK_TRUE, UINT64_MAX),
        "vkWaitForFences");

    vkDestroyFence(m_device, fence, nullptr);
    vkFreeCommandBuffers(m_device, m_commandPool, 1, &cmd);
}
