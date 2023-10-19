#include <vulkan/vulkan.h>

#include <iostream>
#include <vector>
#include <cstring>
#include <cassert>
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <fstream>
#include <sstream>

#define EXIT_SUCCESS 0
#define EXIT_FAILURE 1

// Used for validating return values of Vulkan API calls.
#define VK_CHECK_RESULT(f) 																				\
{																										\
    VkResult res = (f);																					\
    if (res != VK_SUCCESS)																				\
    {																									\
        printf("Fatal : VkResult is %d in %s at line %d\n", res,  __FILE__, __LINE__); \
        assert(res == VK_SUCCESS);																		\
    }																									\
}

VkInstance createInstance(const std::vector<const char*>& enabledLayers, const std::vector<const char*>& enabledExtensions) {
    VkApplicationInfo applicationInfo = {};
    applicationInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    applicationInfo.pApplicationName = "vkQuantum";
    applicationInfo.applicationVersion = 1;
    applicationInfo.pEngineName = nullptr;
    applicationInfo.engineVersion = 0;
    applicationInfo.apiVersion = VK_API_VERSION_1_0;

    VkInstanceCreateInfo createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.flags = 0;
    createInfo.pApplicationInfo = &applicationInfo;

    // Give our desired layers and extensions to vulkan.
    createInfo.enabledLayerCount = (uint32_t)enabledLayers.size();
    createInfo.ppEnabledLayerNames = enabledLayers.data();
    createInfo.enabledExtensionCount = (uint32_t)enabledExtensions.size();
    createInfo.ppEnabledExtensionNames = enabledExtensions.data();

    VkInstance instance;
    VK_CHECK_RESULT(vkCreateInstance(&createInfo, NULL, &instance));
    return instance;
}

VkPhysicalDevice findPhysicalDevice(VkInstance instance) {
    uint32_t deviceCount;
    vkEnumeratePhysicalDevices(instance, &deviceCount, NULL);
    if (deviceCount == 0)
        throw std::runtime_error("could not find a device with vulkan support");

    std::vector<VkPhysicalDevice> devices(deviceCount);
    vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

    std::string type[] = {
        "Other", "Integrated GPU", "Discrete GPU", "Virtual GPU", "CPU"
    };

    for (auto device : devices) {
        VkPhysicalDeviceProperties deviceProperties;
        vkGetPhysicalDeviceProperties(device, &deviceProperties);
        std::cout << "device ID: " << deviceProperties.deviceID << '\n';
        std::cout << "device name: " << deviceProperties.deviceName << '\n';
        std::cout << "device type: " << type[deviceProperties.deviceType] << '\n';

        /*
        maxComputeWorkGroupCount[3] is the maximum number of local workgroups that can be dispatched by a single dispatching command.
        These three values represent the maximum number of local workgroups for the X, Y, and Z dimensions, respectively.
        The workgroup count parameters to the dispatching commands must be less than or equal to the corresponding limit.
        See https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#dispatch.
        */
        uint32_t* count = deviceProperties.limits.maxComputeWorkGroupCount;
        std::cout << "max compute workgroup count: " << count[0] << ' ' << count[1] << ' ' << count[2] << '\n';

        /*
        maxComputeWorkGroupSize[3] is the maximum size of a local compute workgroup, per dimension.
        These three values represent the maximum local workgroup size in the X, Y, and Z dimensions, respectively.
        The x, y, and z sizes, as specified by the LocalSize or LocalSizeId execution mode or by the object decorated by the
        WorkgroupSize decoration in shader modules, must be less than or equal to the corresponding limit.
        */
        uint32_t* maxSize = deviceProperties.limits.maxComputeWorkGroupSize;
        std::cout << "max compute workgroup size: " << maxSize[0] << ' ' << maxSize[1] << ' ' << maxSize[2] << '\n';

        /*
        maxComputeWorkGroupInvocations is the maximum total number of compute shader invocations in a single local workgroup.
        The product of the X, Y, and Z sizes, as specified by the LocalSize or LocalSizeId execution mode in shader modules or
        by the object decorated by the WorkgroupSize decoration, must be less than or equal to this limit.
        */
        std::cout << "max compute workgroup invocations: " << deviceProperties.limits.maxComputeWorkGroupInvocations << '\n';

        /*
        maxComputeSharedMemorySize is the maximum total storage size, in bytes, available for variables declared with the Workgroup
        storage class in shader modules (or with the shared storage qualifier in GLSL) in the compute shader stage.
        */
        std::cout << "max compute shared memory size: " << deviceProperties.limits.maxComputeSharedMemorySize << " bytes\n";

        /*
        maxPushConstantsSize is the maximum size, in bytes, of the pool of push constant memory.
        For each of the push constant ranges indicated by the pPushConstantRanges member of the VkPipelineLayoutCreateInfo
        structure, (offset + size) must be less than or equal to this limit.
        */
        std::cout << "max pushConstants size: " << deviceProperties.limits.maxPushConstantsSize << " bytes\n";


        VkPhysicalDeviceFeatures deviceFeatures;
        vkGetPhysicalDeviceFeatures(device, &deviceFeatures);
        std::cout << "geometry shader: " << (deviceFeatures.geometryShader ? "yes" : "no") << '\n';
        std::cout << "tesselation shader: " << (deviceFeatures.tessellationShader ? "yes" : "no") << "\n\n";
    }

    return devices.back();
    //return devices[1];
}

uint32_t getComputeQueueFamilyIndex(VkPhysicalDevice physicalDevice) {
    uint32_t queueFamilyCount;
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, NULL);

    // Retrieve all queue families.
    std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, queueFamilies.data());

    std::cout << "queue families:\n";
    for (const VkQueueFamilyProperties& props : queueFamilies) {
        std::cout << "queueCount " << props.queueCount << '\n';
        std::cout << "flags " << (props.queueFlags) << '\n';
        std::cout << "graphics " << ((props.queueFlags & VK_QUEUE_GRAPHICS_BIT) != 0) << '\n';
        std::cout << "compute " << ((props.queueFlags & VK_QUEUE_COMPUTE_BIT) != 0) << "\n";
        std::cout << "transfer " << ((props.queueFlags & VK_QUEUE_TRANSFER_BIT) != 0) << "\n";
        std::cout << "sparse binding " << ((props.queueFlags & VK_QUEUE_SPARSE_BINDING_BIT) != 0) << "\n";
        std::cout << '\n';
    }

    for (uint32_t i = 0; i < queueFamilyCount; ++i) {
        VkQueueFamilyProperties props = queueFamilies[i];
        if (props.queueCount > 0 && (props.queueFlags & VK_QUEUE_COMPUTE_BIT)) {
            // found a queue with compute. We're done!
            return i;
        }
    }
    throw std::runtime_error("could not find a queue family that supports operations");
    return 0;
}

VkDevice createDevice(VkPhysicalDevice physicalDevice, uint32_t queueFamilyIndex, const std::vector<const char*>& enabledLayers) {
    VkDeviceQueueCreateInfo queueCreateInfo = {};
    queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queueCreateInfo.queueFamilyIndex = queueFamilyIndex;
    queueCreateInfo.queueCount = 1;   // create one queue in this family. We don't need more.
    float queuePriorities = 1.0; // we only have one queue, so this is not that imporant. 
    queueCreateInfo.pQueuePriorities = &queuePriorities;

    // Specify any desired device features here. We do not need any for this application, though.
    VkPhysicalDeviceFeatures deviceFeatures = {};

    /*
    Now we create the logical device. The logical device allows us to interact with the physical
    device.
    */
    VkDeviceCreateInfo deviceCreateInfo = {};
    deviceCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    deviceCreateInfo.enabledLayerCount = (uint32_t)enabledLayers.size();  // need to specify validation layers here as well.
    deviceCreateInfo.ppEnabledLayerNames = enabledLayers.data();
    deviceCreateInfo.pQueueCreateInfos = &queueCreateInfo; // when creating the logical device, we also specify what queues it has.
    deviceCreateInfo.queueCreateInfoCount = 1;
    deviceCreateInfo.pEnabledFeatures = &deviceFeatures;

    VkDevice device;
    VK_CHECK_RESULT(vkCreateDevice(physicalDevice, &deviceCreateInfo, NULL, &device)); // create logical device.

    return device;
}

VkQueue getDeviceQueue(VkDevice device, uint32_t queueFamilyIndex) {
    VkQueue queue;
    vkGetDeviceQueue(device, queueFamilyIndex, 0, &queue);
    return queue;
}

VkBuffer createBuffer(VkDevice device, uint32_t bufferSize) {
    VkBufferCreateInfo bufferCreateInfo = {};
    bufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferCreateInfo.size = bufferSize; // buffer size in bytes. 
    bufferCreateInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT; // buffer is used as a storage buffer.
    bufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE; // buffer is exclusive to a single queue family at a time. 

    VkBuffer buffer;
    VK_CHECK_RESULT(vkCreateBuffer(device, &bufferCreateInfo, NULL, &buffer)); // create buffer.
    return buffer;
}


uint32_t findMemoryType(VkPhysicalDevice physicalDevice, uint32_t memoryTypeBits, VkMemoryPropertyFlags properties) {
    VkPhysicalDeviceMemoryProperties memoryProperties;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memoryProperties);

    for (uint32_t i = 0; i < memoryProperties.memoryTypeCount; ++i) {
        if ((memoryTypeBits & (1 << i)) &&
            ((memoryProperties.memoryTypes[i].propertyFlags & properties) == properties))
            return i;
    }
    return -1;
}

VkMemoryAllocateInfo getAllocateInfo(VkDevice device, VkPhysicalDevice physicalDevice, VkBuffer buffer) {
    VkMemoryRequirements memoryRequirements;
    vkGetBufferMemoryRequirements(device, buffer, &memoryRequirements);
    std::cout << "memory requirements: \n";
    std::cout << "size             : " << memoryRequirements.size << " bytes \n";
    std::cout << "alignment        : " << memoryRequirements.alignment << " bytes \n";
    std::cout << "memory type bits : " << memoryRequirements.memoryTypeBits << '\n';

    VkMemoryAllocateInfo allocateInfo = {};
    allocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocateInfo.allocationSize = memoryRequirements.size; // specify required memory.

    allocateInfo.memoryTypeIndex = findMemoryType(physicalDevice,
        memoryRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
    std::cout << "memory type index = " << allocateInfo.memoryTypeIndex << '\n';
    return allocateInfo;
}

VkDeviceMemory createBufferMemory(VkDevice device, VkPhysicalDevice physicalDevice, VkBuffer buffer) {
    VkMemoryAllocateInfo allocateInfo = getAllocateInfo(device, physicalDevice, buffer);

    VkDeviceMemory bufAMemory;
    VK_CHECK_RESULT(vkAllocateMemory(device, &allocateInfo, NULL, &bufAMemory)); // allocate memory on device.

    // Now associate that allocated memory with the buffer. With that, the buffer is backed by actual memory. 
    VK_CHECK_RESULT(vkBindBufferMemory(device, buffer, bufAMemory, 0));

    return bufAMemory;
}

VkDescriptorSetLayout createDescriptorSetLayout(VkDevice device) {
    VkDescriptorSetLayoutBinding descriptorSetLayoutBinding[3] = {};
    for (int i = 0; i < 3; i++) {
        descriptorSetLayoutBinding[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        descriptorSetLayoutBinding[i].binding = i;
        descriptorSetLayoutBinding[i].descriptorCount = 1;
        descriptorSetLayoutBinding[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }

    VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo = {};
    descriptorSetLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    descriptorSetLayoutCreateInfo.bindingCount = 3;
    descriptorSetLayoutCreateInfo.pBindings = descriptorSetLayoutBinding;

    // Create the descriptor set layout. 
    VkDescriptorSetLayout descriptorSetLayout;
    VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device, &descriptorSetLayoutCreateInfo, NULL, &descriptorSetLayout));
    return descriptorSetLayout;
}

VkDescriptorPool createDescriptorPool(VkDevice device) {
    VkDescriptorPoolSize descriptorPoolSize = {};
    descriptorPoolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptorPoolSize.descriptorCount = 3; // 

    VkDescriptorPoolCreateInfo descriptorPoolCreateInfo = {};
    descriptorPoolCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    descriptorPoolCreateInfo.maxSets = 1; // we only need to allocate one descriptor set from the pool.
    descriptorPoolCreateInfo.poolSizeCount = 1;
    descriptorPoolCreateInfo.pPoolSizes = &descriptorPoolSize;

    // create descriptor pool.
    VkDescriptorPool descriptorPool;
    VK_CHECK_RESULT(vkCreateDescriptorPool(device, &descriptorPoolCreateInfo, NULL, &descriptorPool));
    return descriptorPool;
}

VkDescriptorSet createDescriptorSet(VkDevice device, VkDescriptorPool descriptorPool, VkDescriptorSetLayout& descriptorSetLayout) {
    /*
    With the pool allocated, we can now allocate the descriptor set.
    */
    VkDescriptorSetAllocateInfo descriptorSetAllocateInfo = {};
    descriptorSetAllocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    descriptorSetAllocateInfo.descriptorPool = descriptorPool; // pool to allocate from.
    descriptorSetAllocateInfo.descriptorSetCount = 1; // allocate a single descriptor set.
    descriptorSetAllocateInfo.pSetLayouts = &descriptorSetLayout;

    // allocate descriptor set.
    VkDescriptorSet descriptorSet;
    VK_CHECK_RESULT(vkAllocateDescriptorSets(device, &descriptorSetAllocateInfo, &descriptorSet));
    return descriptorSet;
}

std::string loadSource(std::string file) {
    std::ifstream in(file, std::ios::binary);
    std::stringstream buffer;
    buffer << in.rdbuf();
    std::string s = buffer.str();
    size_t filesizepadded = size_t(ceil(s.size() / 4.0)) * 4;
    s.resize(filesizepadded);
    return s;
}

VkShaderModule createShaderModule(VkDevice device, std::string scode) {
    VkShaderModuleCreateInfo createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.pCode = (const uint32_t*)scode.c_str();
    createInfo.codeSize = scode.size();
    
    VkShaderModule computeShaderModule;
    VK_CHECK_RESULT(vkCreateShaderModule(device, &createInfo, NULL, &computeShaderModule));
    return computeShaderModule;
}

VkPipelineLayout createPipelineLayout(VkDevice device, VkDescriptorSetLayout& descriptorSetLayout, uint32_t push_constant_size) {
    VkPushConstantRange push_constant_info;
    push_constant_info.offset = 0;
    push_constant_info.size = push_constant_size;
    push_constant_info.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = {};
    pipelineLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutCreateInfo.setLayoutCount = 1;
    pipelineLayoutCreateInfo.pSetLayouts = &descriptorSetLayout;
    pipelineLayoutCreateInfo.pPushConstantRanges = &push_constant_info;
    pipelineLayoutCreateInfo.pushConstantRangeCount = 1;

    VkPipelineLayout pipelineLayout;
    VK_CHECK_RESULT(vkCreatePipelineLayout(device, &pipelineLayoutCreateInfo, NULL, &pipelineLayout));
    return pipelineLayout;
}

VkPipeline createComputePipeline(VkDevice device, VkShaderModule computeShaderModule, VkPipelineLayout pipelineLayout) {
    VkPipelineShaderStageCreateInfo shaderStageCreateInfo = {};
    shaderStageCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    shaderStageCreateInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    shaderStageCreateInfo.module = computeShaderModule;
    shaderStageCreateInfo.pName = "main";

    VkComputePipelineCreateInfo pipelineCreateInfo = {};
    pipelineCreateInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineCreateInfo.stage = shaderStageCreateInfo;
    pipelineCreateInfo.layout = pipelineLayout;

    VkPipeline pipeline;
    VK_CHECK_RESULT(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineCreateInfo, nullptr, &pipeline));
    return pipeline;
}

VkCommandPool createCommandPool(VkDevice device, uint32_t queueFamilyIndex) {
    /*
    We are getting closer to the end. In order to send commands to the device(GPU),
    we must first record commands into a command buffer.
    To allocate a command buffer, we must first create a command pool. So let us do that.
    */
    VkCommandPoolCreateInfo commandPoolCreateInfo = {};
    commandPoolCreateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    commandPoolCreateInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    // the queue family of this command pool. All command buffers allocated from this command pool,
    // must be submitted to queues of this family ONLY. 
    commandPoolCreateInfo.queueFamilyIndex = queueFamilyIndex;

    VkCommandPool commandPool;
    VK_CHECK_RESULT(vkCreateCommandPool(device, &commandPoolCreateInfo, NULL, &commandPool));
    return commandPool;
}

VkCommandBuffer createCommandBuffer(VkDevice device, VkCommandPool commandPool) {
    /*
    Now allocate a command buffer from the command pool.
    */
    VkCommandBufferAllocateInfo commandBufferAllocateInfo = {};
    commandBufferAllocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    commandBufferAllocateInfo.commandPool = commandPool; // specify the command pool to allocate from. 
    // if the command buffer is primary, it can be directly submitted to queues. 
    // A secondary buffer has to be called from some primary command buffer, and cannot be directly 
    // submitted to a queue. To keep things simple, we use a primary command buffer. 
    commandBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    commandBufferAllocateInfo.commandBufferCount = 1; // allocate a single command buffer. 

    VkCommandBuffer commandBuffer;
    VK_CHECK_RESULT(vkAllocateCommandBuffers(device, &commandBufferAllocateInfo, &commandBuffer)); // allocate command buffer.
    return commandBuffer;
}

void runCommandBuffer(VkDevice device, VkCommandBuffer commandBuffer, VkQueue queue) {
    VkFenceCreateInfo fenceCreateInfo = {};
    fenceCreateInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceCreateInfo.flags = 0;
    VkFence fence;
    VK_CHECK_RESULT(vkCreateFence(device, &fenceCreateInfo, NULL, &fence));

    VkSubmitInfo submitInfo = {};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1; // submit a single command buffer
    submitInfo.pCommandBuffers = &commandBuffer; // the command buffer to submit.
    VK_CHECK_RESULT(vkQueueSubmit(queue, 1, &submitInfo, fence));

    //VK_CHECK_RESULT(vkWaitForFences(device, 1, &fence, VK_TRUE, 100000000000));
    if (vkWaitForFences(device, 1, &fence, VK_TRUE, 100000000000) == VK_ERROR_DEVICE_LOST) {
        std::cout << "DEVICE LOST\n";
        
    }

    vkDestroyFence(device, fence, NULL);
}

struct DeviceInfo {
    VkInstance                instance;
    VkPhysicalDevice          physicalDevice;
    uint32_t                  queueFamilyIndex;
    VkDevice                  device;

    DeviceInfo(const std::vector<const char*>& enabledLayers, const std::vector<const char*>& enabledExtensions) {
        instance = createInstance(enabledLayers, enabledExtensions);
        physicalDevice = findPhysicalDevice(instance);
        queueFamilyIndex = getComputeQueueFamilyIndex(physicalDevice);
        device = createDevice(physicalDevice, queueFamilyIndex, enabledLayers);
    }

    void clear() {
        vkDestroyDevice(device, NULL);
        vkDestroyInstance(instance, NULL);
    }
};

struct Buffer {
    VkDevice                  device;
    uint32_t                  bufferSize;
    VkBuffer                  buf;
    VkDeviceMemory            bufMemory;

    Buffer(DeviceInfo device_info, uint32_t size) {
        device = device_info.device;
        bufferSize = size;
        buf = createBuffer(device, bufferSize);
        bufMemory = createBufferMemory(device, device_info.physicalDevice, buf);
    }

    void clear() {
        vkFreeMemory(device, bufMemory, NULL);
        vkDestroyBuffer(device, buf, NULL);
    }

    // Map the buffer memory, so that we can read from it on the CPU.
    template<class T>
    T* map() {
        T* pmappedMemory;
        vkMapMemory(device, bufMemory, 0, bufferSize, 0, (void**)&pmappedMemory);
        return pmappedMemory;
    }

    void unmap() {
        vkUnmapMemory(device, bufMemory);
    }
};

struct vec2 {
    float x, y;
};

// Implementar a leitura de 1 qubit

struct Buffer2D : Buffer {
    uint32_t width, height;

    Buffer2D(DeviceInfo device_info, uint32_t w, uint32_t h) :
        Buffer{ device_info, (uint32_t)sizeof(float) * w * h }, //2*sizeof(float) para complexo
        width{ w }, height{ h } {}

    void print(const char* label) {
        float* p = map<float>();

        printf("%s =\n", label);
        for (uint32_t y = 0; y < height; y++) {
            for (uint32_t x = 0; x < width; x++, p++)
                printf("%f\t", *p);
            puts("");
        }

        unmap();
    }
    template<class F>
    void init(F f) {
        float* p = map<float>();

        for (uint32_t y = 0; y < height; y++)
            for (uint32_t x = 0; x < width; x++, p++)
                *p = f(x, y);

        unmap();
    }
};

float fA(int x, int y) {
    //return y + (float)0.01 * x;
    return (float)((x) + (y));
}

float fB(int x, int y) {
    //return x + (float)0.01 * y;
    return (float)((x) + (y));
}

struct ComputeShader {
    VkDevice                  device;
    VkDescriptorPool          descriptorPool;
    VkDescriptorSetLayout     descriptorSetLayout;
    VkDescriptorSet           descriptorSet;
    VkShaderModule            computeShaderModule;
    VkPipelineLayout          pipelineLayout;
    VkPipeline                pipeline;
    VkQueue                   queue;
    VkCommandPool             commandPool;
    VkCommandBuffer           commandBuffer;

    struct {
        uint32_t X, Y, Z;
    }const WorkgroupSize = { 32, 32, 1 };

    struct PushConstants {
        int operation;
        int linesA;
        int columnsA;
        int linesB;
        int columnsB;
    }constants = { 0, 0, 0, 0, 0 };

    ComputeShader(DeviceInfo device_info) {
        device = device_info.device;
        descriptorPool = createDescriptorPool(device);
        descriptorSetLayout = createDescriptorSetLayout(device);
        descriptorSet = createDescriptorSet(device, descriptorPool, descriptorSetLayout);

        // the code in comp.spv was created by running the command:
        // glslangValidator.exe -V shader3.comp
        // glslc.exe -O .\shader3.comp -o comp.spv
        computeShaderModule = createShaderModule(device, loadSource("shaders/comp.spv"));

        pipelineLayout = createPipelineLayout(device, descriptorSetLayout, sizeof(PushConstants));
        pipeline = createComputePipeline(device, computeShaderModule, pipelineLayout);
        queue = getDeviceQueue(device, device_info.queueFamilyIndex);
        commandPool = createCommandPool(device, device_info.queueFamilyIndex);
        commandBuffer = createCommandBuffer(device, commandPool);
    }

    void clear() {
        vkDestroyCommandPool(device, commandPool, NULL);
        vkDestroyPipeline(device, pipeline, NULL);
        vkDestroyPipelineLayout(device, pipelineLayout, NULL);
        vkDestroyShaderModule(device, computeShaderModule, NULL);
        vkDestroyDescriptorSetLayout(device, descriptorSetLayout, NULL);
        vkDestroyDescriptorPool(device, descriptorPool, NULL);
    }

    void connect(Buffer buffer, int32_t binding) {
        // Specify the buffer to bind to the descriptor.
        VkDescriptorBufferInfo descriptorBufferInfo = {};
        descriptorBufferInfo.buffer = buffer.buf;
        descriptorBufferInfo.offset = 0;
        descriptorBufferInfo.range = buffer.bufferSize;

        VkWriteDescriptorSet writeDescriptorSet = {};
        writeDescriptorSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writeDescriptorSet.dstSet = descriptorSet; // write to this descriptor set.
        writeDescriptorSet.dstBinding = binding;       // write to the first, and only binding.
        writeDescriptorSet.descriptorCount = 1;             // update a single descriptor.
        writeDescriptorSet.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; // storage buffer.
        writeDescriptorSet.pBufferInfo = &descriptorBufferInfo;

        // perform the update of the descriptor set.
        vkUpdateDescriptorSets(device, 1, &writeDescriptorSet, 0, NULL);
    }

    void run(uint32_t nx = 1, uint32_t ny = 1, uint32_t nz = 1) {
        VkCommandBufferBeginInfo beginInfo = {};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT; // the buffer is only submitted and used once in this application.
        VkResult err = vkBeginCommandBuffer(commandBuffer, &beginInfo);
        VK_CHECK_RESULT(err); // start recording commands.

        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, &descriptorSet, 0, NULL);
        vkCmdPushConstants(commandBuffer, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstants), &constants);
        uint32_t sx = (uint32_t)ceil(nx / float(WorkgroupSize.X));
        uint32_t sy = (uint32_t)ceil(ny / float(WorkgroupSize.Y));
        uint32_t sz = (uint32_t)ceil(nz / float(WorkgroupSize.Z)); //Alterar a quantidade de workgroups 
        std::cout << "Invocations: " << sx << " " << sy << " " << sz << std::endl;
        vkCmdDispatch(commandBuffer, sx, sy, sz);

        VK_CHECK_RESULT(vkEndCommandBuffer(commandBuffer)); // end recording commands.

        runCommandBuffer(device, commandBuffer, queue);
    }
};

#define MATMUL
#ifndef MATMUL
//Example of Kronecker Product
#define COLUMNS 2 //WIDTH -> Columns
#define LINES 3 //HEIGHT -> Lines
#define COLUMNS_B 4
#define LINES_B 5
#define COLUMNS_C COLUMNS*COLUMNS_B
#define LINES_C LINES*LINES_B
#define OPERATION 0
#else
//Example of Matrix Multiplication
#define COLUMNS 4 //COLUMNS -> Columns
#define LINES 3 //HEIGHT -> Lines
#define COLUMNS_B 5
#define LINES_B COLUMNS
#define COLUMNS_C COLUMNS_B
#define LINES_C LINES
#define OPERATION 1
#endif
int main() {
    std::vector<const char*> enabledLayers = {"VK_LAYER_KHRONOS_validation"};
    std::vector<const char*> enabledExtensions;
    DeviceInfo device_info{ enabledLayers, enabledExtensions };

    ComputeShader shader{ device_info };

    Buffer2D bufA{ device_info, COLUMNS, LINES };
    Buffer2D bufB{ device_info, COLUMNS_B, LINES_B };
    Buffer2D bufC{ device_info, COLUMNS_C, LINES_C };

    bufA.init(fA);
    bufB.init(fB);

    bufA.print("A");
    bufB.print("B");

    shader.connect(bufA, 0); // connect buffer A to binding 0
    shader.connect(bufB, 1); // connect buffer B to binding 1
    shader.connect(bufC, 2); // connect buffer C to binding 2

    shader.constants.columnsA= COLUMNS;
    shader.constants.linesA = LINES; 
    shader.constants.columnsB = COLUMNS_B;
    shader.constants.linesB = LINES_B;
    shader.constants.operation = OPERATION; 

    shader.run(COLUMNS, LINES);
    bufC.print("A * B");

    bufA.clear();
    bufB.clear();
    bufC.clear();
    shader.clear();
    device_info.clear();
}

