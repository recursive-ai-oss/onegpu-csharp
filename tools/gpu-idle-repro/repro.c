// Pure Vulkan repro: upload with a transfer copy, idle the GPU for a while, run one compute dispatch, read back, compare on the CPU.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <vulkan/vulkan_core.h>

#define CK(x) do { VkResult r_ = (x); if (r_ != VK_SUCCESS) { printf("%s failed: %d\n", #x, r_); exit(2); } } while (0)
enum { N = 2 * 12 * 128 * 64, NOUT = 1536 };

static VkDevice dev; static VkPhysicalDevice phys; static VkPhysicalDeviceMemoryProperties mp;
static uint32_t pickType(uint32_t bits, VkMemoryPropertyFlags want, VkMemoryPropertyFlags avoid) {
    for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
        if ((bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & want) == want && !(mp.memoryTypes[i].propertyFlags & avoid)) return i;
    printf("no memory type\n"); exit(3);
}
static VkBuffer mkbuf(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags want, VkMemoryPropertyFlags avoid, void **map) {
    VkBufferCreateInfo bi = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, 0, 0, size, usage, VK_SHARING_MODE_EXCLUSIVE};
    VkBuffer b; CK(vkCreateBuffer(dev, &bi, 0, &b));
    VkMemoryRequirements rq; vkGetBufferMemoryRequirements(dev, b, &rq);
    VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, 0, rq.size, pickType(rq.memoryTypeBits, want, avoid)};
    VkDeviceMemory m; CK(vkAllocateMemory(dev, &ai, 0, &m)); CK(vkBindBufferMemory(dev, b, m, 0));
    if (map) CK(vkMapMemory(dev, m, 0, size, 0, map));
    return b;
}

int main(int argc, char **argv) {
    int iters = argc > 1 ? atoi(argv[1]) : 300;
    int sleepMs = argc > 2 ? atoi(argv[2]) : 20;
    int warm = argc > 3 ? atoi(argv[3]) : 0;      // 1: run one throw-away dispatch right before the measured one
    const char *spv = argc > 4 ? argv[4] : "min.spv";
    int devIdx = argc > 5 ? atoi(argv[5]) : 0;
    VkApplicationInfo app = {VK_STRUCTURE_TYPE_APPLICATION_INFO, 0, "repro", 1, "repro", 1, VK_API_VERSION_1_1};
    VkInstanceCreateInfo ici = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, 0, 0, &app};
    VkInstance inst; CK(vkCreateInstance(&ici, 0, &inst));
    uint32_t np = 8; VkPhysicalDevice pds[8]; CK(vkEnumeratePhysicalDevices(inst, &np, pds));
    phys = pds[devIdx];
    VkPhysicalDeviceProperties pp; vkGetPhysicalDeviceProperties(phys, &pp);
    printf("device: %s (driver %u)\n", pp.deviceName, pp.driverVersion);
    vkGetPhysicalDeviceMemoryProperties(phys, &mp);
    float prio = 1; uint32_t nq = 16; VkQueueFamilyProperties qf[16]; vkGetPhysicalDeviceQueueFamilyProperties(phys, &nq, qf);
    uint32_t fam = 0; for (uint32_t i = 0; i < nq; i++) if (qf[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { fam = i; break; }
    VkDeviceQueueCreateInfo qi = {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, 0, 0, fam, 1, &prio};
    VkDeviceCreateInfo dci = {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, 0, 0, 1, &qi};
    CK(vkCreateDevice(phys, &dci, 0, &dev));
    VkQueue q; vkGetDeviceQueue(dev, fam, 0, &q);

    void *stg, *rb;
    VkBuffer bStage = mkbuf(N * 4, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &stg);
    VkBuffer bRb = mkbuf(NOUT * 4, VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | (getenv("CACHED") ? VK_MEMORY_PROPERTY_HOST_CACHED_BIT : 0), 0, &rb);
    VkBuffer bIn = mkbuf(N * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0, 0);
    VkBuffer bOut = mkbuf(NOUT * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0, 0);

    FILE *f = fopen(spv, "rb"); if (!f) { printf("no spv\n"); return 4; }
    static uint32_t code[8192]; size_t nb = fread(code, 1, sizeof code, f); fclose(f);
    VkShaderModuleCreateInfo smi = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, 0, 0, nb, code};
    VkShaderModule sm; CK(vkCreateShaderModule(dev, &smi, 0, &sm));
    VkDescriptorSetLayoutBinding lb[2] = {{0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, 0}, {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, 0}};
    VkDescriptorSetLayoutCreateInfo dli = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, 0, 0, 2, lb};
    VkDescriptorSetLayout dl; CK(vkCreateDescriptorSetLayout(dev, &dli, 0, &dl));
    static uint8_t push[256]; size_t pushLen = 0; { const char *pp2 = getenv("PUSH"); if (pp2) { FILE *pf = fopen(pp2, "rb"); pushLen = fread(push, 1, sizeof push, pf); fclose(pf); } }
    VkPushConstantRange pcr = {VK_SHADER_STAGE_COMPUTE_BIT, 0, (uint32_t)pushLen};
    VkPipelineLayoutCreateInfo pli = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, 0, 0, 1, &dl, pushLen ? 1u : 0u, &pcr};
    VkPipelineLayout pl; CK(vkCreatePipelineLayout(dev, &pli, 0, &pl));
    VkComputePipelineCreateInfo cpi = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cpi.stage = (VkPipelineShaderStageCreateInfo){VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, 0, 0, VK_SHADER_STAGE_COMPUTE_BIT, sm, getenv("ENTRY") ? getenv("ENTRY") : "main"};
    cpi.layout = pl;
    VkPipeline pipe; CK(vkCreateComputePipelines(dev, 0, 1, &cpi, 0, &pipe));
    VkDescriptorPoolSize ps = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2};
    VkDescriptorPoolCreateInfo dpi = {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, 0, 0, 1, 1, &ps};
    VkDescriptorPool dp; CK(vkCreateDescriptorPool(dev, &dpi, 0, &dp));
    VkDescriptorSetAllocateInfo dai = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, 0, dp, 1, &dl};
    VkDescriptorSet ds; CK(vkAllocateDescriptorSets(dev, &dai, &ds));
    VkDescriptorBufferInfo bin = {bIn, 0, VK_WHOLE_SIZE}, bout = {bOut, 0, VK_WHOLE_SIZE};
    VkWriteDescriptorSet wr[2] = {{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, 0, ds, 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 0, &bin}, {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, 0, ds, 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 0, &bout}};
    vkUpdateDescriptorSets(dev, 2, wr, 0, 0);

    VkCommandPoolCreateInfo cpci = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, 0, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, fam};
    VkCommandPool cp; CK(vkCreateCommandPool(dev, &cpci, 0, &cp));
    VkCommandBufferAllocateInfo cbai = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, 0, cp, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 2};
    VkCommandBuffer cbs[2]; CK(vkAllocateCommandBuffers(dev, &cbai, cbs));
    VkFenceCreateInfo fci = {VK_STRUCTURE_TYPE_FENCE_CREATE_INFO}; VkFence fence; CK(vkCreateFence(dev, &fci, 0, &fence));

    static float x[N]; static float want[NOUT];
    uint32_t s = 12345; int bad = 0, badDispatches = 0, warmBad = 0;
    for (int it = 0; it < iters; it++) {
        for (int i = 0; i < N; i++) { s = s * 1664525u + 1013904223u; x[i] = -0.5f + 1.5f * ((s >> 8) / 16777216.f); }
        for (int o = 0; o < NOUT; o++) { int base = (o / 64) * 8192 + (o % 64); float a = x[base]; for (int r = 1; r < 128; r++) if (x[base + r * 64] < a) a = x[base + r * 64]; want[o] = a; }
        memcpy(stg, x, sizeof x);

        // 1. upload: staging -> device-local buffer, made visible to compute reads with a barrier, then wait for the queue to go idle.
        VkCommandBufferBeginInfo bi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, 0, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
        CK(vkResetCommandBuffer(cbs[0], 0)); CK(vkBeginCommandBuffer(cbs[0], &bi));
        VkBufferCopy cr = {0, 0, N * 4}; vkCmdCopyBuffer(cbs[0], bStage, bIn, 1, &cr);
        VkMemoryBarrier mb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER, 0, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT};
        vkCmdPipelineBarrier(cbs[0], VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, 0, 0, 0);
        CK(vkEndCommandBuffer(cbs[0]));
        VkSubmitInfo si = {VK_STRUCTURE_TYPE_SUBMIT_INFO, 0, 0, 0, 0, 1, &cbs[0]};
        CK(vkResetFences(dev, 1, &fence)); CK(vkQueueSubmit(q, 1, &si, fence)); CK(vkWaitForFences(dev, 1, &fence, VK_TRUE, ~0ull));
        CK(vkQueueWaitIdle(q));
        if (sleepMs > 0) usleep(sleepMs * 1000);

        // 2. one dispatch (optionally preceded by a throw-away one in the same submission), then copy the result to host memory.
        CK(vkResetCommandBuffer(cbs[1], 0)); CK(vkBeginCommandBuffer(cbs[1], &bi));
        vkCmdBindPipeline(cbs[1], VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
        vkCmdBindDescriptorSets(cbs[1], VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &ds, 0, 0);
        if (pushLen) vkCmdPushConstants(cbs[1], pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, (uint32_t)pushLen, push);
        vkCmdDispatch(cbs[1], getenv("NGROUPS") ? atoi(getenv("NGROUPS")) : 6, 1, 1);
        VkMemoryBarrier mb2 = {VK_STRUCTURE_TYPE_MEMORY_BARRIER, 0, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT};
        vkCmdPipelineBarrier(cbs[1], VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &mb2, 0, 0, 0, 0);
        VkBufferCopy cr2 = {0, 0, NOUT * 4}; vkCmdCopyBuffer(cbs[1], bOut, bRb, 1, &cr2);
        VkMemoryBarrier mb3 = {VK_STRUCTURE_TYPE_MEMORY_BARRIER, 0, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT};
        vkCmdPipelineBarrier(cbs[1], VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &mb3, 0, 0, 0, 0);
        CK(vkEndCommandBuffer(cbs[1]));
        if (warm) { // run the dispatch once before, so the GPU is awake
            VkSubmitInfo sw = {VK_STRUCTURE_TYPE_SUBMIT_INFO, 0, 0, 0, 0, 1, &cbs[1]};
            CK(vkResetFences(dev, 1, &fence)); CK(vkQueueSubmit(q, 1, &sw, fence)); CK(vkWaitForFences(dev, 1, &fence, VK_TRUE, ~0ull));
            int nw0 = 0; uint32_t *g0 = rb, *w0 = (uint32_t*)want; for (int o = 0; o < NOUT; o++) if (g0[o] != w0[o]) nw0++;
            if (nw0) { warmBad++; }
            memset(rb, 0xAB, NOUT * 4);
        }
        VkSubmitInfo s2 = {VK_STRUCTURE_TYPE_SUBMIT_INFO, 0, 0, 0, 0, 1, &cbs[1]};
        CK(vkResetFences(dev, 1, &fence)); CK(vkQueueSubmit(q, 1, &s2, fence)); CK(vkWaitForFences(dev, 1, &fence, VK_TRUE, ~0ull));
        int nw = 0; uint32_t *g = rb, *w = (uint32_t*)want; int fi = -1;
        for (int o = 0; o < NOUT; o++) if (g[o] != w[o]) { if (fi < 0) fi = o; nw++; }
        if (nw) { badDispatches++; bad += nw; printf("iter %d: %d wrong, first idx %d got 0x%08X want 0x%08X\n", it, nw, fi, g[fi], w[fi]); }
    }
    printf("warm-up dispatch wrong: %d\n", warmBad); printf("RESULT device=%s iters=%d sleep=%dms warm=%d: %d bad dispatches, %d wrong values\n", pp.deviceName, iters, sleepMs, warm, badDispatches, bad);
    return badDispatches != 0;
}
